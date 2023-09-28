import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gene_df', help='path to gene annot csv with columns gene id, chr, start, end')
parser.add_argument('--CAGE_tracks', help='tracks to extract from output')
args = parser.parse_args()

## HARD CODED VARIABLES
prefix = '/home/s1mi/borzoi_tutorial/'
params_file = prefix + 'params_pred.json'
targets_file = prefix + 'targets_human.txt' #Subset of targets_human.txt

import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import baskerville
from baskerville import seqnn
from baskerville import dna
from baskerville import gene as bgene

import json

import pysam

#### LOAD PARSL PARAMETERS
import parsl
from parsl import python_app
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints

run_dir = "/home/s1mi/Github/deep-learning-in-genomics/posts/2023-09-25-borzoi-CAGE-TEST"
user_opts = {
    "worker_init":      f"module load conda; conda activate borzoi; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand" , # specify any PBS options here, like filesystems
    "account":          "AIHPC4EDU",
    "queue":            "preemptable",
    "walltime":         "01:30:00",
    "nodes_per_block":  1, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    "cpus_per_node": 64, # Up to 64 with multithreading
    "available_accelerators": 4, # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.  
}

checkpoints = get_all_checkpoints(run_dir)
print("Found the following checkpoints: ", checkpoints)

config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                available_accelerators=user_opts["available_accelerators"], # if this is set, it will override other settings for max_workers if set
                cores_per_worker=user_opts["cores_per_worker"],
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                    # Which launcher to use?  Check out the note below for some details.  Try MPI first!
                    # launcher=GnuParallelLauncher(),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1, # Can increase more to have more parallel jobs
                    cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"]
                ),
            ),
        ],
        checkpoint_files = checkpoints,
        run_dir=run_dir,
        checkpoint_mode = 'task_exit',
        retries=2,
        app_cache=True,
)

parsl.load(config)
print("Loaded parsl parameters for one task per GPU")



    
# Predict tracks
@python_app
def extract_predictions(models, fasta_open, chrom, start, end):
    from borzoi_helpers import process_sequence, predict_tracks
    sequence_one_hot = process_sequence(fasta_open, chrom, start, end)
    y = predict_tracks(models, sequence_one_hot)
    return np.average(y[..., 8174:8178,track_index])


start = time.perf_counter()

#Model configuration


seq_len = 524288
n_folds = 4       #To use only one model fold, change to 'n_folds = 1'
rc = True         #Average across reverse-complement prediction
transcriptome = bgene.Transcriptome(prefix + 'gencode41_basic_nort.gtf')
#Read model parameters

with open(params_file) as params_open :
    
    params = json.load(params_open)
    
    params_model = params['model']
    params_train = params['train']

#Read targets

targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

#Create local index of strand_pair (relative to sliced targets)
if rc :
    strand_pair = targets_df.strand_pair
    
    target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
    slice_pair = np.array([
        target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
    ], dtype='int32')

#Initialize model ensemble

models = []
for fold_ix in range(n_folds) :
    
    model_file = prefix + "saved_models/f" + str(fold_ix) + "/model0_best.h5"

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    if rc :
        seqnn_model.strand_pair.append(slice_pair)
    seqnn_model.build_ensemble(rc, '0')
    
    models.append(seqnn_model)


    #Initialize fasta sequence extractor

fasta_open = pysam.Fastafile(prefix + 'hg38.fa')

targets_df['local_index'] = np.arange(len(targets_df))


# Predict on input gene list

import sys
genes_file = args.gene_df
tracks = args.CAGE_tracks
track_index = [int(track) for track in tracks.split(",")]
gene_df = pd.read_csv(genes_file, header=None)
#gpus = tf.config.experimental.list_physical_devices('GPU')



app_futures = []
for index, row in gene_df.iterrows():
    chrom = "chr" + str(row[1])
    center_pos = row[2]
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    #Determine output sequence start
    seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
    seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

    app_futures.append(extract_predictions(models, fasta_open, chrom, start, end))

gene_df["predictions"] = [p.result() for p in app_futures]
gene_df.to_csv("gene_predictions.csv", header=False, index=False)
 