import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gene_df', help='path to gene annot csv with columns gene id, chr, start, end')
parser.add_argument('--tracks', help='tracks to extract from output')
args = parser.parse_args()

import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

print("On GPU:", tf.config.list_physical_devices('GPU'))

import baskerville
from baskerville import seqnn
from baskerville import dna
from baskerville import gene as bgene

import json

import pysam

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

import intervaltree
import pyBigWig

import gc

# Helper functions (prediction, attribution, visualization)

# Make one-hot coded sequence
def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # Extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna.dna_1hot(seq_dna)
    return seq_1hot


# Predict tracks
def predict_tracks(models, sequence_one_hot):

    predicted_tracks = []
    for fold_ix in range(len(models)):

        yh = models[fold_ix](sequence_one_hot[None, ...])[:, None, ...].astype(
            "float16"
        )

        predicted_tracks.append(yh)

    predicted_tracks = np.concatenate(predicted_tracks, axis=1)

    return predicted_tracks


# Helper function to get (padded) one-hot
def process_sequence(fasta_open, chrom, start, end, seq_len=524288):

    seq_len_actual = end - start

    # Pad sequence to input window size
    start -= (seq_len - seq_len_actual) // 2
    end += (seq_len - seq_len_actual) // 2

    # Get one-hot
    sequence_one_hot = make_seq_1hot(fasta_open, chrom, start, end, seq_len)

    return sequence_one_hot.astype("float32")

def inverse_transform(y_wt, track_scale, track_transform, clip_soft):

    y_wt_curr = np.array(np.copy(y_wt), dtype=np.float32)

    # undo scale
    y_wt_curr /= track_scale

    # undo soft_clip
    y_wt_curr_unclipped = (y_wt_curr - clip_soft) ** 2 + clip_soft

    unclip_mask_wt = y_wt_curr > clip_soft

    y_wt_curr[unclip_mask_wt] = y_wt_curr_unclipped[unclip_mask_wt]

    # undo sqrt
    y_wt_curr = y_wt_curr ** (1.0 / track_transform)

    return y_wt_curr

def expr_attr(y_wt, track_index, gene_slice):
    y_wt_curr = inverse_transform(y_wt, 0.01, 3./4., 384.)
    y_wt_track = np.mean(y_wt_curr[..., track_index], axis=(0, 1, 3))
    sum_wt = np.sum(y_wt_track[gene_slice])
    return sum_wt

start = time.perf_counter()

#Model configuration
prefix = '/home/s1mi/borzoi_tutorial/'
params_file = prefix + 'params_pred.json'
targets_file = prefix + 'targets_gtex.txt' #Subset of targets_human.txt

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

    
end = time.perf_counter()
print("Models loaded in", end - start, "seconds")

    #Initialize fasta sequence extractor

fasta_open = pysam.Fastafile(prefix + 'hg38.fa')

#Load splice site annotation

splice_df = pd.read_csv(prefix + 'gencode41_basic_protein_splice.csv.gz', sep='\t', compression='gzip')

print("len(splice_df) = " + str(len(splice_df)))

targets_df['local_index'] = np.arange(len(targets_df))


# Predict on input gene list

import sys
genes_file = args.gene_df
tracks = args.tracks
track_index = [int(track) for track in tracks.split(",")]
gene_df = pd.read_csv(genes_file, header=None)
tic = time.perf_counter()
predictions = []
for index, row in gene_df.iterrows():
    chrom = "chr" + str(row[1])
    search_gene = row[0]
    gene_start = row[2]
    gene_end = row[3]
    center_pos = (gene_start + gene_end) // 2
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    #print(chr, start, end)
    gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]
    try:
        gene = transcriptome.genes[gene_keys[0]]
    except:
        predictions.append(np.nan)
        continue
    #Determine output sequence start
    seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
    seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

    gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)
    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)
    
    y_wt = predict_tracks(models, sequence_one_hot_wt)
    y_wt_curr = inverse_transform(y_wt, 0.01, 3./4., 384.)
    predictions.append(expr_attr(y_wt, track_index, gene_slice))
    if (index+1) % 100 == 0:
        toc = time.perf_counter()
        print(index+1, "genes out of", gene_df.shape[0], "done...", toc-tic, "seconds")
        tic = toc

gene_df[4] = predictions
gene_df.to_csv("gene_predictions.csv", header=False, index=False)
end = time.perf_counter()
print("Total GPU mins for predictions:", end-start // 60, "minutes")
 