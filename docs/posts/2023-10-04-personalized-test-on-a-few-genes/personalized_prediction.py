import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gene_df', help='path to gene annot csv with columns gene id, chr, start, end')
parser.add_argument('--fasta_file')
parser.add_argument('--output_dir', help='folder to write predictions')
parser.add_argument('--vcf_dir', help='folder with vcfs split by chromosome')
parser.add_argument('--individuals_file', help='list of individuals to query from vcf')
parser.add_argument('--model_dir', help='path to borzoi models')
args = parser.parse_args()


import os
import h5py
import numpy as np
import pysam
import tensorflow as tf
from multiprocessing import Process
from borzoi_helpers import *
import pandas as pd

gene_df = pd.read_csv(args.gene_df, header=0)
output_dir = args.output_dir
vcf_dir = args.vcf_dir
fasta_open = pysam.Fastafile(args.fasta_file)
model_dir = args.model_dir
with open(args.individuals_file, "r") as f:
    individuals = f.read().splitlines()

def enformer_predict_on_region(row, vcf_dir, samples, output_dir, models, fasta_open):
    interval_object = {'chr': 'chr' + str(row["chromosome_name"]), 'start': row["transcript_start"], 'end': row["transcript_end"]}
    target_interval = resize(interval_object)
    path_to_vcf = os.path.join(vcf_dir, f"ALL.{interval_object['chr']}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz")
    sequence_one_hot_ref = process_sequence(fasta_open, target_interval["chr"], target_interval["start"], target_interval["end"])
    print("Extract Reference Sequence:", target_interval)
    vcf_chr = cyvcf2.cyvcf2.VCF(path_to_vcf, samples=samples)
    variants_array = find_variants_in_vcf_file(vcf_chr, target_interval, samples, mode="phased")
    mapping_dict = create_mapping_dictionary(variants_array, samples, target_interval["start"])
    print("Create Personalized Sequence for", len(samples), "Individuals")
    samples_variants_encoded = replace_variants_in_reference_sequence(sequence_one_hot_ref, mapping_dict, samples)
    for sample in samples:
        print("Predicting on Sample:", sample)
        sample_input = samples_variants_encoded[sample]
        # print("Predict on Sequence:", sample_input)
        sample_predictions = predict_on_sequence(models, sample_input)
        sample_dir = os.path.join(output_dir, sample)
        output_path = os.path.join(sample_dir, f'{interval_object["chr"]}_{interval_object["start"]}_{interval_object["end"]}_predictions.h5')
        if not os.path.exists(sample_dir): os.makedirs(sample_dir, exist_ok=True)
        print("Write Predictions to", output_path)
        with h5py.File(output_path, "w") as hf:
            for hap in sample_predictions.keys():
                hf[hap]= np.squeeze(sample_predictions[hap], axis=0)

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Available GPUs:", gpus)
models = get_model(model_dir)
sample_batches = list(batch(individuals, n = 5))
for index, row in gene_df.iterrows():
    for samples in sample_batches:
        enformer_predict_on_region(row, vcf_dir, samples, output_dir, models, fasta_open)

