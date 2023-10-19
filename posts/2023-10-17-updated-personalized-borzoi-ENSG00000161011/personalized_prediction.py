import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--intervals_file', help='path to gene annot csv with columns gene id, chr, start, end')
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
from borzoi_helpers import *
import time


import pandas as pd
output_dir = args.output_dir
vcf_dir = args.vcf_dir
fasta_open = pysam.Fastafile(args.fasta_file)
model_dir = args.model_dir
with open(args.individuals_file, "r") as f:
    individuals = f.read().splitlines()
with open(args.intervals_file, "r") as f:
    intervals = f.read().splitlines()

def enformer_predict_on_region(interval_object, samples, path_to_vcf, output_dir, models, fasta_open):
    sequence_one_hot_ref = process_sequence(fasta_open, interval_object["chr"], interval_object["start"], interval_object["end"])
    vcf_chr = cyvcf2.cyvcf2.VCF(path_to_vcf, samples=samples)
    target_interval = resize(interval_object)
    variants_array = find_variants_in_vcf_file(vcf_chr, target_interval, samples, mode="phased")
    # mapping_dict = create_mapping_dictionary(variants_array, samples, target_interval["start"])
    # samples_variants_encoded = replace_variants_in_reference_sequence(sequence_one_hot_ref, mapping_dict, samples)
    samples_variants_encoded = replace_variants_in_reference_sequence(sequence_one_hot_ref, variants_array, target_interval["start"], samples=samples)
    for sample in samples:
        sample_input = samples_variants_encoded[sample]
        sample_predictions = predict_on_sequence(models, sample_input)
        sample_dir = os.path.join(output_dir, sample)
        output_path = os.path.join(sample_dir, f'{interval_object["chr"]}_{interval_object["start"]}_{interval_object["end"]}_predictions.h5')
        if not os.path.exists(sample_dir): os.makedirs(sample_dir, exist_ok=True)
        with h5py.File(output_path, "w") as hf:
            for hap in sample_predictions.keys():
                hf[hap]= np.squeeze(sample_predictions[hap], axis=0)

models = get_model(model_dir)
sample_batches = list(batch(individuals, n = 5))

for interval in intervals:
    split_interval = interval.split('_')
    interval_object = {'chr': split_interval[0], 'start': int(split_interval[1]), 'end': int(split_interval[2])}
    path_to_vcf = os.path.join(vcf_dir, f"ALL.{interval_object['chr']}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz")
    for i, samples in enumerate(sample_batches):
        tic = time.perf_counter()
        enformer_predict_on_region(interval_object, samples, path_to_vcf, output_dir, models, fasta_open)
        toc = time.perf_counter()
        print("Batch", i, "...", toc-tic, "seconds to predict")
