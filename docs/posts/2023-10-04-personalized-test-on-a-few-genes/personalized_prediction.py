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
from borzoi_helpers import *

import pandas as pd
gene_df = pd.read_csv(args.gene_df)
output_dir = args.output_dir
vcf_dir = args.vcf_dir
fasta_open = pysam.Fastafile(args.fasta_file)
model_dir = args.model_dir
with open(args.individuals_file, "r") as f:
    samples = f.read().splitlines()

def enformer_predict_on_region(target_interval, samples, path_to_vcf, output_dir, models, fasta_open):
    sequence_one_hot_ref = process_sequence(fasta_open, target_interval["chr"], target_interval["start"], target_interval["end"])
    vcf_chr = cyvcf2.cyvcf2.VCF(path_to_vcf, samples=samples)
    variants_array = find_variants_in_vcf_file(vcf_chr, target_interval, samples, mode="phased")
    mapping_dict = create_mapping_dictionary(variants_array, samples, target_interval["start"])
    samples_variants_encoded = replace_variants_in_reference_sequence(sequence_one_hot_ref, mapping_dict, samples)
    for sample in samples:
        sample_input = samples_variants_encoded[sample]
        sample_predictions = predict_on_sequence(models, sample_input)
        sample_dir = os.path.join(output_dir, sample)
        output_path = os.path.join(sample_dir, f'{target_interval["chr"]}_{target_interval["start"]}_{target_interval["end"]}_predictions.h5')
        if not os.path.exists(sample_dir): os.makedirs(sample_dir, exist_ok=True)
        with h5py.File(output_path, "w") as hf:
            for hap in sample_predictions.keys():
                hf[hap]= np.squeeze(sample_predictions[hap], axis=0)

models = get_model(model_dir)

for _, row in gene_df.iterrows():
    interval_object = {'chr': 'chr' + str(row["chromosome_name"]), 'start': row["transcript_start"], 'end': row["transcript_end"]}
    target_interval = resize(interval_object)
    path_to_vcf = os.path.join(vcf_dir, f"ALL.{interval_object['chr']}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz")
    
    enformer_predict_on_region(interval_object, samples, path_to_vcf, output_dir, models, fasta_open)



