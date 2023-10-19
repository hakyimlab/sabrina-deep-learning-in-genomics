import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import baskerville
from baskerville import seqnn
from baskerville import dna

import pysam
import cyvcf2

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


def find_variants_in_vcf_file(cyvcf2_object, interval_object, samples, mode="phased"):
    start = max(interval_object['start'], 0)
    query = f"{interval_object['chr']}:{start}-{interval_object['end']}"
    variants_dictionary = {}
    variants_dictionary['chr'] = interval_object['chr']
    variants_dictionary['positions'] = tuple(variant.POS for variant in cyvcf2_object(query))
    if mode == 'phased':
        delim = '|'
    elif mode == 'unphased':
        delim = '/'
    for i, sample in enumerate(samples):
        if sample in cyvcf2_object.samples:
            variants_dictionary[sample] = tuple([variant.genotypes[i][0:2], variant.gt_bases[i].split(delim)] for variant in cyvcf2_object(query))
    return variants_dictionary

def resize(region, seq_len=524288):
    center_bp = (region['end'] + region['start']) // 2
    start = center_bp - seq_len // 2
    end = center_bp + seq_len // 2
    return {"chr": region['chr'], "start": start, "end": end}

# def replace_variants_in_reference_sequence(query_sequences_encoded, mapping_dict, samples):
#     import copy
#     import numpy as np
#     positions = mapping_dict['positions']
#     variant_encoded = {}
#     for sample in samples:
#         haplotype1_encoded = np.copy(query_sequences_encoded)
#         haplotype2_encoded = np.copy(query_sequences_encoded)
#         for i, position in enumerate(positions):
#             haplotype1_encoded[position] = mapping_dict[sample]["haplotype1"][i]
#             haplotype2_encoded[position] = mapping_dict[sample]["haplotype2"][i]
#         variant_encoded[sample] = {"haplotype1": haplotype1_encoded, "haplotype2": haplotype2_encoded}
#     return variant_encoded

def mutate_sequence(sequence_one_hot, start, poses, alts):
    
    #Induce mutation(s)
    sequence_one_hot_mut = np.copy(sequence_one_hot)

    for pos, alt in zip(poses, alts) :
        alt_ix = -1
        if alt == 'A' :
            alt_ix = 0
        elif alt == 'C' :
            alt_ix = 1
        elif alt == 'G' :
            alt_ix = 2
        elif alt == 'T' :
            alt_ix = 3

        sequence_one_hot_mut[pos-start-1] = 0.
        sequence_one_hot_mut[pos-start-1, alt_ix] = 1.
    return sequence_one_hot_mut

def replace_variants_in_reference_sequence(sequence_one_hot_ref, variants_array, interval_start, samples):
    poses = variants_array['positions']
    variant_encoded = {}
    for sample in samples:
        alts_1 = [variants_array[sample][i][1][0] for i in range(len(poses))]
        alts_2 = [variants_array[sample][i][1][1] for i in range(len(poses))]
        haplotype1_encoded = mutate_sequence(sequence_one_hot_ref, interval_start, poses, alts_1)
        haplotype2_encoded = mutate_sequence(sequence_one_hot_ref, interval_start, poses, alts_2)
        variant_encoded[sample] = {'haplotype1': haplotype1_encoded, 'haplotype2': haplotype2_encoded}
    return variant_encoded

def get_model(model_dir):
    import os
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
    params_file = os.path.join(model_dir, 'params_pred.json') 
    targets_file = os.path.join(model_dir, 'targets_human.txt') 

    n_folds = 1       #To use only one model fold, change to 'n_folds = 1'
    rc = True         #Average across reverse-complement prediction
    #Read model parameters

    with open(params_file) as params_open :
        
        params = json.load(params_open)
        
        params_model = params['model']

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
        
        model_file = os.path.join(model_dir, "saved_models/f" + str(fold_ix) + "/model0_best.h5")

        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, 0)
        seqnn_model.build_slice(target_index)
        if rc :
            seqnn_model.strand_pair.append(slice_pair)
        seqnn_model.build_ensemble(rc, '0')
        
        models.append(seqnn_model)
    return models


def predict_on_sequence(models, sample_input):
    prediction_output = {}
    for haplotype, sequence_encoding in sample_input.items():
        prediction = predict_tracks(models, sequence_encoding)
        prediction_output[haplotype] = prediction
    return prediction_output

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]