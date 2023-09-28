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

