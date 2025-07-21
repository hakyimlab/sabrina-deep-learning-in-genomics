import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
import os
import time
import bisect

def reference_epigenome_matrix(track, chr, genes_df, reference_dir):
    reference_file = f'{reference_dir}/{chr}_genes.h5'
    with h5py.File(reference_file, 'r') as ref:
        rows = []
        for gene in genes_df['gene']:
            founder_predictions = ref[gene][:, 446:450, track]
            rows.append(founder_predictions)
        ref_matrix = np.stack(rows, axis=0)
        ref_tensor = tf.reduce_mean(tf.convert_to_tensor(ref_matrix, dtype=tf.float32), axis=2)
        return ref_tensor
    

def probabilities_matrix(chr, genes_df, individuals, probabilities_dir):
    probabilities_file = f'{probabilities_dir}/{chr}_probabilities.h5'
    with h5py.File(probabilities_file, 'r') as prob:
        positions = prob['positions'][:]
        population_prob = []
        indices = []
        for tss in genes_df['tss']:
            index = bisect.bisect_left(positions, tss)
            if index == 0:
                indices += [0,0]
            elif index == len(positions):
                indices += [index-1,index-1]
            else: # 0 < index < len(positions)
                indices += [index-1, index]
        for sample in individuals:
            dataset = prob[sample][:]
            sample_prob = tf.convert_to_tensor(dataset[indices], dtype = tf.float32)
            sample_prob = tf.reshape(sample_prob, (-1, 2, sample_prob.shape[1]))
            sample_prob = tf.reduce_mean(sample_prob, axis=1)
            population_prob.append(sample_prob)
        prob_tensor = tf.stack(population_prob, axis=0)
        return prob_tensor
    
    
def predict_epigenome(track, chr, genes_df, individuals, output_file, reference_dir, probabilities_dir):
    ref_tensor = reference_epigenome_matrix(chr, genes_df, track, reference_dir)
    prob_tensor = probabilities_matrix(chr, genes_df, individuals, probabilities_dir)
    epigenome_tensor = tf.einsum('ijk,jk->ij', prob_tensor, ref_tensor)
    epigenome_df = pd.DataFrame(epigenome_tensor.numpy(), columns=genes_df['gene'], index = individuals)
    epigenome_df.to_csv(output_file)
