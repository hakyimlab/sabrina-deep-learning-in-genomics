#!/bin/bash
#PBS -A AIHPC4EDU
#PBS -q preemptable
#PBS -l walltime=6:00:00
#PBS -l select=2:ncpus=64:ngpus=16
#PBS -l filesystems=home:eagle
#PBS -N borzoi_across_geuvadis_parallel

module load conda
conda activate borzoi
cd /home/s1mi/Github/deep-learning-in-genomics/posts/2023-10-20-borzoi-vs-enformer-personalized-prediction

python3 personalized_prediction.py \
--intervals_file intervals.txt \
--fasta_file /home/s1mi/borzoi_tutorial/hg38.fa \
--vcf_dir /grand/TFXcan/imlab/data/1000G/vcf_snps_only \
--individuals_file individuals.txt \
--model_dir /home/s1mi/borzoi_tutorial \
--output_dir /eagle/AIHPC4Edu/sabrina/borzoi-personalized-predictions