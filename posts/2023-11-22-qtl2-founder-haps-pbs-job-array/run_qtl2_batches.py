import pandas as pd
import time
import subprocess
import argparse


def qtl_command(batch_dir, output_file, n_cores=1):
    cmd = (
        'library(qtl2); '
        f'cross <- read_cross2("{batch_dir}/control.yaml"); '
        f'pr <- calc_genoprob(cross, error_prob = 0.01, cores = {n_cores}); '
        f'pr <- genoprob_to_alleleprob(pr); saveRDS(pr, "{output_file}")'
    )
    return cmd


p = argparse.ArgumentParser(description="Wrapper for R/qtl2 to calculate founder haplotype probabilities")
p.add_argument("--data_dir", help="chromosome input folder generated in make_qtl2_inputs")
p.add_argument("--output_dir", help="folder to output probabilities saved as rds")
p.add_argument("--work_dir", default="tmp-qtl2-founder-haps", help="Name of directory to write qtl2 input files")
p.add_argument("--cores", type=int, default=1, help="Number of cores to use when calculating probabilities")
args = p.parse_args()


geno = pd.read_csv(f'{args.data_dir}/geno.csv', index_col='id')
batch_size = 10
n_batches = len(geno.columns) // batch_size

for i in range(n_batches):
    tic = time.perf_counter()
    geno_df = geno.iloc[:, i:i+batch_size]
    samples = geno_df.columns.to_list()
    covar_df = pd.DataFrame({'id': samples, 'generations': [90] *  len(samples)})
    geno_df.to_csv(f'{args.work_dir}/geno.csv', index=True)
    covar_df.to_csv(f'{args.work_dir}/covar.csv', index=False)
    cmd = qtl_command(args.work_dir, f'{args.output_dir}/batch{i}_prob.rds', n_cores = 32)
    subprocess.run(f"R -e '{cmd}'", shell=True)
    toc = time.perf_counter()
    print("Batch:", i+1, "...", (toc-tic)/60, "minutes")

