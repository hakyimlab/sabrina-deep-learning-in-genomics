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

run_dir = "/home/s1mi/Github/deep-learning-in-genomics/posts/2023-08-31-Br-personalized-prediction-on-more-genes"
user_opts = {
    "worker_init":      f"module load conda; conda activate ml-python; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand" , # specify any PBS options here, like filesystems
    "account":          "AIHPC4EDU",
    "queue":            "preemptable",
    "walltime":         "01:00:00",
    "nodes_per_block":  2, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    "cpus_per_node": 8, # Up to 64 with multithreading
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

#### GET LIST OF GENES
with open(f"{run_dir}/gene_list.txt", "r") as file:
    gene_list = file.read().splitlines()
print(len(gene_list), "genes with enformer predictions")


#### INITIALIZE H5 FILE WITH INDEX NAMES (INDIVIDUALS)
import h5py
import pandas as pd
project_dir = "/home/s1mi/Br_predictions/predictions_folder/personalized_Br_selected_genes"

obs_gene_expr = pd.read_csv("/home/s1mi/enformer_rat_data/expression_data/Brain.rn7.expr.tpm.bed", sep="\t", nrows=1)
with h5py.File(f"{project_dir}/selected_genes_mouse_and_human_predictions.h5", "w") as file:
    file.attrs["index"] = obs_gene_expr.columns.to_list()[4:]

#### JOIN WITH ENFORMER PREDICTIONS FUNCTION
@python_app
def collect_predictions(gene):
    ### IMPORT MODULES
    import pandas as pd
    import numpy as np
    import h5py
    project_dir = "/home/s1mi/Br_predictions/predictions_folder/personalized_Br_selected_genes"
    predictions_dir = f"{project_dir}/predictions_2023-09-02/enformer_predictions"
    obs_gene_expr = pd.read_csv("/home/s1mi/enformer_rat_data/expression_data/Brain.rn7.expr.tpm.bed", sep="\t", header=0, index_col='gene_id')
    annot_df = pd.read_csv("/home/s1mi/enformer_rat_data/annotation/rn7.gene.txt", sep="\t", header= 0, index_col='geneId')

    ### INITIALIZE EXPRESSION MATRIX WITH OBSERVED DATA
    expr_df = pd.DataFrame({"observed": obs_gene_expr.loc[gene][3:].astype("float32")})    

    ### READ PREDICTIONS
    gene_annot = annot_df.loc[gene]
    interval = f"chr{gene_annot['chromosome']}_{gene_annot['tss']}_{gene_annot['tss']}"
    human_predicted = []
    mouse_predicted = []
    for individual in expr_df.index:
        with h5py.File(f"{predictions_dir}/{individual}/haplotype0/{interval}_predictions.h5", "r") as input_file:
            human_prediction = input_file["human"][446:450, 4980]
            mouse_prediction = input_file["mouse"][446:450, 1300]
            human_predicted.append(np.average(human_prediction))
            mouse_predicted.append(np.average(mouse_prediction))
    expr_df["human predicted"] = human_predicted
    expr_df["mouse predicted"] = mouse_predicted
    print(expr_df.dtypes)

    ### WRITE TO h5
    with h5py.File(f"{project_dir}/selected_genes_mouse_and_human_predictions.h5", "a") as output_file:
        output_file[gene] = expr_df


#### JOIN CONCURRENTLY ACROSS GENES
app_futures = []
for gene in gene_list:
    app_futures.append(collect_predictions(gene))
print("Waiting for apps to finish executing...")
exec_futures = [q.result() for q in app_futures]
print("Finished writing expression matrices for", len(gene), "genes")