module load conda
conda activate /lus/grand/projects/TFXcan/imlab/shared/software/conda_envs/enformer-predict-tools

python /home/s1mi/Github/shared_folder/enformer_pipeline/scripts/enformer_predict.py --parameters chr8_reference.json

