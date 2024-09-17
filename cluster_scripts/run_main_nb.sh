#!/bin/bash

#SBATCH --partition=shortq
#SBATCH --qos=shortq
#SBATCH -o "../logs/main_runs/main_run_out_%j.txt"
#SBATCH -e "../logs/main_runs/main_run_err_%j.txt" 
#SBATCH --time=2:00:00
#SBATCH --mem=64000

echo "======================"
echo $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NAME
echo $SLURM_JOB_PARTITION
echo $SLURM_NTASKS
echo $SLURM_NPROCS
echo $SLURM_JOB_ID
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NODELIST
echo $SLURM_CPUS_ON_NODE
echo "======================" 

## input folder, output folder, sample id, cellprofiler pipeline and dapi channel should be submitted to this script with  the --export flag
echo "Running job with the following parameters"
echo "experiment id: ${experiment_id}"
echo "output path: ${output_folder}"

if [ ! -d ${output_folder} ]; then
    mkdir -p ${output_folder}
fi

# module load Python/3.8.2-GCCcore-9.3.0
# export PATH="/home/bhaladik/.local/bin:$PATH"

source /nobackup/lab_boztug/projects/ccasey/miniconda3/etc/profile.d/conda.sh
conda activate nb_screening_post

annotation_folder=/nobackup/lab_ccri_bicu/internal/ccasey/projects/pop/nb/soren/neuroblastoma_screening/test_dataset/annotations/

python ../main.py ${experiment_id} ${output_folder} --expression_model ${expression_model} --viability_model none --relaxed_image_qc --no_morphology_well_exclusion --annotation_folder ${annotation_folder} --config_prefix nb_ --write_latent