#!/bin/bash
#SBATCH --job-name=main
#SBATCH --mail-user=tsharma2@uthsc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=ACF-UTHSC0001
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --output=main.o%j
#SBATCH --error=main.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01-00:00:00

###########################################

source activate env

echo "The environment has been activated."

python main.py cancer_type clinical_outcome_endpoint event_time_threshold target_minority_group features_count

echo "The execution has been done."



