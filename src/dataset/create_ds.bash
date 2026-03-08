#!/bin/bash
#SBATCH -p small               
#SBATCH -n 47
#SBATCH --mem-per-cpu=3G
#SBATCH -t 3-00:00:00 
module purge
module load anaconda
export HF_HOME=""
module load gcc
module load cuda/12.4
module load gurobi

# Check system info
nvidia-smi
echo "CPU cores: $(nproc)"
echo "Memory available:"
free -h

conda --version
which python

conda activate JD_inventory

python create_dataset.py
