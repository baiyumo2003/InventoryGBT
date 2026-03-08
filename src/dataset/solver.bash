#!/bin/bash
#SBATCH -p batch                # Use 'lowpri' if you want lower priority
#SBATCH -n 1024
#SBATCH --mem-per-cpu=1G
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

python concurent_solver.py
