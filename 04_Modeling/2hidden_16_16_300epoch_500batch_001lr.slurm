#!/bin/bash
#SBATCH --mail-user=xueyuan.jia@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --output=2hidden_16_16_300epoch_500batch_001lr.png.out
#SBATCH --error=2hidden_16_16_300epoch_500batch_001lr.png.err

module restore finalproj
source activate finalproj
~/.conda/envs/finalproj/bin/python 2hidden_16_16_300epoch_500batch_001lr.png.py