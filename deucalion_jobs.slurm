#!/bin/bash
#SBATCH --job-name=plantaos_lstm_train
#SBATCH --account=2025.00020.AIVLAB.DEUCALION
#SBATCH --partition=acc_a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/job1_lstm_%j.out

echo "Starting PlantaOS Job on Deucalion A100..."
python3 plantaos.py train --model c3_lbm --epochs 500 --batch-size 256
