#!/bin/bash
#SBATCH --job-name=frag
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/mol-hrl/resource/log/job_%j.log
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=8

srun \
  --container-image=sungsahn0215/mol-opt:latest \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/mol-hrl:/mol-hrl" \
  --container-workdir="/mol-hrl/src" \
  bash ../script/${1}