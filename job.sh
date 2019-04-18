#!/usr/local_rwth/bin/zsh

# ask for 10 GB memory
#SBATCH --mem-per-cpu=4G
# name the job
#SBATCH --job-name=RubiksDL
# declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt

# GPU
#SBATCH --gres=gpu:1

# time
#SBATCH --time=00:59:00

### Change to the work directory
export PATH=$PATH:~/.local/bin
module load python/3.6.0

cd ~/Rubiks-Cube-DL/

### Execute your application
pipenv run python train.py --ini ini/cube2x2-zero-goal-d200.ini -n run_${LSB_JOBID}
