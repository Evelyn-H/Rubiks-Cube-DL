#!/usr/local_rwth/bin/zsh

# ask for 10 GB memory
#SBATCH --mem-per-cpu=4G
# name the job
#SBATCH --job-name=RubiksDL
# declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt

# GPU
#SBATCH --gres=gpu:1

### Change to the work directory
cd ~/Rubiks-Cube-DL/
pipenv shell

### Execute your application
python train.py --ini ini/cube2x2-zero-goal-d200.ini -n run_${LSB_JOBID}
