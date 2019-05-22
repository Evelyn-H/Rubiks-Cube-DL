#!/usr/local_rwth/bin/zsh

### COMMANDS to remember:
#  - shh login:     ssh -l ppxxxxxx login18-g-1.hpc.itc.rwth-aachen.de
#  - submit job:    sbatch <script>.sh
#  - list jobs:     sacct
#  - list gpu's:    nvidia-smi


### CONFIGURATION

# memory
#SBATCH --mem-per-cpu=16G

# job name
#SBATCH --job-name=Optimal

# declare the merged STDOUT/STDERR file
#commented SBATCH --output=output/output.%J.txt

# max runing time
#SBATCH --time=48:00:00


### SCRIPT TO RUN

# enter git repo folder
cd ~/Rubiks-Cube-DL/optimal

# run file through pipenv
# (makes sure dependencies are all there)

optiqtm <scrambles-big-2.txt |tee results-big-2
