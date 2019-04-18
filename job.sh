#!/usr/bin/env zsh

### Job name
#BSUB -J RubiksDL

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o RubiksDL.%J.%I

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 1:00

### Request memory you need for your job in TOTAL in MB
#BSUB -M 4096

### GPU
#BSUB -gpu -
#BSUB -R pascal

### Change to the work directory
cd ~/Rubiks-Cube-DL/
pipenv shell

### Execute your application
python train.py --ini ini/cube2x2-zero-goal-d200.ini -n run_${LSB_JOBID}
