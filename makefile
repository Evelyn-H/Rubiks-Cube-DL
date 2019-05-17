.PHONY: default connect forward board output plots submit debug mount


default: ;

connect:
	ssh -l pp563877 login18-g-1.hpc.itc.rwth-aachen.de

forward:
	ssh -l pp563877 login18-g-1.hpc.itc.rwth-aachen.de -L 54321:localhost:54321

board:
	pipenv run tensorboard --logdir=runs/ --port 54321

output:
	tail -f -n 50 output/$(shell ls output -t | head -n1)

plots:
	pipenv run python solver.py --env cube$(e)x$(e) --cuda --plot plots/$(name) --model saves/$(model) --max-steps 10000 --samples 20 --max-depth 20

debug:
	pipenv run python train_debug.py --env cube$(e)x$(e) --model saves/$(model) --output plots/$(name)

submit:
	sbatch --output="output/output.%J.$(shell date +"%Y-%m-%d_%H-%M").txt" "$(job).sh"

mount:
	gvfs-mount sftp://pp563877@login18-g-1.hpc.itc.rwth-aachen.de/home/pp563877
