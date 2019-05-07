.PHONY: default connect forward board output plots submit


default: ;

connect:
	ssh -l pp563877 login18-g-1.hpc.itc.rwth-aachen.de

forward:
	ssh -l pp563877 login18-g-1.hpc.itc.rwth-aachen.de -L 54321:localhost:54321

board:
	tensorboard --logdir=runs/ --port 54321

output:
	cd output && tail -f -n 50 $(ls -t | head -n1)

plots:
	python solver.py --env cube3x3 --cuda --plot plots/ --model saves/$(model) --max-time 20 --samples 5

submit:
	sbatch --output="output/output.%J.$(date +"%Y-%m-%d_%H-%M").txt" ""$(job).sh"
