from libcube import mcts
from libcube import cubes
import solver


def solve_cube(cube_env, state, net, device, max_iterations):
    # tree = mcts.MCTS(env, cube_state, net, device=device)
    tree = mcts.Greedy(cube_env, state, net, device=device, max_iterations=max_iterations)

    solution = tree.search()
    return tree, solution


def solve_random_cubes(cube_env, scramble_depth, amount, max_iterations, net, device):
        solutions = []
        iterations_needed = []
        try:
            scramble = solver.generate_task(cube_env, scramble_depth)
            scrambled = cube_env.scramble(map(cube_env.action_enum, scramble))

            tree, solution = solve_cube(cube_env, scrambled, net, device, max_iterations)

            # check if it's actually solved
            if solution is not None:
                final_state, is_valid = solver.is_solution_valid(cube_env, scrambled, solution)
                if not is_valid:
                    print('INVALID SOLUTION RETURNED:', cube_env.render(final_state))
                    print('scramble:', scramble)
                    print('solution:', solution)

            solutions.append(solution)
            iterations_needed.append(len(tree) if solution else None)

        except Exception:
            pass

        return solutions, iterations_needed
