#!/usr/bin/env python3
"""
Ad-hoc utility to analyze trained model and various training process details
"""
import argparse
import logging

import torch
import torch.nn.functional as F
import matplotlib as mpl; mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from libcube import cubes
from libcube import model

import optimal_lengths


log = logging.getLogger("train_debug")


# How many data to generate for plots
MAX_DEPTH = 30
ROUND_COUNTS = 500
# debug params
#MAX_DEPTH = 5
#ROUND_COUNTS = 2


def gen_states(cube_env, max_depth, round_counts):
    """
    Generate random states of various scramble depth
    :param cube_env: CubeEnv instance
    :return: list of list of (state, correct_action_index) pairs
    """
    assert isinstance(cube_env, cubes.CubeEnv)

    result = [[] for _ in range(max_depth)]
    for _ in range(round_counts):
        data = cube_env.scramble_cube(max_depth, return_inverse=True)
        for depth, state, inv_action in data:
            result[depth-1].append((state, inv_action.value))
    return result


if __name__ == "__main__":
    sns.set()

    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-o", "--output", required=True, help="Output prefix for plots")
    args = parser.parse_args()

    cube_env = cubes.get(args.env)
    log.info("Selected cube: %s", cube_env)
    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum))
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.eval()
    log.info("Network loaded from %s", args.model)

#    model.make_train_data(cube_env, net, device='cpu', batch_size=10, scramble_depth=2, shuffle=False)

    states_by_depth = gen_states(cube_env, max_depth=MAX_DEPTH, round_counts=ROUND_COUNTS)
    # for idx, states in enumerate(states_by_depth):
    #     log.info("%d: %s", idx, states)

    # flatten returned data
    data = []
    for depth, states in enumerate(states_by_depth):
        for s, inv_action in states:
            data.append((depth+1, s, inv_action))
    depths, states, inv_actions = map(list, zip(*data))

    # process states with net
    enc_states = model.encode_states(cube_env, states)
    enc_states_t = torch.tensor(enc_states)
    policy_t, value_t = net(enc_states_t)
    value_t = value_t.squeeze(-1)
    value = value_t.cpu().detach().numpy()
    policy = F.softmax(policy_t, dim=1).cpu().detach().numpy()

    # plot value per depth of scramble
    optimal = optimal_lengths.all
    straight_line = [-d for d in depths]
    # plot = sns.lineplot(depths, value, ci="sd")
    plot = sns.lineplot(depths, value, ci=None)
    plot = sns.scatterplot(depths, value, alpha=0.02, edgecolors='none')
    sns.lineplot([d[0] for d in optimal], [-d[1] for d in optimal], ci="sd", ax=plot)
    plot.plot(depths, straight_line, scaley=False)
    plot.set_xlim(0, MAX_DEPTH)
    plot.set_ylim(-25, 0)
    plot.set_title("Values per depths")
    plot.set_xlabel('scramble depth')
    plot.set_ylabel('state value')
    plot.legend(['value network', 'optimal solution', 'y = -x'])
    plot.get_figure().savefig(args.output + "-vals_vs_depths.png")

    # plot action match
    plt.clf()
    actions = np.argmax(policy, axis=1)
    actions_match = (actions == inv_actions).astype(np.int8)
    plot = sns.lineplot(depths, actions_match)
    plot.set_title("Actions accuracy per depths")
    plot.get_figure().savefig(args.output + "-acts_vs_depths.png")

    pass
