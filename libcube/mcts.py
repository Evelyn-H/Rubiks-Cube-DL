import random
import numpy as np
import collections
import queue
from profilehooks import profile

from . import cubes
from . import model

import torch
import torch.nn.functional as F


class Greedy:
    """
    Best-first search using the value network
    """

    def __init__(self, cube_env, state, net, device, max_depth=500, max_iterations=5000):
        assert isinstance(cube_env, cubes.CubeEnv)
        assert cube_env.is_state(state)

        self.cube_env = cube_env
        self.root_state = state
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.net = net
        self.device = device
        self.iterations_needed = -1
        self.nodes_evaluated = 0

    def simple_traversal(self):
        path = []
        next_state = self.root_state

        past_values = []

        depth = 0
        while depth < self.max_depth:
            depth += 1

            # explore possible moves
            c_states, c_goals = self.cube_env.explore_state(next_state)
            # if one of them solves it, go with that
            for a_idx, (c_state, c_goal) in enumerate(zip(c_states, c_goals)):
                if c_goal:
                    self.iterations_needed = len(path) + 1
                    print(f'values: {[f"{v:.2f}" for v in past_values]}')
                    return path + [a_idx]

            # otherwise...

            # policy search
            # policy = self.evaluate_states([next_state])[0][0]
            #
            # action = np.argmax(policy)
            # next_state = c_states[action]
            # path += [action]

            # # value search
            _, values = self.evaluate_states(c_states)

            action = np.argmax(values)
            next_state = c_states[action]
            path += [action]

            past_values.append(values[action])

        self.iterations_needed = self.max_depth
        return None

    @profile
    def bfs(self):
        # note, the queue is lowest-first,
        # but our values need to be best-first (i.e. highest)
        # so, we just used `-value` instead
        q = queue.PriorityQueue()
        q.put((-1000, self.root_state, [], []))
        seen = set()

        iterations = 0

        # TODO: How can we parallelise this procedure?
        #  - maybe just push the first n items in the list to the NN at once?

        while not q.empty():
            iterations += 1
            # if iterations > 10000:
            if iterations > self.max_iterations:
                self.iterations_needed = iterations
                l = [0] * 40
                v = [[] for _ in range(40)]
                o = 0
                o_v = []
                while not q.empty():
                    value, s, path, states = q.get()
                    if len(path)-1 < 40:
                        l[len(path)-1] += 1
                        v[len(path)-1].append(value)
                    else:
                        o += 1
                        o_v.append(value)

                print(*[f"{d}, {np.percentile(v, [0, 50, 100]) if len(v) > 0 else None}" for d, v in zip(l, v)], sep='\n')
                # print(*[f"{d}" for d in l], sep='\n')
                print('>40:', o)
                return None

            value, s, path, states = q.get()
            # print(value)

            # seen.add(s)
            c_states, c_goals = self.cube_env.explore_state(s)
            # policy, values = self.evaluate_states(c_states)
            values = self.eval_states_values(c_states)
            # best_action = np.argmax(values)
            # for a_idx, (probability, value, c_state, c_goal) in enumerate(zip(policy[0], values, c_states, c_goals)):
            for a_idx, (value, c_state, c_goal) in enumerate(zip(values, c_states, c_goals)):
                self.nodes_evaluated += 1

                if value > -1.5:
                    rendered = self.cube_env.render(c_state)
                    print(value, rendered)

                p = path + [a_idx]
                if c_goal:
                    self.iterations_needed = iterations
                    # print(iterations)
                    def target_value(s):
                        c_states, c_goals = self.cube_env.explore_state(s)
                        values = self.eval_states_values(c_states)
                        return max([v[0] for v in values]) - 1

                    if len(p) > 1:
                        vals = self.eval_states_values(states)
                        # print(f'values: {[f"{v[0]:.2f}" for v in vals]}')
                        # print(f'target: {[f"{target_value(s):.2f}" for s in states]}')
                    return p

                if len(p) > 40:
                    continue

                if c_state in states:
                    # print('seen')
                    continue

                # if c_state in seen:
                    # continue

                # curve_val = -8.346825 + 8.494041 * np.exp(-0.1786749*len(p))
                # heuristic = -value #- curve_val*0.8# - probability
                heuristic = -value #+ len(p)# - probability

                q.put((heuristic, c_state, p, states + [s]))

    def search(self):
        return self.bfs()
        # return self.simple_traversal()

    def find_solution(self):
        return []

    def evaluate_states(self, states):
        enc_states = model.encode_states(self.cube_env, states)
        enc_states_t = torch.tensor(enc_states).to(self.device)
        policy_t, value_t = self.net(enc_states_t)
        policy_t = F.softmax(policy_t, dim=1)
        return policy_t.detach().cpu().numpy(), value_t.squeeze(-1).detach().cpu().numpy()

    def eval_states_values(self, states):
        enc_states = model.encode_states(self.cube_env, states)
        enc_states_t = torch.tensor(enc_states).to(self.device)
        value_t = self.net(enc_states_t, value_only=True)
        return value_t.detach().cpu().numpy()

    def dump_solution(self, solution):
        assert isinstance(solution, list)

        print("==== SOLUTION ====")
        s = self.root_state
        r = self.cube_env.render(s)
        print(r)
        for aidx in solution:
            a = self.cube_env.action_enum(aidx)
            print(a, aidx)
            s = self.cube_env.transform(s, a)
            r = self.cube_env.render(s)
            print(r)
        print("==== END OF SOLUTION ====")


    def __len__(self):
        return self.iterations_needed

    def get_depth_stats(self):
        return {
            'max': 0,
            'mean': 0,
            'leaves': self.nodes_evaluated,
        }


class MCTS:
    """
    Monte Carlo Tree Search state and method
    """
    def __init__(self, cube_env, state, net, exploration_c=10, virt_loss_nu=100.0, device="cpu"):
        assert isinstance(cube_env, cubes.CubeEnv)
        assert cube_env.is_state(state)

        self.cube_env = cube_env
        self.root_state = state
        self.net = net
        self.exploration_c = exploration_c
        self.virt_loss_nu = virt_loss_nu
        self.device = device

        # Tree state
        shape = (len(cube_env.action_enum), )
        # correspond to N_s(a) in the paper
        self.act_counts = collections.defaultdict(lambda: np.zeros(shape, dtype=np.uint32))
        # correspond to W_s(a)
        self.val_maxes = collections.defaultdict(lambda: np.zeros(shape, dtype=np.float32))
        # correspond to P_s(a)
        self.prob_actions = {}
        # values
        self.values = {}
        # correspond to L_s(a)
        self.virt_loss = collections.defaultdict(lambda: np.zeros(shape, dtype=np.float32))
        # TODO: check speed and memory of edge-less version
        self.edges = {}

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "MCTS(states=%d)" % len(self.edges)

    def dump_root(self):
        print("Root state:")
        self.dump_state(self.root_state)
        # states, _ = cubes.explore_state(self.cube_env, self.root_state)
        # for idx, s in enumerate(states):
        #     print("")
        #     print("State %d" % idx)
        #     self.dump_state(s)

    def dump_state(self, s):
        print("")
        print("act_counts: %s" % ", ".join(map(lambda v: "%8d" % v, self.act_counts[s].tolist())))
        print("probs:      %s" % ", ".join(map(lambda v: "%.2e" % v, self.prob_actions[s].tolist())))
        print("val_maxes:  %s" % ", ".join(map(lambda v: "%.2e" % v, self.val_maxes[s].tolist())))

        act_counts = self.act_counts[s]
        N_sqrt = np.sqrt(np.sum(act_counts))
        u = self.exploration_c * N_sqrt / (act_counts + 1)
        print("u:          %s" % ", ".join(map(lambda v: "%.2e" % v, u.tolist())))
        u *= self.prob_actions[s]
        print("u*prob:     %s" % ", ".join(map(lambda v: "%.2e" % v, u.tolist())))
        q = self.val_maxes[s] - self.virt_loss[s]
        print("q:          %s" % ", ".join(map(lambda v: "%.2e" % v, q.tolist())))
        fin = u + q
        print("u*prob + q: %s" % ", ".join(map(lambda v: "%.2e" % v, fin.tolist())))
        act = np.argmax(fin, axis=0)
        print("Action: %s" % act)

    def search(self):
        s, path_actions, path_states = self._search_leaf()

        child_states, child_goal = self.cube_env.explore_state(s)
        self.edges[s] = child_states

        value = self._expand_leaves([s])[0]
        self._backup_leaf(path_states, path_actions, value)

        if np.any(child_goal):
            path_actions.append(np.argmax(child_goal))
            return path_actions
        return None

    def _search_leaf(self):
        """
        Starting the root state, find path to the leaf node
        :return: tuple: (state, path_actions, path_states)
        """
        s = self.root_state
        path_actions = []
        path_states = []

        # walking down the tree
        while True:
            next_states = self.edges.get(s)
            if next_states is None:
                break

            act_counts = self.act_counts[s]
            N_sqrt = np.sqrt(np.sum(act_counts))
            if N_sqrt < 1e-6:
                act = random.randrange(len(self.cube_env.action_enum))
            else:
                u = self.exploration_c * N_sqrt / (act_counts + 1)
                u *= self.prob_actions[s]
                # IDEA: use softmax of values instead of probability network
                # e = np.exp(np.array(self.values[s]))
                # u *= e / np.sum(e)
                q = self.val_maxes[s] - self.virt_loss[s]
                act = np.argmax(u + q)
            self.virt_loss[s][act] += self.virt_loss_nu
            path_actions.append(act)
            path_states.append(s)
            s = next_states[act]
        return s, path_actions, path_states

    def _expand_leaves(self, leaf_states):
        """
        From list of states expand them using the network
        :param leaf_states: list of states
        :return: list of state values
        """
        policies, values = self.evaluate_states(leaf_states)
        for s, p in zip(leaf_states, policies):
            self.prob_actions[s] = p
            # get values
            # child_states, child_goal = self.cube_env.explore_state(s)
            # values = self.eval_states_values(child_states)
            # self.values[s] = np.ravel(values)
        return values

    def _backup_leaf(self, states, actions, value):
        """
        Update tree state after reaching and expanding the leaf node
        :param states: path of states (without final leaf state)
        :param actions: path of actions
        :param value: value of leaf node
        """
        for path_s, path_a in zip(states, actions):
            self.act_counts[path_s][path_a] += 1
            w = self.val_maxes[path_s]
            w[path_a] = max(w[path_a], value)
            self.virt_loss[path_s][path_a] -= self.virt_loss_nu

    def search_batch(self, batch_size):
        """
        Perform a batches search to increase efficiency.
        :param batch_size: size of search batch
        :return: path to solution or None if not found
        """
        batch_size = min(batch_size, len(self) + 1)
        batch_states, batch_actions, batch_paths = [], [], []
        for _ in range(batch_size):
            s, path_acts, path_s = self._search_leaf()
            batch_states.append(s)
            batch_actions.append(path_acts)
            batch_paths.append(path_s)

        for s, path_actions in zip(batch_states, batch_actions):
            child, goals = self.cube_env.explore_state(s)
            self.edges[s] = child
            if np.any(goals):
                return path_actions + [np.argmax(goals)]

        values = self._expand_leaves(batch_states)
        for value, path_states, path_actions in zip(values, batch_paths, batch_actions):
            self._backup_leaf(path_states, path_actions, value)
        return None

    def evaluate_states(self, states):
        """
        Ask network to return policy and values
        :param net:
        :param states:
        :return:
        """
        enc_states = model.encode_states(self.cube_env, states)
        enc_states_t = torch.tensor(enc_states).to(self.device)
        policy_t, value_t = self.net(enc_states_t)
        policy_t = F.softmax(policy_t, dim=1)
        return policy_t.detach().cpu().numpy(), value_t.squeeze(-1).detach().cpu().numpy()

    def eval_states_values(self, states):
        enc_states = model.encode_states(self.cube_env, states)
        enc_states_t = torch.tensor(enc_states).to(self.device)
        value_t = self.net(enc_states_t, value_only=True)
        return value_t.detach().cpu().numpy()

    def get_depth_stats(self):
        """
        Calculate minimum, maximum, and mean depth of children in the tree
        :return: dict with stats
        """
        max_depth = 0
        sum_depth = 0
        leaves_count = 0
        q = collections.deque([(self.root_state, 0)])
        met = set()

        while q:
            s, depth = q.popleft()
            met.add(s)
            for ss in self.edges[s]:
                if ss not in self.edges:
                    max_depth = max(max_depth, depth+1)
                    sum_depth += depth+1
                    leaves_count += 1
                elif ss not in met:
                    q.append((ss, depth+1))
        return {
            'max': max_depth,
            'mean': sum_depth / leaves_count,
            'leaves': leaves_count
        }

    def dump_solution(self, solution):
        assert isinstance(solution, list)

        s = self.root_state
        r = self.cube_env.render(s)
        print(r)
        for aidx in solution:
            a = self.cube_env.action_enum(aidx)
            print(a, aidx)
            s = self.cube_env.transform(s, a)
            r = self.cube_env.render(s)
            print(r)

    def find_solution(self):
        queue = collections.deque([(self.root_state, [])])
        seen = set()

        while queue:
            s, path = queue.popleft()
            seen.add(s)
            c_states, c_goals = self.cube_env.explore_state(s)
            for a_idx, (c_state, c_goal) in enumerate(zip(c_states, c_goals)):
                p = path + [a_idx]
                if c_goal:
                    return p
                if c_state in seen or c_state not in self.edges:
                    continue
                queue.append((c_state, p))
