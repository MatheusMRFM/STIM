import numpy as np
import random
import scipy as sp
import constants

RL_MODE     = 0
ADJ_MODE    = 1
ADJ_RL_MODE = 2

class Batch():
    def __init__(self, num_nodes):
        self.features = []
        self.step = []
        self.adj = []
        self.node = []
        self.node_sup = None
        self.node_type = []
        self.r = []
        self.bootstrap = None
        self.num_nodes = num_nodes
        self.best_q_dist = []
        self.size = 0

    #---------------------------------------------------------------------------
    def add_info(self, A, X, node, r, step, best_q_dist):
        self.adj.append(A)
        self.features.append(X)
        if not node is None:
            self.node.append(node)
        self.r.append(r)
        self.best_q_dist.append(best_q_dist)
        self.size += 1
        self.step.append(step)
        if self.size > constants.BATCH_SIZE:
            self.adj = self.adj[1:]
            self.features = self.features[1:]
            if not node is None:
                self.node = self.node[1:]
            self.r = self.r[1:]
            self.step = self.step[1:]

    #---------------------------------------------------------------------------
    def add_bootstrap(self, b):
        self.bootstrap = b

    #---------------------------------------------------------------------------
    def set_hidden_states(self, hidden_states_f1, hidden_states_f2):
        hs1 = np.array(hidden_states_f1)
        hs1 = np.transpose(hs1, (1, 0, 2, 3))
        hs2 = np.array(hidden_states_f2)
        hs2 = np.transpose(hs2, (1, 0, 2, 3))
        self.hidden_states_f1 = hs1
        self.hidden_states_f2 = hs2

    #---------------------------------------------------------------------------
    def set_node_sup(self, graph):
        self.node_sup, self.node_type = graph.get_n_node_ids(self.size)
        self.true_cent = []
        for i in range(self.size):
            node = self.node_sup[i]
            initial_step = self.step[i]
            cent = graph.simulate_difusion(node, initial_step)
            #r = self.graph.node_type[node]
            #r = self.graph.get_node_class(node)
            #batch.r.append([r])
            self.true_cent.append(cent)
        self.true_cent = np.array(self.true_cent)

    #---------------------------------------------------------------------------
    def reset(self):
        self.features = []
        self.adj = []
        self.node = []
        self.node_sup = None
        self.r = []
        self.bootstrap = None
        self.step = []
        self.best_q_dist = []
        self.size = 0
