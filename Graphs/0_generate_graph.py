import networkx as nx
import numpy as np
import os
from time import sleep
from time import time
import numpy as np
import time, random, threading
import multiprocessing
from graph import *

num_workers = 1



TRAIN   = 0
TEST    = 1

TRAIN_FLD = 'Train/'
TEST_FLD = 'Test/'

"""
Variables that control how the graphs are generated
"""
NUM_TRAIN_GRAPH		= 500
NUM_TEST_GRAPH		= 50

GRAPH_TYPE 			= SCALEFREE

MAX_NODES			= 5000
MIN_NODES			= 500

MAX_P				= 0.1	
MIN_P 				= 0.05

MAX_K				= 6
MIN_K 				= 2

MAX_M				= 20
MIN_M 				= 5

MAX_T				= 60
MIN_T				= 30

MAX_PERT_P			= 0.07
MIN_PERT_P			= 0.01

"""
Variables for controlling race conditions
"""
graph_train_number = -1
graph_test_number = -1

class Graph_Generator():
	def __init__(self, id, num_worker):
		self.num_worker = num_worker
		self.tid = id
		self.folder_name = "./Graphs/"

	#---------------------------------------------------------------------------
	def generate_set(self, mode, id):
		graph_train_number = -1
		graph_test_number = -1

		if mode == TRAIN:
			num_it = int(NUM_TRAIN_GRAPH/self.num_worker)
			start = id * num_it
			end = (id+1) * num_it
			sub_folder = TRAIN_FLD
		else:
			num_it = int(NUM_TEST_GRAPH/self.num_worker)
			start = id * num_it
			end = (id+1) * num_it
			sub_folder = TEST_FLD
		print(start, " --- ", end)

		graph = Graph()

		for g_number in range(start, end):
			if mode == TRAIN:
				print(g_number, " -- TRAIN")
			else:
				print(g_number, " -- TEST")
			n, p, k, m, T, pert_p, type = self.fetch_variables()
			graph.generate_mag(sub_folder, g_number, n, p, k, m, T, pert_p, type)

	#---------------------------------------------------------------------------
	def fetch_variables(self):
		np.random.seed(int(time.time()))
		#n = np.random.randint(MIN_NODES, MAX_NODES+1)
		n = MAX_NODES
		p = float(np.random.randint(int(1000*MIN_P), int(1000*(MAX_P)))) / 1000.0
		k = np.random.randint(MIN_K, MAX_K+1)
		m = np.random.randint(MIN_M, MAX_M+1)
		T = np.random.randint(MIN_T, MAX_T+1)
		pert_p = float(np.random.randint(int(1000*MIN_PERT_P), int(1000*(MAX_PERT_P)))) / 1000.0

		type = GRAPH_TYPE
		if GRAPH_TYPE == MIX:
			type = np.random.randint(0, RANDOM+1)
		elif GRAPH_TYPE == SCALE_SMALL:
			type = np.random.randint(0, SCALEFREE+1)

		return n, p, k, m, T, pert_p, type

	#---------------------------------------------------------------------------
	def run_graph_generator(self, id):
		self.generate_set(TRAIN, id)
		self.generate_set(TEST, id)






workers = []
for i in range(num_workers):
	print (i)
	workers.append(Graph_Generator(i, num_workers))

"""
Initializes the worker threads
"""
worker_threads = []
for i in range(num_workers):
	t = multiprocessing.Process(target=workers[i].run_graph_generator, args=(i,))
	t.start()
	sleep(1)
	worker_threads.append(t)

for i in range(num_workers):
	worker_threads[i].join()
