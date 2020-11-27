from abc import ABC, abstractmethod
import os, sys
import numpy as np
import time, random, threading, sys
import multiprocessing
from graph import *
import constants

lock = threading.Lock()
graph_train		 	= None
graph_test		 	= None
graph_train_number 	= 0
graph_test_number	= 0
test_end = False

class Agent(ABC):

	@abstractmethod
	def __init__(self, id, pct_max_steps):
		self.id = id
		self.mean_reward = []
		self.pct_max_steps = pct_max_steps
		self.graph = None
		self.folder_name = constants.FOLDER
		self.train_subfolder = constants.TRAIN_FOLDER
		self.test_subfolder = constants.TEST_FOLDER
		self.real_net_subfolder = "Real_Networks/"
		if self.id != 'global':
			self._load_graph_set_data()

	#---------------------------------------------------------------------------
	def _load_graph_set_data(self):
		folder = self.folder_name + self.train_subfolder
		self.train_files = [f for f in os.listdir(folder) if not os.path.isfile(folder + f)]
		self.train_size = len(self.train_files)
		folder = self.folder_name + self.test_subfolder
		self.test_files = [f for f in os.listdir(folder) if not os.path.isfile(folder + f)]
		self.test_size = len(self.test_files)
		folder = self.folder_name + self.real_net_subfolder
		self.real_net_files = [f for f in os.listdir(folder) if not os.path.isfile(folder + f)]
		self.real_net_size = len(self.real_net_files)

		if not constants.LOW_MEMORY:
			self._load_dataset_in_memory()

	#---------------------------------------------------------------------------
	def _load_in_memory(self, sub_folder, tvg_list):
		tvg_dict = {}
		for tvg in tvg_list:
			print("Loading ", sub_folder + tvg)
			folder = self.folder_name + sub_folder + str(tvg) + '/'
			tvg_dict[tvg] = Graph(folder, tvg, self.pct_max_steps)
		return tvg_dict

	#---------------------------------------------------------------------------
	def _load_dataset_in_memory(self):
		if constants.MODE == constants.TRAIN:
			self.train_tvg_list = self._load_in_memory(self.train_subfolder, self.train_files)
		if constants.MODE == constants.TRAIN or constants.MODE == constants.TEST:
			self.test_tvg_list = self._load_in_memory(self.test_subfolder, self.test_files)
		else:
			self.real_tvg_list = self._load_in_memory(self.real_net_subfolder, self.real_net_files)

	#---------------------------------------------------------------------------
	def _load_tvg_from_memory(self, graph_id):
		if constants.MODE == constants.TRAIN:
			g = self.train_tvg_list[graph_id]
		elif constants.MODE == constants.TEST:
			g = self.test_tvg_list[graph_id]
		else:
			g = self.real_tvg_list[graph_id]
		'''else:
			print("Trying to load non-training graph from memory!")
			exit(0)'''

		g.reset_tvg()

		return g

	#---------------------------------------------------------------------------
	def _load_next_graph(self):
		global graph_train_number, graph_train
		global graph_test_number, graph_test
		global test_end

		lock.acquire()

		if constants.MODE == constants.TRAIN:
			if graph_train is None:
				graph_train = np.arange(self.train_size)
				np.random.shuffle(graph_train)
				graph_train_number = 0
			g_number = graph_train[graph_train_number]
			sub_folder = self.train_subfolder 
			graph_id = self.train_files[g_number]
			graph_train_number += 1
			if graph_train_number >= self.train_size:
				graph_train = None
		else:
			if graph_test is None:
				if constants.MODE == constants.TEST:
					graph_test = np.arange(self.test_size)
					np.random.shuffle(graph_test)
				else:
					graph_test = np.arange(self.real_net_size)
				graph_test_number = 0
			g_number = graph_test[graph_test_number]
			if constants.MODE == constants.TEST:
				file_list = self.test_files
				sub_folder = self.test_subfolder
			else:
				file_list = self.real_net_files
				sub_folder = self.real_net_subfolder
			graph_id = file_list[g_number]
			graph_test_number += 1
			if constants.MODE == constants.TEST:
				set_size = self.test_size
			else:
				set_size = self.real_net_size
			if graph_test_number >= set_size:
				if constants.MODE != constants.TRAIN:
					test_end = True
				graph_test = None 
		lock.release()

		folder = self.folder_name + sub_folder + str(graph_id) + '/'
		if constants.LOW_MEMORY: #or constants.MODE != constants.TRAIN:
			self.graph = Graph(folder, graph_id, self.pct_max_steps)
		else:
			self.graph = self._load_tvg_from_memory(graph_id)

	#---------------------------------------------------------------------------
	def take_action(self, node):
		is_final_step = False
		is_final_step = self.graph.difuse_step(node)

		return is_final_step

	#---------------------------------------------------------------------------
	def get_score(self):
		reached_nodes, finished = self.graph.get_current_difusion()
		return reached_nodes, finished

	#---------------------------------------------------------------------------
	def work(self):
		size_result = {}
		for _ in range(constants.N_GRAPH_ITERATE): 
			self._load_next_graph()
			is_final_step = False
			end = False
			score = 0
			for i in range(self.graph.max_time):
				'''if i == 0:
					self.graph.build_degree_vec()
				else:
					self.graph.print_degree_diff(num=10)
				'''
				if i >= self.graph.starting_time:
					node = self._select_node()
					is_final_step = self.take_action(node)
					#score, end = self.get_score()
				else:
					self.graph.advance_time_step()
				score, end = self.get_score()

				if is_final_step or end:
					break

			#if not self.graph.n_nodes in size_result.keys():
			#	size_result[self.graph.n_nodes] = []
			#size_result[self.graph.n_nodes].append(score)
			#print('\n\n')
			#for key in size_result.keys():
			#	if len(size_result[key]) > 0:
			#		print(key, " ====> ", np.mean(size_result[key]), ' ___ ', np.std(size_result[key]))

			self.mean_reward.append(score)
			mean = np.array(self.mean_reward).mean()
			print('SCORE: %f - %f (%d)' % (score, mean, len(self.mean_reward)))

	#---------------------------------------------------------------------------
	@abstractmethod
	def _select_node(self):
		pass