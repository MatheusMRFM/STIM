import networkx as nx
import numpy as np
import scipy.sparse as sp
import os, sys
import time, random, threading
import constants
import math

RANDOM_TYPE	= 0
MIN_MAX_TYPE= 1

class Graph():
	def __init__(self, folder, graph_id, pct_max_steps):
		print('Graph Number ', graph_id)
		self.graph_id = graph_id
		self.start_node_stg = constants.START_NODE_STG
		self.folder = folder
		self.max_time = self._get_max_timestep()
		self._read_TVG()
		self._load_graph(0)
		self.node_difusion = np.zeros(self.n_nodes, dtype=np.int)
		self.node_withhold = np.zeros(self.n_nodes, dtype=np.int)
		self.node_reached = np.zeros(self.n_nodes, dtype=np.int)

		self.node_type = np.zeros(self.graph.number_of_nodes(), dtype=np.int)

		self.time_step = 0
		self.starting_time = int(self.max_time * constants.PCT_START_TIME)
		self.max_steps = int(pct_max_steps * self.max_time)
		if self.max_steps > constants.MAX_STEPS:
			self.max_steps = constants.MAX_STEPS
		if self.max_steps < constants.MIN_STEPS:
			self.max_steps = constants.MIN_STEPS
		print(self.starting_time, ' ------- ', self.max_steps)

		self._read_node_types()
		self._select_starting_node_info()

	#---------------------------------------------------------------------------
	def reset_tvg(self):
		self.time_step = 0
		self._load_graph(0)
		self.node_difusion = np.zeros(self.n_nodes, dtype=np.int)
		self.node_withhold = np.zeros(self.n_nodes, dtype=np.int)
		self.node_reached = np.zeros(self.n_nodes, dtype=np.int)
		self._select_starting_node_info()

	#---------------------------------------------------------------------------
	def _read_node_types(self):
		try:
			file_name = self.folder + "/type.dat"
			file = open(file_name, 'r')
			i = 0
			for line in file:
				self.node_type[i] = int(line)
				i += 1
		except:
			pass

	#---------------------------------------------------------------------------
	def _read_TVG(self):
		self.graph_vec = []
		self.n_node_vec = []
		self.degree_vec = []
		self.norm_degree_vec = []
		self.degree_rank_vec = []
		self.norm_eigen_vec = []
		self.eigen_rank_vec = []

		for t in range(self.max_time):
			self._read_single_graph(t)

		n_edge_list = []
		for i in range(len(self.graph_vec)):
			n_edge_list.append(self.graph_vec[i].number_of_edges())

		n_edge_list = np.array(n_edge_list)
		print(self.graph_id, ': \tNumber of edges = ', n_edge_list.mean(), ' +- ', n_edge_list.std())

	#---------------------------------------------------------------------------
	def _read_single_graph(self, t):
		file_name = self.folder + str(self.graph_id) + "_" + str(t) + ".graph6"
		graph = nx.read_graph6(file_name)
		n_nodes = graph.number_of_nodes()
		degree = [val for (node, val) in graph.degree()]
		norm_degree, degree_rank = self._normalized_degree_rank(degree)
		norm_eigen, eigen_rank = self._normalized_eigenvector_rank(graph)

		self.graph_vec.append(graph)
		self.n_node_vec.append(n_nodes)
		self.degree_vec.append(degree)
		self.norm_degree_vec.append(norm_degree)
		self.degree_rank_vec.append(degree_rank)
		self.norm_eigen_vec.append(norm_eigen)
		self.eigen_rank_vec.append(eigen_rank)

	#---------------------------------------------------------------------------
	def _load_graph(self, t, reset=False, node_difusion=None, node_withhold=None, node_reached=None):
		self.graph = self.graph_vec[t]
		self.n_nodes = self.n_node_vec[t]
		self.degree = self.degree_vec[t]
		self.norm_degree = self.norm_degree_vec[t]
		self.degree_rank = self.degree_rank_vec[t]
		self.norm_eigen = self.norm_eigen_vec[t]
		self.eigen_rank = self.eigen_rank_vec[t]
		if reset:
			self.node_difusion = np.zeros(self.n_nodes, dtype=np.int)
			self.node_withhold = np.zeros(self.n_nodes, dtype=np.int)
			self.node_reached = np.zeros(self.n_nodes, dtype=np.int)
		elif not node_difusion is None:
			self.node_difusion = node_difusion
			self.node_withhold = node_withhold
			self.node_reached = node_reached

	#---------------------------------------------------------------------------
	def _get_max_timestep(self):
		files = [f for f in os.listdir(self.folder) if '.graph6' in f]
		timesteps = len(files) 
		return timesteps

	#---------------------------------------------------------------------------
	def is_min_max(self, node):
		return self.node_type[node], float(self.degree[node]) / float(self.n_nodes)

	#---------------------------------------------------------------------------
	def _random_start_info(self, n_start_nodes):
		for i in range(n_start_nodes):
			node_ok = False
			while not node_ok:
				node = np.random.randint(0, self.n_nodes)
				if self.node_withhold[node] == 0:
					self.node_difusion[node] = 1
					self.node_withhold[node] = 1
					node_ok = True
	
	#---------------------------------------------------------------------------
	def _least_connected_start_info(self, n_start_nodes):
		argsort = np.argsort(self.degree_rank)
		for i in range(n_start_nodes):
			index = self.n_nodes - i - 1
			node = argsort[index]
			#print(self.degree_rank[index])
			self.node_withhold[node] = 1
			self.node_difusion[node] = 1

	#---------------------------------------------------------------------------
	def check_special_neighbor(self, node):
		min_special_degree = 100
		neighbor_special = False
		for n in self.graph.neighbors(node):
			if self.node_type[n] == MIN_MAX_TYPE:
				neighbor_special = True
				degree = float(self.degree[n]) / float(self.n_nodes)
				if degree < min_special_degree:
					min_special_degree = degree
		return neighbor_special, min_special_degree

	#---------------------------------------------------------------------------
	def _strategic_start_info(self, n_start_nodes):
		argsort = np.argsort(self.degree_rank)
		remaining_nodes = n_start_nodes
		special_start_node = False
		index = self.n_nodes - 1
		while remaining_nodes > 0:
			node = argsort[index]
			if not special_start_node:
				neighbor_special, _ = self.check_special_neighbor(node)
				if neighbor_special:
					self.node_withhold[node] = 1
					self.node_difusion[node] = 1
					remaining_nodes -= 1
					special_start_node = True
				elif remaining_nodes > 1:
					self.node_withhold[node] = 1
					self.node_difusion[node] = 1
					remaining_nodes -= 1
			else:
				self.node_withhold[node] = 1
				self.node_difusion[node] = 1
				remaining_nodes -= 1
			index -= 1
			
	#---------------------------------------------------------------------------
	def _select_starting_node_info(self):
		n_start_nodes = int(constants.PCT_START_NODE * self.n_nodes)
		if self.start_node_stg == constants.RANDOM:
			self._random_start_info(n_start_nodes)
		elif self.start_node_stg == constants.LEAST_CONNECT:
			self._least_connected_start_info(n_start_nodes)
		else:
			self._strategic_start_info(n_start_nodes)

	#---------------------------------------------------------------------------
	def simulate_difusion(self, node, initial_step):
		original_step = self.time_step
		self.time_step = initial_step
		node_difusion = self.node_difusion
		node_withhold = self.node_withhold
		node_reached = self.node_reached
		self._load_graph(self.time_step, reset=True)
		
		self.node_difusion[node] = 1
		is_final_step = self.difuse_step(node, block_dif=False, withold=True)
		time_constraint = initial_step + constants.TIME_LIMIT

		for t in range(initial_step+1, time_constraint):
			self.difuse_step(block_dif=True, withold=False)
		score, finished = self.get_current_difusion()

		self.time_step = original_step
		self._load_graph(original_step, reset=False, node_difusion=node_difusion,
						 node_withhold=node_withhold, node_reached=node_reached)
		
		if constants.USE_CLASS:
			label = np.zeros((constants.OUT_SIZE))
			prev = 1.0
			for i in range(constants.OUT_SIZE):
				if score <= prev and score > constants.CLASS_RANGE[i]:
					label[i] = 1
				prev = constants.CLASS_RANGE[i]

			print("score = ", score, " ---- ", label)
			return label
		else:
			return [score]

	#---------------------------------------------------------------------------
	def _normalize_array(self, true_value):
		rank = np.argsort(true_value, kind='mergesort', axis=None)
		n_nodes = len(true_value)
		max = np.amax(true_value)
		min = np.amin(true_value)
		norm = np.empty([n_nodes])
		if max > 0.0 and max > min:
			for i in range(0, n_nodes):
				#norm[i] = 2.0*(float(true_value[i] - min) / float(max - min)) - 1.0
				norm[i] = (float(true_value[i] - min) / float(max - min))

		return norm, rank

	#---------------------------------------------------------------------------
	def _normalize_array_by_rank(self, true_value):
		rank = np.argsort(true_value, kind='mergesort', axis=None)
		n_nodes = len(true_value)
		norm = np.empty([n_nodes])
		for i in range(0, n_nodes):
			norm[rank[i]] = float(i+1) / float(n_nodes)
		max = np.amax(norm)
		min = np.amin(norm)
		if max > 0.0 and max > min:
			for i in range(0, n_nodes):
				#norm[i] = 2.0*(float(norm[i] - min) / float(max - min)) - 1.0
				norm[i] = (float(norm[i] - min) / float(max - min))
		else:
			print("Max value = 0")

		return norm, rank

	#---------------------------------------------------------------------------
	def _normalized_degree_rank(self, degree):
		if constants.NODE_RANK:
			norm_degree, degree_rank = self._normalize_array_by_rank(degree)
		else:
			norm_degree, degree_rank = self._normalize_array(degree)

		return norm_degree, degree_rank

	#---------------------------------------------------------------------------
	def _normalized_eigenvector_rank(self, graph):
		e = [v for v in nx.eigenvector_centrality_numpy(graph).values()]
		if constants.NODE_RANK:
			norm_eigen, eigen_rank = self._normalize_array_by_rank(e)
		else:
			norm_eigen, eigen_rank = self._normalize_array(e)

		return norm_eigen, eigen_rank

	#---------------------------------------------------------------------------
	def _get_degree(self, i):
		return self.norm_degree[i]

	#---------------------------------------------------------------------------
	def _get_degree_eigen_mean_std(self, node):
		if self.time_step == 0:
			return self._get_degree(node), 0.0, self._get_eigen(node), 0.0

		start_time = max(self.time_step - constants.MAX_TO_KEEP, 0)
		prev_degree = []
		prev_eigen = []
		for i in range(start_time, self.time_step):
			prev_degree.append(self.norm_degree_vec[i][node])
			prev_eigen.append(self.norm_eigen_vec[i][node])

		mean_d = np.mean(prev_degree)
		std_d = np.std(prev_degree)
		mean_e = np.mean(prev_eigen)
		std_e = np.std(prev_eigen)

		return mean_d, std_d, mean_e, std_e

	#---------------------------------------------------------------------------
	def get_simple_degree(self, node):
		return float(self.degree[node]) / float(self.n_nodes)

	#---------------------------------------------------------------------------
	def _get_eigen(self, i):
		return self.norm_eigen[i]

	#---------------------------------------------------------------------------
	def _get_node_features(self, n):
		node_feat = []
		node_feat = []
		node_feat.append(float(self._get_degree(n)))
		if constants.NUM_FEATURES > 1:
			node_feat.append(float(self._get_eigen(n)))
		if constants.NUM_FEATURES > 2:
			mean_d, std_d, mean_e, std_e = self._get_degree_eigen_mean_std(n)
			node_feat.append(mean_d)
			node_feat.append(std_d)
			if constants.NUM_FEATURES > 4:
				node_feat.append(mean_e)
				node_feat.append(std_e)
		if constants.NUM_FEATURES > 6:
			node_feat.append(float(self.node_reached[n]))
		if constants.NUM_FEATURES == 8:
			node_feat.append(float(self.node_withhold[n]))

		return node_feat

	#---------------------------------------------------------------------------
	def _get_feature_matrix(self):
		feat = []
		for n in range(0, self.n_nodes):
			node_feat = self._get_node_features(n)
			feat.append(node_feat)

		return feat

	#---------------------------------------------------------------------------
	def count_connected_node_withold(self):
		num = 0
		for n in range(0, self.n_nodes):
			if self.node_withhold[n] > 0:
				degree = float(self.degree[n]) / float(self.n_nodes)
				if degree > constants.WITHOLD_DEGREE:
					num += 1
		return num

	#---------------------------------------------------------------------------
	def advance_time_step(self, time=None):
		is_final_step = False
		if time is None:
			self.time_step += 1
		else:
			self.time_step = time
		if self.time_step > self.max_time or self.time_step - self.starting_time > self.max_steps:
			#print("FINAL: ", self.time_step, " -- ", self.max_time, " -- ", self.max_steps)
			is_final_step = True
		else:
			self._load_graph(self.time_step)

		eligible, _ = self.get_witholding_nodes()
		if len(eligible) == 0:
			#print("\tNo Eligible nodes")
			is_final_step = True

		return is_final_step

	#---------------------------------------------------------------------------
	def _difuse_step_single_node(self, node, block_dif=True, withold=True):
		degree = float(self.degree[node]) / float(self.n_nodes)
		yes = 0
		no = 0
		for n in self.graph.neighbors(node):
			if self.node_reached[n] == 0:
				degree_n = float(self.degree[n]) / float(self.n_nodes)
				difuse_prob = float(np.random.randint(0, 1000)) / 1000.0
				diff = degree / degree_n
				if self.node_type[n] == MIN_MAX_TYPE and diff > 1:
					th = constants.DIFUSION_RATE * math.exp( diff ) * 3
				else:
					th = constants.DIFUSION_RATE * math.exp( diff )
				
				if difuse_prob < th:
					self.node_difusion[n] = 1
					self.node_reached[n] = 1
					yes += 1
					if withold:
						self.node_withhold[n] = 1
				else:
					no += 1
		
		'''
		The node spreads the information only for 1 timestep. After that,
		the info vanishes from him
		'''
		if block_dif:
			self.node_difusion[node] = 0

	#---------------------------------------------------------------------------
	def difuse_step(self, nodes_difuse=None, block_dif=True, withold=True):
		self.difusion_bkp = np.copy(self.node_difusion)
		self.node_withhold_bkp = np.copy(self.node_withhold)

		if not nodes_difuse is None:
			if not isinstance(nodes_difuse, list):
				nodes_difuse = [nodes_difuse]
			for node in nodes_difuse:
				degree = float(self.degree[node]) / float(self.n_nodes)
				if self.node_difusion[node] == 0:
					print('Trying to start difusion process from node %d which does not contain any information!' % (node))
				self.node_withhold[node] = 0
				self.node_difusion[node] = 1
				self.node_reached[node] = 1

		for node in range(self.n_nodes):
			if self.difusion_bkp[node] == 1 and self.node_withhold[node] == 0:
				self._difuse_step_single_node(node, block_dif, withold)

		is_final_step = self.advance_time_step()

		return is_final_step

	#---------------------------------------------------------------------------
	def revert_timestep(self):
		self.node_difusion = self.difusion_bkp
		self.node_withhold = self.node_withhold_bkp
		self.time_step -= 1
		self._load_graph(self.time_step)

	#---------------------------------------------------------------------------
	def get_witholding_nodes(self):
		degree = []
		eligible = []
		#for i in range(len(self.node_withhold)):
		for i in range(self.n_nodes):
			if self.node_withhold[i] == 1:
				eligible.append(i)
				degree.append(float(self.degree[i]) / float(self.n_nodes))
		return eligible, degree

	#---------------------------------------------------------------------------
	def check_node_withold(self, node):
		if self.node_withhold[node] == 1:
			return True
		return False

	#---------------------------------------------------------------------------
	def get_influence_count(self):
		influence_node = []
		for i in range(len(self.node_withhold)):
			if self.node_withhold[i] == 1:
				influence_node.extend(self.graph.neighbors(i))
		
		influence_node = np.unique(np.array(influence_node))
		'''influence_node_final = []
		for i in range(len(influence_node)):
			if self.node_reached[influence_node[i]] == 0 and self.node_withhold[influence_node[i]] == 0:
				influence_node_final.append(influence_node[i])

		return float(len(influence_node_final)) / float(self.n_nodes)'''

		return float(len(influence_node)) / float(self.n_nodes)

	#---------------------------------------------------------------------------
	def get_current_difusion(self):
		reached_nodes = np.sum(self.node_reached)
		finished = False
		if reached_nodes == self.n_nodes:
			finished = True
		reached_nodes = float(reached_nodes) / float(self.n_nodes)
		
		return reached_nodes, finished

	#---------------------------------------------------------------------------
	def get_adj_feat_matrix_step_t(self):
		adj = nx.to_scipy_sparse_matrix(self.graph, format='coo')
		if constants.EMBED_METHOD != constants.S2VEC:
			adj_ = adj + sp.eye(adj.shape[0])
			rowsum = np.array(adj_.sum(1))
			degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
			adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
			adj = adj_normalized
			
		feat = self._get_feature_matrix()

		return adj, feat

	#---------------------------------------------------------------------------
	def get_node_type(self, node):
		#degree = self._get_degree(node)
		is_min_max, degree_special = self.is_min_max(node)
		if degree_special > 0.4:
			if self.node_type[node] == MIN_MAX_TYPE:
				type_node = 4
			else:
				type_node = 3
		else:
			neighbor_special, degree_neighbor = self.check_special_neighbor(node)
			d = self._get_degree(node)
			if self.node_type[node] == MIN_MAX_TYPE:
				type_node = 1
			elif neighbor_special and degree_neighbor < 0.4:
				type_node = 2
			else:
				type_node = 0
		return type_node

	#---------------------------------------------------------------------------
	def get_n_node_ids(self, n_val):
		min_max = True
		min_max_id = []
		min_max_neighbor_id = []
		normal = []
		selected_nodes = []
		node_types = []
		for node in range(self.graph.number_of_nodes()):
			if self.node_type[node] == MIN_MAX_TYPE:
				min_max_id.append(node)
			else:
				neighbor_special, degree_special = self.check_special_neighbor(node)
				d = self._get_degree(node)
				if neighbor_special and degree_special < d * 1.1:
					min_max_neighbor_id.append(node)
				normal.append(node)

		for i in range(n_val):
			if min_max and len(min_max_id) > 0:
				id = np.random.randint(0, 10)
				if id > 5 and len(min_max_neighbor_id) > 0:
					id = np.random.randint(0, len(min_max_neighbor_id))
					node = min_max_neighbor_id[id]
					type_node = 1
				else:
					id = np.random.randint(0, len(min_max_id))
					node = min_max_id[id]
					type_node = 2
			else:
				id = np.random.randint(0, len(normal))
				node = normal[id]
				type_node = 0
			#node = np.random.randint(0, self.graph.number_of_nodes())
			selected_nodes.append(node)
			node_types.append(type_node)
			min_max = not min_max

		return selected_nodes, node_types

	#---------------------------------------------------------------------------
	def print_graph_stats(self):
		print("Number of Nodes: ", self.n_nodes)
		print("Time steps: ", self.max_time)


