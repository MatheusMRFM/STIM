import networkx as nx
import numpy as np
import scipy as sp
import os, sys, shutil

SMALLWORLD  = 0
SCALEFREE   = 1
RANDOM      = 2
SCALE_SMALL	= 3
MIX         = 4

RANDOM_TYPE	= 0
MIN_MAX_TYPE= 1

USE_MAX_MIN	= True

MIN_MAX_PCT	= 0.02
MIN_MAX_HIGH= 0.4
MIN_MAX_LOW	= 0.01
MIN_MAX_RATE= 0.15

DECREASE	= 0
INCREASE	= 1


class Graph():
	def __init__(self):
		self.graph = nx.Graph()

	#---------------------------------------------------------------------------
	def get_num_nodes(self):
		return self.graph.number_of_nodes()

	#---------------------------------------------------------------------------
	def build_degree_vec(self):
		self.degree_norm = [float(val) / float(self.graph.number_of_nodes()) for (node, val) in self.graph.degree()]

	#---------------------------------------------------------------------------
	def print_degree_diff(self, num=-1):
		old_degree = self.degree_norm.copy()
		self.build_degree_vec()
		cont = 0
		for node in range(self.graph.number_of_nodes()):
			if self.node_type[node] == MIN_MAX_TYPE:
				print('%d - (%f, %f)' % (node, old_degree[node], self.degree_norm[node]))
				if num > 0:
					cont += 1
					if cont >= num:
						break

	#---------------------------------------------------------------------------
	def generate_random_graph(self, n, p):
		self.graph = nx.erdos_renyi_graph(n, p)

	#---------------------------------------------------------------------------
	def generate_smallworld_graph(self, n, p, k):
		self.graph = nx.newman_watts_strogatz_graph(n, k, p)

	#---------------------------------------------------------------------------
	def generate_scalefree_graph(self, n, m):
		self.graph = nx.barabasi_albert_graph(n, m)

	#---------------------------------------------------------------------------
	def define_node_types(self):
		#id_rank = np.argsort(self.degree, kind='mergesort', axis=None)
		n_minmax = int(self.graph.number_of_nodes() * MIN_MAX_PCT)
		alternate = False
		for n in range(self.graph.number_of_nodes()):
			if n < n_minmax:
				self.node_type[n] = MIN_MAX_TYPE
				if alternate:
					self.node_state[n] = DECREASE	
				else:
					self.node_state[n] = INCREASE
				alternate = True
			else:
				self.node_type[n] = RANDOM_TYPE

	#---------------------------------------------------------------------------
	def save_types_file(self, folder, graph_id):
		sub_folder = folder + str(graph_id) + '/'
		file = open(sub_folder + 'type.dat', "w")
		for n in range(self.graph.number_of_nodes()):
			file.write(str(self.node_type[n]) + '\n')
		file.close()

	#---------------------------------------------------------------------------
	def make_n_connections(self, node, n_connect):
		add = []
		for _ in range(n_connect):
			n = np.random.randint(0, self.graph.number_of_nodes())
			add.append([node, n])

		return add

	#---------------------------------------------------------------------------
	def remove_n_connections(self, node, n_connect):
		remove = []
		for _ in range(n_connect):
			neighbor = list(self.graph.neighbors(node))
			if len(neighbor) > 1:
				n = np.random.randint(0, len(neighbor))
				neighbor_n = list(self.graph.neighbors(neighbor[n]))
				if len(neighbor_n) > 1:
					remove.append([node, neighbor[n]])

		return remove

	#---------------------------------------------------------------------------
	def pertubation_node_type_min_max(self, node, pert_p):
		remove = []
		add = []
		new_degree = [val for (node, val) in self.graph.degree()]
		n_nodes = self.get_num_nodes()
		if self.node_state[node] == INCREASE:
			if new_degree[node] <= n_nodes * MIN_MAX_HIGH:
				add = self.make_n_connections(node, int(n_nodes * MIN_MAX_RATE))
			else:
				self.node_state[node] = DECREASE
				remove = self.remove_n_connections(node, int(n_nodes * MIN_MAX_RATE))
		else:
			#if self.degree[node] < new_degree[node]:
			if new_degree[node] > n_nodes * MIN_MAX_LOW:
				remove = self.remove_n_connections(node, int(n_nodes * MIN_MAX_RATE))
			else:
				self.node_state[node] = INCREASE
				add = self.make_n_connections(node, int(n_nodes * MIN_MAX_RATE))


		return remove, add

	#---------------------------------------------------------------------------
	def pertubation_node_type_random(self, node, pert_p):
		remove = []
		add = []
		for n in self.graph.neighbors(node):
			p = np.random.randint(0, 1000)
			if p < pert_p*1000:
				remove.append([node, n])
			p = np.random.randint(0, 1000)
			if p < pert_p*1000:
				rand_n = np.random.randint(0, self.graph.number_of_nodes())
				add.append([node, rand_n])

		return remove, add

	#---------------------------------------------------------------------------
	def _pertubate_node(self, node, pert_p):
		if self.node_type[node] == RANDOM_TYPE:
			r, a = self.pertubation_node_type_random(node, pert_p)
		elif self.node_type[node] == MIN_MAX_TYPE:
			r, a = self.pertubation_node_type_min_max(node, pert_p)

		return r, a

	#---------------------------------------------------------------------------
	def apply_graph_perturbations(self, pert_p):
		self.degree = [val for (node, val) in self.graph.degree()]

		remove = []
		add = []
		for n in range(self.graph.number_of_nodes()):
			r, a = self._pertubate_node(n, pert_p)
			for pair in r:
				remove.append(pair)
			for pair in a:
				add.append(pair)

		for node_pair in remove:
			neighbor_0 = list(self.graph.neighbors(node_pair[0]))
			neighbor_1 = list(self.graph.neighbors(node_pair[1]))
			if node_pair[1] in neighbor_0:
				if len(neighbor_0) > 1 and len(neighbor_1) > 1:
					self.graph.remove_edge(node_pair[0], node_pair[1])
		for node_pair in add:
			self.graph.add_edge(node_pair[0], node_pair[1])

	#---------------------------------------------------------------------------
	def save_mag_current_time(self, folder, graph_id, t):
		sub_folder = folder + str(graph_id) + '/'
		if t == 0:
			if os.path.exists(sub_folder):
				shutil.rmtree(sub_folder)
			os.mkdir(sub_folder)

		self.save_file(sub_folder, graph_id, t)

	#---------------------------------------------------------------------------
	def generate_mag(self, folder, graph_id, n, p, k, m, T, pert_p, graph_type):
		print("NUM NODES = ", n)
		if graph_type == RANDOM:
			self.generate_random_graph(n, p)
		elif graph_type == SMALLWORLD:
			self.generate_smallworld_graph(n, p, k)
		elif graph_type == SCALEFREE:
			self.generate_scalefree_graph(n, m)
		else:
			print("Unexpected GRAPH_TYPE value.")

		self.node_type = np.zeros(self.graph.number_of_nodes(), dtype=np.int)
		self.node_state = np.zeros(self.graph.number_of_nodes(), dtype=np.int)
		if USE_MAX_MIN:
			self.define_node_types()

		for t in range(0, T):
			'''if t == 0:
				self.build_degree_vec()
			else:
				self.print_degree_diff(num=10)
				print('\n')
			'''
			self.save_mag_current_time(folder, graph_id, t)
			self.apply_graph_perturbations(pert_p)
		
		self.save_types_file(folder, graph_id)

	#---------------------------------------------------------------------------
	def save_file(self, folder, id, timestep):
		#name = folder + str(id) + "_" + str(timestep) + ".gexf"
		#nx.write_gexf(self.graph, name)

		#name = folder + str(id) + "_" + str(timestep) + ".sparse6"
		#nx.write_sparse6(self.graph, name)

		name = folder + str(id) + "_" + str(timestep) + ".graph6"
		nx.write_graph6(self.graph, name)
		


