from agent import *
import constants
import networkx as nx

class Temp_Degree(Agent):

	def __init__(self, id, pct_max_steps):
		super().__init__(id, pct_max_steps)

	#---------------------------------------------------------------------------
	def _get_spreader_cent(self, node, degree, clust):
		sum_cent_neighbor = 0
		for n in self.graph.graph.neighbors(node):
			sum_cent_neighbor += clust[n]
		
		cent = sum_cent_neighbor
		if degree[node] > 0:
			cent += degree[node] * 1.0 / (clust[node] + 1.0 / float(degree[node]))

		return cent
	
	#---------------------------------------------------------------------------
	def _select_node(self):
		eligible_node, degree = self.graph.get_witholding_nodes()
		degree_vec = self.graph.degree
		clust_vec = [v for v in nx.clustering(self.graph.graph).values()]
		max = -1
		chosen_node = -1
		for node, d in zip(eligible_node, degree):
			cent = self._get_spreader_cent(node, degree_vec, clust_vec)
			if cent > max:
				chosen_node = node
				max = cent
		if chosen_node != -1:
			return chosen_node
		return None
