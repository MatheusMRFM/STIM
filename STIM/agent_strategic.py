from agent import *
import constants

class Strategic(Agent):

	def __init__(self, id, pct_max_steps):
		super().__init__(id, pct_max_steps)
	
	#---------------------------------------------------------------------------
	def _select_node(self):
		eligible_node, degree = self.graph.get_witholding_nodes()
		chosen_node = -1
		max = -1
		for node, d in zip(eligible_node, degree):
			is_min_max, degree_special = self.graph.is_min_max(node)
			if is_min_max and degree_special > 0.4:
				chosen_node = node
				break
			elif not is_min_max:
				neighbor_special, degree_neighbor = self.graph.check_special_neighbor(node)
				if neighbor_special and degree_neighbor < 0.4:
					chosen_node = node
					max = 100
				elif d > max:
					chosen_node = node
					max = d
		#self.graph.print_prev_degree()
		if chosen_node != -1:
			return chosen_node
		return None

