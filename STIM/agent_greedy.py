from agent import *
import constants

class Greedy(Agent):

	def __init__(self, id, pct_max_steps):
		super().__init__(id, pct_max_steps)
	
	#---------------------------------------------------------------------------
	def _select_node(self):
		eligible_node, degree = self.graph.get_witholding_nodes()
		max = -1
		chosen_node = -1
		for node, d in zip(eligible_node, degree):
			if d > max:
				chosen_node = node
				max = d
		if chosen_node != -1:
			return chosen_node
		return None


