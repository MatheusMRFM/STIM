from agent import *
from neural_net import *
from graph import *
import copy 
import math
import time

class NStep_QL_agent(Agent):

	def __init__(self, id, n_steps, graph_count, session, learning_rate, gamma, exp_rt, batch_size, pct_max_steps, summary):
		super().__init__(id, pct_max_steps)
		
		self.name = "worker_" + str(id)
		self.session = session
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.possible_actions = None
		self.batch_size = batch_size
		self.exp_rt = exp_rt

		self.selected_type_score = np.zeros((5,5), dtype=int)

		self.positive_exp = []

		self.max_difusion_r = None
		self.max_count_r = {'neg':None, 'pos':None}
		self.max_degree_r = {'neg':None, 'pos':None}
		self.min_difusion_r = None
		self.min_count_r = {'neg':None, 'pos':None}
		self.min_degree_r = {'neg':None, 'pos':None}
		if constants.LOAD and constants.REWARD_TYPE == constants.INFLUENCE_NORM:
			self._load_min_max_value()

		self.mean_dist_type = []
		for i in range(5):
			self.mean_dist_type.append([])

		self.summary = summary

		self.n_steps = n_steps
		self.increse_n_steps = self.n_steps.assign_add(1)
		self.graph_count = graph_count
		self.increse_graph_count = self.graph_count.assign_add(1)

		self.local_net = Neural_Net(self.name, session, learning_rate, constants.LR_DECAY, gamma, constants.NUM_FEATURES)
		self.update_local_net = self.local_net.update_network_op('worker_global')
		if constants.LOAD:
			self.summary.load_threashold_file()

	#---------------------------------------------------------------------------
	def _load_min_max_value(self):
		f = open(constants.MODEL_PATH + 'min_max_value.txt', 'r')
		self.min_count_r['neg'] = float(f.readline())
		self.min_count_r['pos'] = float(f.readline())
		self.min_degree_r['neg'] = float(f.readline())
		self.min_degree_r['pos'] = float(f.readline())
		self.min_difusion_r = float(f.readline())
		self.max_count_r['neg'] = float(f.readline())
		self.max_count_r['pos'] = float(f.readline())
		self.max_degree_r['neg'] = float(f.readline())
		self.max_degree_r['pos'] = float(f.readline())
		self.max_difusion_r = float(f.readline())
		print(self.min_count_r)
		print(self.min_degree_r)
		print(self.min_difusion_r)
		print(self.max_count_r)
		print(self.max_degree_r)
		print(self.max_difusion_r)
		f.close()

	#---------------------------------------------------------------------------
	def _load_next_graph_wtf(self):
		f = open(constants.MODEL_PATH + 'min_max_value.txt', 'r')
		self.min_count_r = float(f.readline())
		self.min_degree_r = float(f.readline())
		self.min_difusion_r = float(f.readline())
		self.max_count_r = float(f.readline())
		self.max_degree_r = float(f.readline())
		self.max_difusion_r = float(f.readline())
		f.close()

	#---------------------------------------------------------------------------
	def _load_next_graph(self):
		super()._load_next_graph() 
		self.possible_actions = np.arange(self.graph.n_nodes, dtype=np.int)
		print('N, max_t, steps = (%d, %d)' % (self.graph.n_nodes, self.graph.max_time))

	#---------------------------------------------------------------------------
	def _reset_hidden_states(self, num_nodes):
		self.hidden_states = {}
		self.hidden_states['batch_flow_1'] = np.zeros((constants.LSTM_LAYERS_F1, 2, num_nodes, constants.LSTM_SIZE_F1), np.float32)
		self.hidden_states['batch_flow_2'] = np.zeros((constants.LSTM_LAYERS_F2, 2, num_nodes, constants.LSTM_SIZE_F2), np.float32)
		self.hidden_states['step_flow_1'] = np.zeros((constants.LSTM_LAYERS_F1, 2, num_nodes, constants.LSTM_SIZE_F1), np.float32)
		self.hidden_states['step_flow_2'] = np.zeros((constants.LSTM_LAYERS_F2, 2, num_nodes, constants.LSTM_SIZE_F2), np.float32)

	#---------------------------------------------------------------------------
	def _build_Q_vec(self):
		id_vec = []
		Qdist = None
		Qs_a = None
		self._update_global_batch(update_hc=False)
		self.batch.set_hidden_states(self.hidden_states['batch_flow_1'], self.hidden_states['batch_flow_2'])
		q_dist, q = self.local_net.get_Qs_a(self.session, self.batch)
		for i in range(self.graph.n_nodes):
			if self.graph.check_node_withold(i):
				id_vec.append(i)
				single_q = np.expand_dims(q[i], axis=0)
				single_qdist = np.expand_dims(q_dist[i], axis=0)
				if Qs_a is None:
					Qs_a = single_q											#1x1
					Qdist = single_qdist									#1xA
				else:
					Qs_a = np.concatenate((Qs_a, single_q), axis=0)			#Yx1
					Qdist = np.concatenate((Qdist, single_qdist), axis=0)	#YxA

		return id_vec, Qs_a, Qdist

	#---------------------------------------------------------------------------
	def _build_Q_vec_single_step(self):
		id_vec = []
		Qdist = None
		Qs_a = None
		A, X = self.graph.get_adj_feat_matrix_step_t()
		batch = Batch(self.graph.n_nodes)
		batch.add_info(A, X, None, 0, 0, 0)
		batch.set_hidden_states(self.hidden_states['step_flow_1'], self.hidden_states['step_flow_2'])
		q_dist, q = self.local_net.get_Qs_a(self.session, batch)
		for i in range(self.graph.n_nodes):
			if self.graph.check_node_withold(i):
				id_vec.append(i)
				single_q = np.expand_dims(q[i], axis=0)
				single_qdist = np.expand_dims(q_dist[i], axis=0)
				if Qs_a is None:
					Qs_a = single_q											#1x1
					Qdist = single_qdist									#1xA
				else:
					Qs_a = np.concatenate((Qs_a, single_q), axis=0)			#Yx1
					Qdist = np.concatenate((Qdist, single_qdist), axis=0)	#YxA

		return id_vec, Qs_a, Qdist

	#---------------------------------------------------------------------------
	def _build_action_array(self):
		if constants.STEP_ONE:
			id_vec, Q, Qdist = self._build_Q_vec_single_step()
		else:
			id_vec, Q, Qdist = self._build_Q_vec()

		if Q is None:
			return None, None, None, None
		index_sort = np.argsort(Q)[::-1]
		Q_index_real = []
		for index in index_sort:
			Q_index_real.append(id_vec[index])

		#node_type = []
		node_type = np.zeros(5)
		for i in range(len(index_sort)):
			#node_type.append(self.graph.get_node_type(id_vec[index_sort[i]]))
			nt = self.graph.get_node_type(id_vec[index_sort[i]])
			node_type[nt] += 1
			self.mean_dist_type[nt].append(Qdist[index_sort[i]])
			if len(self.mean_dist_type[nt]) > 500:
				self.mean_dist_type[nt] = self.mean_dist_type[nt][1:]

		'''print(node_type)
		print('chosen type = ', self.graph.get_node_type(id_vec[index_sort[0]]))
		print('Q = ', Q)
		print('Q index_real = ', Q_index_real)'''
		degree = self.graph._get_degree(id_vec[index_sort[0]])
		print('best node type = ', self.graph.get_node_type(id_vec[index_sort[0]]), '  ', node_type, ' --- (', degree, ')')
		#print("Diff Q = ", Q[index_sort[0]] - Q[index_sort[len(Q)-1]])

		if constants.MODE == constants.TRAIN or constants.TEST:
			node_strategy, node_type_count = self._select_node_strategic()
			if node_strategy == constants.NO_ACTION:
				print("\n\nNO ACTION\n\n")
			else:
				best_type = self.graph.get_node_type(node_strategy)
				selected_type = self.graph.get_node_type(id_vec[index_sort[0]])
				self.selected_type_score[selected_type, best_type] += 1

		return Q, index_sort, Q_index_real, Qdist

	#---------------------------------------------------------------------------
	def _get_bootstrap(self, is_final_state):
		if is_final_state or constants.BEST_ACTION:
			return 0
		Q, Q_index, _, _ = self._build_action_array()
		'''Get the highest Q(s', a')'''
		if Q is None:
			return 0
		Qmax = Q[Q_index[0]]
		return Qmax

	#---------------------------------------------------------------------------
	def _process_batch_train(self, batch):
		lrl = 0.0
		lsup = 0.0
		y_true = None
		y_pred = None
		if constants.MODEL_TO_TRAIN == constants.RL_MODEL:
			return_list = self.local_net.train_network_rl(self.session, batch)
			lrl = return_list[0]
			self.hidden_states['batch_flow_1'] = return_list[1]
			self.hidden_states['batch_flow_2'] = return_list[2]
			self.hidden_states['step_flow_1'] = return_list[1]
			self.hidden_states['step_flow_2'] = return_list[2]
		elif constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
			y_true = batch.true_cent
			return_list = self.local_net.train_network_all(self.session, batch)
			lrl = return_list[0]
			lsup = return_list[1]
			y_pred = return_list[2]
			self.hidden_states['batch_flow_1'] = return_list[3]
			self.hidden_states['batch_flow_2'] = return_list[4]
			self.hidden_states['step_flow_1'] = return_list[3]
			self.hidden_states['step_flow_2'] = return_list[4]
		elif constants.MODEL_TO_TRAIN == constants.SUP_MODEL:
			y_true = batch.true_cent
			return_list = self.local_net.train_network_sup(self.session, batch)
			lsup = return_list[0]
			y_pred = return_list[1]
			self.hidden_states['batch_flow_1'] = return_list[2]
			self.hidden_states['batch_flow_2'] = return_list[3]
			self.hidden_states['step_flow_1'] = return_list[2]
			self.hidden_states['step_flow_2'] = return_list[3]

		return lrl, lsup, y_true, y_pred

	#---------------------------------------------------------------------------
	def _process_batch_test(self, batch):
		lrl = 0.0
		lsup = 0.0
		if constants.MODEL_TO_TRAIN == constants.RL_MODEL:
			return_list = self.local_net.test_network_rl(self.session, batch)
			lrl = return_list[0]
			self.hidden_states['batch_flow_1'] = return_list[1]
			self.hidden_states['batch_flow_2'] = return_list[2]
			self.hidden_states['step_flow_1'] = return_list[1]
			self.hidden_states['step_flow_2'] = return_list[2]
		elif constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
			return_list = self.local_net.test_network_all(self.session, batch)
			lrl = return_list[0]
			lsup = return_list[1]
			self.hidden_states['batch_flow_1'] = return_list[3]
			self.hidden_states['batch_flow_2'] = return_list[4]
			self.hidden_states['step_flow_1'] = return_list[3]
			self.hidden_states['step_flow_2'] = return_list[4]
		elif constants.MODEL_TO_TRAIN == constants.SUP_MODEL:
			return_list = self.local_net.test_network_sup(self.session, batch)
			lsup = return_list[0]
			self.hidden_states['batch_flow_1'] = return_list[2]
			self.hidden_states['batch_flow_2'] = return_list[3]
			self.hidden_states['step_flow_1'] = return_list[2]
			self.hidden_states['step_flow_2'] = return_list[3]

		return lrl, lsup

	#---------------------------------------------------------------------------
	def _select_best_action(self):
		if self.Q is None:
			return constants.NO_ACTION
		action = self.Q_index_real[0]

		return action

	#---------------------------------------------------------------------------
	def _select_random_action(self):
		eligible_node, _ = self.graph.get_witholding_nodes()
		if len(eligible_node) == 0:
			action = constants.NO_ACTION
		else:
			rnd = np.random.randint(0, len(eligible_node))
			action = eligible_node[rnd]

		return action

	#---------------------------------------------------------------------------
	def _select_suboptimal_action(self):
		eligible_node, degree = self.graph.get_witholding_nodes()
		if len(eligible_node) == 0:
			action = constants.NO_ACTION
		else:
			index_high_degree = np.argmax(degree)
			action = eligible_node[index_high_degree]

		return action

	#---------------------------------------------------------------------------
	def _select_node_strategic(self):
		node_type_count = [0,0,0,0,0]
		eligible_node, degree = self.graph.get_witholding_nodes()
		chosen_node = -1
		max = -1
		for node, d in zip(eligible_node, degree):
			node_type = self.graph.get_node_type(node)
			node_type_count[node_type] += 1
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
			return chosen_node, node_type_count
		return constants.NO_ACTION, node_type_count

	#---------------------------------------------------------------------------
	def _select_node(self):
		rnd_num = float(np.random.randint(0, 1000)) / 1000.0
		self.Q, self.Q_index, self.Q_index_real, self.Qdist = self._build_action_array()
		if self.Q is None:
			best_q_dist = 0.0
		else:
			best_q_dist = self.Qdist[self.Q_index[0]]
		if rnd_num < self.exp_rt:
			action = self._select_random_action()
		else:
			if constants.BEST_ACTION:
				action = self._select_best_action()
			else:
				#action = self._select_suboptimal_action()
				action, _ = self._select_node_strategic()

		return action, best_q_dist

	#---------------------------------------------------------------------------
	def _update_lstm_state(self, step=False):
		A, X = self.graph.get_adj_feat_matrix_step_t()
		batch = Batch(self.graph.n_nodes)
		batch.add_info(A, X, None, 0, 0, 0)
		if step:
			batch.set_hidden_states(self.hidden_states['step_flow_1'], self.hidden_states['step_flow_2'])
			return_list = self.local_net.get_hidden_state(self.session, batch)
			self.hidden_states['step_flow_1'] = return_list[0]
			self.hidden_states['step_flow_2'] = return_list[1]
		else:
			batch.set_hidden_states(self.hidden_states['batch_flow_1'], self.hidden_states['batch_flow_2'])
			return_list = self.local_net.get_hidden_state(self.session, batch)
			self.hidden_states['batch_flow_1'] = return_list[0]
			self.hidden_states['batch_flow_2'] = return_list[1]
			self.hidden_states['step_flow_1'] = return_list[0]
			self.hidden_states['step_flow_2'] = return_list[1]

	#---------------------------------------------------------------------------
	def _update_global_batch(self, update_hc=False):
		A, X = self.graph.get_adj_feat_matrix_step_t()
		self.batch.add_info(A, X, None, 0, 0, 0)
		if self.batch.size == constants.BATCH_SIZE and update_hc:
			self.batch.set_hidden_states(self.hidden_states['batch_flow_1'], self.hidden_states['batch_flow_2'])
			return_list = self.local_net.get_hidden_state(self.session, self.batch)
			self.hidden_states['batch_flow_1'] = return_list[0]
			self.hidden_states['batch_flow_2'] = return_list[1]

	#---------------------------------------------------------------------------
	def _immediate_reward(self, node):
		dif_prev, _ = self.graph.get_current_difusion()
		is_final_step = self.graph.difuse_step(node)
		dif, _ = self.graph.get_current_difusion()
		reward = float(dif - dif_prev)
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _immediate_reward_neg(self, node):
		dif_prev, _ = self.graph.get_current_difusion()
		is_final_step = self.graph.difuse_step(node)
		dif, _ = self.graph.get_current_difusion()
		reward = float(dif - dif_prev)
		if is_final_step and reward < 0.2:
			reward = 0.0
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _episode_reward(self, node):
		reward = 0.0
		is_final_step = self.graph.difuse_step(node)
		if is_final_step:
			reward, _ = self.graph.get_current_difusion()
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _episode_reward_neg(self, node):
		reward = 0.0
		is_final_step = self.graph.difuse_step(node)
		if is_final_step:
			reward, _ = self.graph.get_current_difusion()
			if reward < 0.2:
				reward = 0.0
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _influence_reward(self, node):
		degree_prev = 0.0
		if not node is None:
			degree_prev = self.graph.get_simple_degree(node)
		dif_prev, _ = self.graph.get_current_difusion()
		count_prev = self.graph.get_influence_count()

		is_final_step = self.graph.difuse_step(node)

		degree = 0.0
		if not node is None:
			degree = self.graph.get_simple_degree(node)
		dif, _ = self.graph.get_current_difusion()
		count = self.graph.get_influence_count()

		dif_degree = float(degree - degree_prev) * constants.DEGREE_WEIGHT
		dif_reward = float(dif - dif_prev) * constants.DIFUSION_WEIGHT
		count_reward = float(count - count_prev) * constants.COUNT_WEIGHT

		#dif_reward = math.log(dif_reward*20 + 1,2) - 1

		reward = (dif_reward + count_reward - dif_degree) / (constants.DEGREE_WEIGHT + constants.DIFUSION_WEIGHT + constants.COUNT_WEIGHT)
		reward = min(max(reward, constants.VMIN), constants.VMAX)
		print("\t -> %f + %f - %f ===> %f" % (dif_reward, count_reward, dif_degree, reward))

		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _update_min_max(self, value, min_dict, max_dict):
		term = 'pos'
		mult = 1.0
		if value < 0.0:
			term = 'neg'
			mult = -1.0
			value = mult * value
		if max_dict[term] is None:
			max_dict[term] = value + 0.0001
			min_dict[term] = value
		else:
			if value > max_dict[term]:
				max_dict[term] = value
			elif value < min_dict[term]:
				min_dict[term] = value
		
		#print(term + "==> %f * (%f - %f) / (%f - %f)" % (mult, value, min_dict[term], max_dict[term], min_dict[term]))
		value = mult * (value - min_dict[term]) / (max_dict[term] - min_dict[term])

		return value, min_dict, max_dict

	#---------------------------------------------------------------------------
	def _influence_norm_reward(self, node):
		degree_prev = 0.0
		if not node is None:
			degree_prev = self.graph.get_simple_degree(node)
		dif_prev, _ = self.graph.get_current_difusion()
		count_prev = self.graph.get_influence_count()

		is_final_step = self.graph.difuse_step(node)

		degree = 0.0
		if not node is None:
			degree = self.graph.get_simple_degree(node)
		dif, _ = self.graph.get_current_difusion()
		count = self.graph.get_influence_count()

		dif_degree = float(degree - degree_prev)
		dif_reward = float(dif - dif_prev)
		count_reward = float(count - count_prev)

		#print('count')
		count_reward, self.min_count_r, self.max_count_r = self._update_min_max(count_reward, self.min_count_r, self.max_count_r)
		count_reward = constants.COUNT_WEIGHT * count_reward

		#print('degree')
		dif_degree, self.min_degree_r, self.max_degree_r = self._update_min_max(dif_degree, self.min_degree_r, self.max_degree_r)
		dif_degree = -1 * constants.DEGREE_WEIGHT * dif_degree

		if self.max_difusion_r is None:
			self.max_difusion_r = dif_reward + 0.0001
			self.min_difusion_r = dif_reward
		else:
			if dif_reward > self.max_difusion_r:
				self.max_difusion_r = dif_reward
			elif dif_reward < self.min_difusion_r:
				self.min_difusion_r = dif_reward
		dif_reward = constants.DIFUSION_WEIGHT * (dif_reward - self.min_difusion_r) / (self.max_difusion_r - self.min_difusion_r)

		#print("%f - %f" % (self.min_degree_r, self.max_degree_r))
		#print("%f - %f" % (self.min_count_r, self.max_count_r))
		#print("%f - %f" % (self.min_difusion_r, self.max_difusion_r))

		reward = (dif_degree + dif_reward + count_reward) / (constants.DEGREE_WEIGHT + constants.DIFUSION_WEIGHT + constants.COUNT_WEIGHT)
		#reward = dif_degree + dif_reward + count_reward	#dont normalize reward
		reward = min(constants.VMAX, max(constants.VMIN, reward))
		print("\t -> difusion_reward = %f (%f -> %f)" % (dif_reward, dif_prev, dif))
		print("\t -> count_reward    = %f (%f -> %f)" % (count_reward, count_prev, count))
		print("\t -> degree_reward   = %f (%f -> %f)" % (dif_degree, degree_prev, degree))
		print("\t -> %f + %f + %f ===> %f" % (dif_reward, count_reward, dif_degree, reward))

		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _special_reward(self, node):
		if node is None:
			is_final_step = self.graph.difuse_step(node)
			return 0, is_final_step

		node_type = self.graph.get_node_type(node)
		if node_type == 0:
			dif_prev, _ = self.graph.get_current_difusion()
			is_final_step = self.graph.difuse_step(node)
			dif, _ = self.graph.get_current_difusion()
			reward = float(dif - dif_prev)
			return reward, is_final_step
		elif node_type == 1:
			reward = -1.0
		elif node_type == 2:
			reward = 0.2
		else:
			reward = 1.0

		is_final_step = self.graph.difuse_step(node)
		
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def _high_diffusion_reward(self, node):
		reward = 0
		if node is None:
			is_final_step = self.graph.difuse_step(node)
			return reward, is_final_step

		is_min_max, degree_special = self.graph.is_min_max(node)
		is_final_step = self.graph.difuse_step(node)
		if degree_special > 0.5:
			reward += 1
		#n_withold_high = self.graph.count_connected_node_withold()
		#reward += n_withold_high * 0.2

		return reward-0.2, is_final_step

	#---------------------------------------------------------------------------
	def _immediate_reward_discount(self, node):
		relative_step = 1.0 - (float((self.graph.time_step - self.graph.starting_time)) / float(self.graph.max_steps))

		dif_prev, _ = self.graph.get_current_difusion()
		is_final_step = self.graph.difuse_step(node)
		dif, _ = self.graph.get_current_difusion()
		reward = float(dif - dif_prev) * relative_step
		
		return reward, is_final_step

	#---------------------------------------------------------------------------
	def take_action(self, node):
		if node == constants.NO_ACTION:
			print("\n\nNO ACTION\n\n")
			node = None
		else:
			node_type = self.graph.get_node_type(node)
			print("Selected node = ", node_type)
		if constants.REWARD_TYPE == constants.IMMEDIATE_REWARD:
			reward, is_final_step = self._immediate_reward(node)
		elif constants.REWARD_TYPE == constants.IMMEDIATE_REWARD_NEG:
			reward, is_final_step = self._immediate_reward_neg(node)
		elif constants.REWARD_TYPE == constants.EPISODE_REWARD:
			reward, is_final_step = self._episode_reward(node)
		elif constants.REWARD_TYPE == constants.EPISODE_REWARD_NEG:
			reward, is_final_step = self._episode_reward_neg(node)
		elif constants.REWARD_TYPE == constants.SPECIAL_REWARD:
			reward, is_final_step = self._special_reward(node)
		elif constants.REWARD_TYPE == constants.HIGH_DIFFUSION:
			reward, is_final_step = self._high_diffusion_reward(node)	
		elif constants.REWARD_TYPE == constants.INFLUENCE:
			reward, is_final_step = self._influence_reward(node)
		elif constants.REWARD_TYPE == constants.INFLUENCE_NORM:
			reward, is_final_step = self._influence_norm_reward(node)
		else:
			reward, is_final_step = self._immediate_reward_discount(node)	

		self._update_lstm_state(step=True)

		return reward, is_final_step

	#---------------------------------------------------------------------------
	def save_exp(self, ep_batch_vec):
		self.positive_exp.append(ep_batch_vec)
		if len(self.positive_exp) > constants.MAX_N_EXP:
			_ = self.positive_exp.pop(0)

	#---------------------------------------------------------------------------
	def retrieve_exp(self):
		if len(self.positive_exp) == 0:
			return None
		index = random.randint(0, len(self.positive_exp)-1)
		return self.positive_exp[index]

	#---------------------------------------------------------------------------
	def train_on_exp(self):
		train_success = True
		ep_batch_vec = self.retrieve_exp()

		if ep_batch_vec is None:
			train_success = False
			return train_success

		for i in range(len(ep_batch_vec)):
			batch = ep_batch_vec[i]
			lrl, lsup, y_true, y_pred = self._process_batch_train(batch)
			self.session.run(self.update_local_net)

		return train_success

	#---------------------------------------------------------------------------
	def _update_loss_list(self, loss_rl, loss_sup, ytrue_vec, ypred_vec, lrl, lsup, ytrue, ypred):
		try:
			for x in lrl:
				loss_rl.append(x)
		except:
			loss_rl.append(lrl)
		try:
			for x in lsup:
				loss_sup.append(x)
		except:
			loss_sup.append(lsup)

		if not ytrue is None:
			for i in range(len(ytrue)):
				ytrue_vec.append(ytrue[i])
				ypred_vec.append(ypred[i])

		return loss_rl, loss_sup, ytrue_vec, ypred_vec

	#---------------------------------------------------------------------------
	def work(self, coordinator, saver):
		global test_end
		#while not coordinator.should_stop():
		use_exp_counter = 0
		size_result = {}
		for _ in range(constants.N_GRAPH_ITERATE): 
			if constants.USE_EXP and use_exp_counter >= constants.EXP_RATE:
				success = self.train_on_exp()
				if success:
					use_exp_counter = 0
					continue
			loss_rl = []
			loss_sup = []
			ytrue_vec = []
			ypred_vec = []
			self._load_next_graph()
			self._reset_hidden_states(self.graph.n_nodes)
			n_steps = 0
			if test_end:
				print("TEST END")
				break
			ep_batch_vec = []
			batch = Batch(self.graph.n_nodes)
			is_final_step = False
			end = False
			number_edge_vec = []
			self.batch = Batch(self.graph.n_nodes)
			for i in range(self.graph.max_time):
				if i >= self.graph.starting_time:
					timer_start = time.time()
					A, X = self.graph.get_adj_feat_matrix_step_t()

					number_edge_vec.append(self.graph.graph.number_of_edges())
					if len(number_edge_vec) > constants.BATCH_SIZE:
						number_edge_vec = number_edge_vec[1:]
					
					if constants.MODEL_TO_TRAIN == constants.RL_MODEL or constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
						node, best_q_dist = self._select_node() 
						r, is_final_step = self.take_action(node)
					else:
						node = None
						best_q_dist = None
						r = 0
						is_final_step = self.graph.advance_time_step()
					
					timer_end = time.time()
					#time_file = open('time/' + str(self.graph.n_nodes) + '.dat', "a+")
					time_file = open('time/time_large.dat', "a+")
					mean_n_edge = np.mean(number_edge_vec)
					time_file.write(str(mean_n_edge) + '\t' + str(timer_end - timer_start) + '\n')

					batch.add_info(A, X, node, r, i, best_q_dist) 

					score, end = self.get_score()

					if batch.size >= constants.BATCH_SIZE or is_final_step or end:
						is_final_state = is_final_step or end
						if constants.MODEL_TO_TRAIN == constants.RL_MODEL or constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
							Qmax = self._get_bootstrap(is_final_state)
							batch.add_bootstrap(Qmax)

						if constants.MODEL_TO_TRAIN != constants.RL_MODEL:
							batch.set_node_sup(self.graph)

						batch.set_hidden_states(self.hidden_states['batch_flow_1'], self.hidden_states['batch_flow_2'])

						if constants.MODE == constants.TRAIN:
							lrl, lsup, y_true, y_pred = self._process_batch_train(batch)
							loss_rl, loss_sup, ytrue_vec, ypred_vec = self._update_loss_list(loss_rl, loss_sup, ytrue_vec, ypred_vec, lrl, lsup, y_true, y_pred)
							self.session.run(self.update_local_net)
							n_steps = self.session.run(self.increse_n_steps)
							self._save_model(saver, n_steps)
							if constants.MODEL_TO_TRAIN == constants.RL_MODEL or constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
								if constants.USE_EXP:
									batch_copy = copy.deepcopy(batch)
									ep_batch_vec.append(batch_copy)

						else:
							lrl, lsup = self._process_batch_test(batch)
						batch.reset()
				else:
					if constants.STEP_ONE:
						self._update_lstm_state(step=False)
					else:
						self._update_global_batch(update_hc=False)
					self.graph.advance_time_step()
					
				if is_final_step or end:
					break

			if constants.MODEL_TO_TRAIN == constants.RL_MODEL or constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
				if constants.USE_EXP:
					use_exp_counter += 1
					if score > constants.SCORE_TH:
						self.save_exp(ep_batch_vec)
			
			'''if not self.graph.n_nodes in size_result.keys():
				size_result[self.graph.n_nodes] = []
			size_result[self.graph.n_nodes].append(score)
			print('\n\n')
			for key in size_result.keys():
				if len(size_result[key]) > 0:
					print(key, " ====> ", np.mean(size_result[key]), ' ___ ', np.std(size_result[key]), ' (', len(size_result[key]), ')')'''

			self.mean_reward.append(score)
			mean = np.array(self.mean_reward).mean()
			print('\n\nSCORE: %f ---- MEAN: %f (%d)' % (score, mean, len(self.mean_reward)))
			if constants.MODE == constants.TRAIN or constants.MODE == constants.TEST:
				print(self.selected_type_score)
			if constants.MODE == constants.TRAIN:
				graph_count = self.session.run(self.increse_graph_count)
				self.summary.add_score(score)
				self.summary.write_score(graph_count)
				mean_loss_rl = np.array(loss_rl).mean()
				mean_loss_sup = np.array(loss_sup).mean()
				self.summary.add_loss(mean_loss_rl, mean_loss_sup, ytrue_vec, ypred_vec)
				self.summary.write_loss(graph_count)
				self.exp_rt = self.exp_rt - self.exp_rt * constants.EXP_DECAY
				if self.exp_rt < constants.MIN_EXP_RT:
					self.exp_rt = constants.MIN_EXP_RT
				print("\tNew EXP_DECAY = ", self.exp_rt)

			print("\n\n")

	#---------------------------------------------------------------------------
	def _save_model(self, saver, n_steps):
		if n_steps % constants.SAVER_INTERVAL == 0: 
			print ("Saving model..............")
			if constants.SAVE_NETWORK == True:
				saver.save(self.session, constants.MODEL_PATH+'model.cptk')
				if constants.REWARD_TYPE == constants.INFLUENCE_NORM:
					f = open(constants.MODEL_PATH + 'min_max_value.txt', 'w')
					f.write(str(self.min_count_r['neg']) + '\n')
					f.write(str(self.min_count_r['pos']) + '\n')
					f.write(str(self.min_degree_r['neg']) + '\n')
					f.write(str(self.min_degree_r['pos']) + '\n')
					f.write(str(self.min_difusion_r) + '\n')
					f.write(str(self.max_count_r['neg']) + '\n')
					f.write(str(self.max_count_r['pos']) + '\n')
					f.write(str(self.max_degree_r['neg']) + '\n')
					f.write(str(self.max_degree_r['pos']) + '\n')
					f.write(str(self.max_difusion_r) + '\n')
					f.close()
				self.summary.save_threashold_file()
			print ("Model saved!")
