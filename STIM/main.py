import time, random, threading, sys
import multiprocessing
from agent_greedy import *
from agent_strategic import *
from agent_RL import *
from summary import * 
from agent_Temporal import *

tf.reset_default_graph()

if constants.STRATEGY == constants.NSTEP_QL:
	if len(sys.argv) > 1:

		constants.LOAD = bool(int(sys.argv[1]))
		constants.GPU_ID = int(sys.argv[2])
		constants.NUM_FEATURES = int(sys.argv[3])
		constants.REWARD_TYPE = int(sys.argv[4])
		constants.EMBED_METHOD = int(sys.argv[5])
		constants.GAMMA = float(sys.argv[6])
		constants.MODEL_TO_TRAIN = float(sys.argv[7])
		constants.USE_Q = bool(int(sys.argv[8]))
		constants.NORMALIZE_Q = bool(int(sys.argv[9]))
		constants.LAYERS = int(sys.argv[10])
		constants.USE_NORMALIZE = bool(int(sys.argv[11]))
		constants.NODE_RANK = bool(int(sys.argv[12]))
		constants.N_ATOM = int(sys.argv[13])
		constants.VMIN = float(sys.argv[14])
		constants.VMAX = float(sys.argv[15])
		constants.DISCOUNT_REWARD = bool(int(sys.argv[16]))
		constants.USE_EXP = bool(int(sys.argv[17]))
		constants.DIFUSION_WEIGHT = float(sys.argv[18])
		constants.COUNT_WEIGHT = float(sys.argv[19])
		constants.DEGREE_WEIGHT = float(sys.argv[20])
		constants.USE_DOUBLE_FLOW = bool(int(sys.argv[21]))
		constants.USE_LSTM = bool(int(sys.argv[22]))
		constants.LAMBDA = float(sys.argv[23])
		constants.INSTANCE_NUM = int(sys.argv[24])
		
		REWARD_NAME		= 'Im'
		if constants.REWARD_TYPE == constants.IMMEDIATE_REWARD:
			REWARD_NAME		= 'Im'
		elif constants.REWARD_TYPE == constants.IMMEDIATE_REWARD_NEG:
			REWARD_NAME		= 'Im_n'
		elif constants.REWARD_TYPE == constants.EPISODE_REWARD:
			REWARD_NAME		= 'Ep'
		elif constants.REWARD_TYPE == constants.EPISODE_REWARD_NEG:
			REWARD_NAME		= 'Ep_n'
		elif constants.REWARD_TYPE == constants.SPECIAL_REWARD:
			REWARD_NAME		= 'Sp'
		elif constants.REWARD_TYPE == constants.HIGH_DIFFUSION:
			REWARD_NAME		= 'Hd'
		elif constants.REWARD_TYPE == constants.INFLUENCE:
			REWARD_NAME		= 'In'
		elif constants.REWARD_TYPE == constants.INFLUENCE_NORM:
			REWARD_NAME		= 'InNorm_' + str(constants.DIFUSION_WEIGHT) + '_' + str(constants.COUNT_WEIGHT) + '_' + str(constants.DEGREE_WEIGHT)
		else:
			REWARD_NAME		= 'ImR'

		TRAIN_NAME		= 'RL'
		if constants.MODEL_TO_TRAIN == constants.SUP_MODEL:
			TRAIN_NAME		= 'SUP'
		elif constants.MODEL_TO_TRAIN == constants.ALL_MODEL:
			TRAIN_NAME		= 'ALL'

		LAYER_NAME = '2l'
		if constants.LAYERS == 1:
			LAYER_NAME = '1l'
		elif constants.LAYERS == 3:
			LAYER_NAME = '3l'
		elif constants.LAYERS == 4:
			LAYER_NAME = '4l'
		elif constants.LAYERS > 4:
			print("Max layers > 4!")
			exit(0)

		constants.DELTA_Z = (constants.VMAX - constants.VMIN) / float(constants.N_ATOM - 1)
		constants.ATOMS = [constants.VMIN + i * constants.DELTA_Z for i in range(constants.N_ATOM)]

		Q_NAME = 'Qnorm'
		if not constants.USE_Q:
			Q_NAME = 'NoQ'
		elif not constants.NORMALIZE_Q:
			Q_NAME = 'Q'

		DISCOUNT_EXTRA = ''
		if not constants.DISCOUNT_REWARD:
			DISCOUNT_EXTRA = '_DIS=' + str(constants.DISCOUNT_REWARD)

		FLOW_EXTRA = ''
		if constants.USE_DOUBLE_FLOW:
			FLOW_EXTRA = '_2Flow'
		if not constants.USE_LSTM:
			FLOW_EXTRA = '_SEQ'

		METHOD_NAME = 'GCN'
		if constants.EMBED_METHOD == constants.S2VEC:
			METHOD_NAME = 'S2VEC'
		elif constants.EMBED_METHOD == constants.S2VEC_NORM:
			METHOD_NAME = 'S2VEC_NORM'
		
		constants.SUMMARY_NAME	= 	METHOD_NAME + '_' + Q_NAME + '_VAR_' + LAYER_NAME + '_' + TRAIN_NAME + '_NORM=' + \
									str(constants.USE_NORMALIZE) + '_F=' + str(constants.NUM_FEATURES) + \
									'_M=' + str(constants.MAX_TO_KEEP) + '_Batch=' + str(constants.BATCH_SIZE) + \
									'_R=' + str(int(constants.NODE_RANK)) + '_G=' + \
									str(int(constants.GAMMA*100)) + '_' + REWARD_NAME + '_LAMBDA=' + \
									"{:.0e}".format(constants.LAMBDA) + '_ATOMS=' + str(constants.N_ATOM) + \
									DISCOUNT_EXTRA + '_EXP=' + str(constants.USE_EXP) + FLOW_EXTRA  +  '_LR=' + \
									"{:.0e}".format(constants.LEARNING_RATE) + '_MinLR=' + "{:.0e}".format(constants.MIN_LEARNING_RT) + \
									str(constants.INSTANCE_NUM)
		constants.MODEL_PATH		= './model/' + constants.SUMMARY_NAME + '/'
		print(constants.SUMMARY_NAME)

		if constants.MODE != constants.TRAIN:
			constants.EXPLORATION_RT = 0.3
		

	n_steps = tf.Variable(0,dtype=tf.int32,name='n_steps',trainable=False)
	graph_count = tf.Variable(0,dtype=tf.int32,name='graph_count',trainable=False)
	config = tf.ConfigProto()
	config.allow_soft_placement=True 
	config.log_device_placement=False
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as session:
		"""
		Creates the master worker that maintains the master network.
		We then initialize the workers array.
		"""
		th_file = constants.MODEL_PATH + 'th_file.csv'
		summary_writer = tf.summary.FileWriter(constants.SUMMARY_FODLER+constants.SUMMARY_NAME)
		summary = Summary(constants.SUMMARY_NAME, summary_writer, constants.SUMMARY_INT_SCR, constants.SUMMARY_INT_LOSS, th_file)
		with tf.device("/cpu:0"):
			master_worker = NStep_QL_agent(	'global', n_steps, graph_count, session, constants.LEARNING_RATE, constants.GAMMA, 
											constants.EXPLORATION_RT, constants.BATCH_SIZE, constants.PCT_MAX_STEPS, summary)
			workers = [] 
			for i in range(constants.N_THREADS):
				print (i)
				workers.append(NStep_QL_agent(	i, n_steps, graph_count, session, constants.LEARNING_RATE, constants.GAMMA, 
												constants.EXPLORATION_RT, constants.BATCH_SIZE, constants.PCT_MAX_STEPS, summary))
			saver = tf.train.Saver(max_to_keep=5)

		"""
		Initializes tensorflow variables
		"""
		if constants.LOAD:
			print ("Loading....")
			if not os.path.exists(constants.MODEL_PATH):
				os.mkdir(constants.MODEL_PATH)
			c = tf.train.get_checkpoint_state(constants.MODEL_PATH)
			saver.restore(session,c.model_checkpoint_path)
			print ("Graph loaded!")
		else:
			session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()

		"""
		Initializes the worker threads
		"""
		worker_threads = []
		for i in range(0, constants.N_THREADS):
			t = threading.Thread(target=workers[i].work, args=(coord, saver))
			t.start()
			time.sleep(0.5)
			worker_threads.append(t)

		coord.join(worker_threads)
else:
	if constants.STRATEGY == constants.GREEDY:
		worker = Greedy(0, constants.PCT_MAX_STEPS)
	elif constants.STRATEGY == constants.STRATEGIC_AGENT:
		worker = Strategic(0, constants.PCT_MAX_STEPS)
	else:
		worker = Temp_Degree(0, constants.PCT_MAX_STEPS)
	print("\n\n-------------\n\n")
	worker.work()