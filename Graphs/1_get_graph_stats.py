import networkx as nx
import numpy as np
import os
import pandas as pd

#FOLDER_IN 	= 'Real_Networks/'
#FOLDER_OUT 	= 'stats/real_network/'
FOLDER_IN 	= 'Train/'
FOLDER_OUT 	= 'stats/train_special/'

MAX_FILE	= 30

subfolders = [f for f in os.listdir(FOLDER_IN) if not os.path.isfile(FOLDER_IN + f)]

cont = 0
for folder in subfolders:
	if MAX_FILE <= 0 or cont < MAX_FILE:
		full_folder = FOLDER_IN + folder
		files = [f for f in os.listdir(full_folder) if '.graph6' in f]

		df = pd.DataFrame()

		node_list = None
		degree_mat = []
		for graph_file in files:
			graph = nx.read_graph6(full_folder + '/' + graph_file)
			degree = {node:val for (node, val) in graph.degree()}
			#print(degree)
			if node_list is None:
				node_list = degree.keys()
			degree_list = []
			for node in node_list:
				#degree_list.append(float(degree[node]) / float(len(node_list)))
				degree_list.append(degree[node])
			degree_mat.append(degree_list)

		degree_mat = np.array(degree_mat)
		mean_degree = np.mean(degree_mat, axis=0)

		df['node'] = node_list
		df['mean'] = mean_degree
		for i in range(len(degree_mat)):
			df['t'+str(i)] = degree_mat[i]

		df.sort_values(by=['mean'], ascending=False, inplace=True)
		df.reset_index(drop=True, inplace=True)
		df.index.name = 'id'
		print(df.index.name)

		df.to_csv(FOLDER_OUT + folder + '.csv')
		cont += 1
