import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import os, sys
import time, random, threading


GRAPH_NUM		= 0
DAY_INTERVAL	= 10

SECONDS_PER_DAY	= 60*60*24

FILE_DICT 	= {	0:'email-Eu-core-temporal.txt',
				1:'email-Eu-core-temporal-Dept1.txt',
				2:'email-Eu-core-temporal-Dept2.txt',
				3:'email-Eu-core-temporal-Dept3.txt',
				4:'email-Eu-core-temporal-Dept4.txt'}
OUT_FOLDER	= str(GRAPH_NUM) + '/'


#-----------------------------------------------------------------
def convert_seconds_to_day(sec):
	return int(float(sec) / float(SECONDS_PER_DAY))

#-----------------------------------------------------------------
def get_time_step(sec):
	days = convert_seconds_to_day(sec)
	time_step = int(float(days) / float(DAY_INTERVAL))
	return time_step

#-----------------------------------------------------------------
def read_original_file():
	in_vec = []
	out_vec = []
	time_vec = []
	node_list = []
	file_graph = open(FILE_DICT[GRAPH_NUM], 'r')
	for line in file_graph:
		line = line.strip()
		split = line.split(' ')
		in_vec.append(split[0])
		out_vec.append(split[1])
		node_list.append(split[0])
		node_list.append(split[1])
		time_vec.append(get_time_step(split[2]))

	df = pd.DataFrame()
	df['in'] = in_vec
	df['out'] = out_vec
	df['time'] = time_vec

	node_list = np.unique(node_list)

	max_time = np.max(time_vec)
	if max_time != df['time'].nunique():
		print("DIFFERENT VALUES: ", max_time, " - ", df['time'].nunique())

	return df, node_list, max_time
	
#-----------------------------------------------------------------
def get_edge_list(df, time_step):
	valid = True
	df_step = df[df['time'] == time_step]
	in_vec = df_step['in'].values
	out_vec = df_step['out'].values

	print(time_step, " - ", len(in_vec), " edges")
	if len(in_vec) == 0:
		valid = False
		return valid, None

	edge_list = [(x, y) for x, y in zip(in_vec, out_vec)]

	return valid, edge_list 

#-----------------------------------------------------------------
def save_graph(node_list, edge_list, time_step):
	graph = nx.Graph()
	graph.add_nodes_from(node_list)
	graph.add_edges_from(edge_list)
	nx.write_graph6(graph, OUT_FOLDER + str(GRAPH_NUM) + '_' + str(time_step) + '.graph6')



#******************************************************************

df, node_list, max_time = read_original_file()
for i in range(max_time):
	valid, edge_list = get_edge_list(df, i)
	if not valid:
		break
	save_graph(node_list, edge_list, i)





