import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NO_NORMALIZATION= 0
NORMALIZE_NODE 	= 1
NORMALIZE_EDGE 	= 2
NORMALIZE 		= NO_NORMALIZATION

GRAPHS = ['FB', 'GN31', 'HU', 'GN30', 'HR', 'SDOT', 'RO', 'Email']
COLOR = ['tab:blue', 'tab:orange']
MARKERS = ['o', 'v']
LINE = ['--', '-']

LEGEND = ['Baseline', 'NCA-GE']

#FIELD = 'avg_cluster'
#X_NAME = 'Average Clustering'
#FIELD = 'nodes'
#X_NAME = 'Number of Nodes'
FIELD = 'edges'
X_NAME = 'Number of Edges'
#FIELD = 'avg_degree'
#X_NAME = 'Average Degree'
#FIELD = 'mix'
#X_NAME = 'Number of Nodes + Edges'
#FIELD = 'mix2'
#X_NAME = 'Number of Nodes * Density'

def plot_csv(df, df_graph, color, marker, line):
	y = []
	std = []
	for g in GRAPHS:
		y.append(df[g].mean())
		std.append(df[g].std())
		
	for i in range(len(y)):
		print(y[i], " +- ", std[i])
	print('\n\n')

	if NORMALIZE != NO_NORMALIZATION:
		norm_field = 'nodes'
		if NORMALIZE == NORMALIZE_EDGE:
			norm_field = 'edges'
		norm_value = df_graph[norm_field].values
		for i in range(len(y)):
			y[i] /= norm_value[i]

	if FIELD == 'mix':
		x = df_graph['nodes'].values + df_graph['edges'].values
	elif FIELD == 'mix2':
		x = df_graph['nodes'].values * df_graph['density'].values
	else:
		x = df_graph[FIELD].values

	order = np.argsort(x)
	x = np.array(x)[order]
	y = np.array(y)[order]

	for i in range(x.shape[0]):
		print(x[i], " - ", y[i])
	print('\n\n')

	line = plt.errorbar(x, y, std, color=color, linestyle=line, marker=marker, linewidth=2)


df_base = pd.read_csv('time_base.csv', header=0)
df_s2vec = pd.read_csv('time_s2vec.csv', header=0)
df_graph = pd.read_csv('graph_data.csv', header=0)
df_graph['density'] = df_graph['edges'] / (df_graph['nodes'] * df_graph['nodes'])
df_graph['density_node'] = df_graph['density'] * df_graph['nodes']
df_graph['density_edge'] = df_graph['density'] * df_graph['edges']

plot_csv(df_base, df_graph, COLOR[0], MARKERS[0], LINE[0])
plot_csv(df_s2vec, df_graph, COLOR[1], MARKERS[1], LINE[1])

plt.xlabel(X_NAME)
plt.ylabel('Time (s)')
plt.legend((LEGEND), loc='lower right')
extra = ''
if NORMALIZE == NORMALIZE_NODE:
	extra = 'norm_node_'
elif NORMALIZE == NORMALIZE_EDGE:
	extra = 'norm_edge_'
plt.savefig(extra + 'time_' + FIELD + '.png')

df_graph.to_csv('graph_data.csv', index=False)
