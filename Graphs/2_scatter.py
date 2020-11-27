import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pylab as pylab

FOLDER_IN 	= 'stats/train_special/'
#FOLDER_IN 	= 'stats/real_network/'

FONT_SIZE = 20
params = {	'legend.fontsize': FONT_SIZE,
			'axes.labelsize':  FONT_SIZE,
			'axes.titlesize':  FONT_SIZE,
			'xtick.labelsize': FONT_SIZE,
			'ytick.labelsize': FONT_SIZE}
pylab.rcParams.update(params)

csv_file = [f for f in os.listdir(FOLDER_IN) if 'csv' in f]

#font = {'size'   : 16}

#matplotlib.rc('font', **font)

# Forbidden Names
list_names = ['id', 'mean', 'node']

for f in csv_file:
	# Read CSV
	print(f)
	df = pd.read_csv(FOLDER_IN + f, header = 0)

	fig, axes = plt.subplots(figsize=(10,8))
	for column in df.columns:
		if column not in list_names:
			df.plot(x='id', y=column, style='bo', markersize=1, ax=axes, label=None)

	#axes.set_xlim([20,200])
	plt.xlabel('Node ID')
	plt.ylabel('Degree')
	axes.get_legend().remove()
	plt.savefig(FOLDER_IN + f[:-4] + '.png')
	plt.close()