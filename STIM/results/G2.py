import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
import numpy as np
import matplotlib.pylab as pylab

# Constants
BASE_FOLDER = './final_results/'
BAR_COLOR = 'navy'
DPI_FIGURE = 100

# Plot Constants
XLABEL = 'Model'
YLABEL = 'Score'

FONT_SIZE = 14
params = {	'legend.fontsize': FONT_SIZE,
			'axes.labelsize':  FONT_SIZE,
			'axes.titlesize':  FONT_SIZE,
			'xtick.labelsize': FONT_SIZE,
			'ytick.labelsize': FONT_SIZE}
pylab.rcParams.update(params)

# ----------------------------------------------------------------------

def get_color_list(size):
	# Color List
	colors = plt.rcParams['axes.prop_cycle']
	colors = colors.by_key()['color']

	# Cycle Color List (Length: #CSV Files)
	colors = list(islice(cycle(colors), size))
	return colors

# ----------------------------------------------------------------------

def plot_function(df, colors, filename):
	fig, ax = plt.subplots()
	y_max = np.round( np.max(df[YLABEL].values) * 1.1, decimals=2 )
	plt.xlabel(XLABEL)
	plt.ylabel(YLABEL)
	df.plot.bar(x=XLABEL, y=YLABEL, ax=ax, legend=None, color=colors, grid=True)
	ax.set_axisbelow(True)
	ax.xaxis.grid(False)
	ax.set_ylim([0.0, y_max])
	plt.xticks(rotation=0)
	fig.savefig(filename, dpi=DPI_FIGURE)
	plt.close(fig)

# ----------------------------------------------------------------------

def plot_graphics(folder):
	# List all .txt files in a specified directory + subdirectories (**)
	files = [f for f in glob.glob(folder + "**/*.txt", recursive=True)]

	# Combined Data Frame
	df_set = pd.DataFrame()

	for f in files:
		# Read input file (be careful with the file separator)
		df = pd.read_csv(f, sep='\t', header=None)
		df.columns = [XLABEL, YLABEL]
		
		# Individual Plot (Alphabetically Sorted)
		#df = df.sort_values(XLABEL)
		ind_colors = get_color_list(df.shape[0])
		plot_function(df, ind_colors, f[:-4] + '.png')
		
		# Append Data Frames
		df_set = pd.concat([df_set, df]).drop_duplicates().reset_index(drop=True)
	
	# Combined Plot (Alphabetically Sorted)
	df_set = df_set.sort_values(XLABEL)
	ind_colors = get_color_list(df_set.shape[0])
	plot_function(df_set, ind_colors, folder + 'combined.png')
	print(df_set)

# ----------------------------------------------------------------------

plot_graphics(BASE_FOLDER)


