import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
import matplotlib.pylab as pylab

# Constants
BASE_FOLDER = './Summary/'
DPI_FIGURE = 100

# Plot Constants
XLABEL = 'Number of Steps'
YLABEL = 'Mean Value (last 10 steps)'

ROLL_WINDOW = 25

BEST_FILE = 'S2VEC_NORM_NoQ_VAR_2l_RL_NORM=False_F=4_M=5_Batch=8_R=0_G=70_InNorm_1.0_0.5_0.0_LAMBDA=1e-02_ATOMS=21_DIS=False_EXP=True_2Flow_LR=7e-05_MinLR=1e-069.csv'


FONT_SIZE = 14
params = {	'legend.fontsize': FONT_SIZE,
			'axes.labelsize':  FONT_SIZE,
			'axes.titlesize':  FONT_SIZE,
			'xtick.labelsize': FONT_SIZE,
			'ytick.labelsize': FONT_SIZE}
pylab.rcParams.update(params)

# ----------------------------------------------------------------------

# Returns the maximum 'total step size' between all datasets in a specific folder
def find_max_steps(folder):
	# Initial Maximum ('Steps' Column)
	max_val = sys.maxsize

	# List all CSV files in folder
	for f in glob.glob(folder + '*.csv'):
		# Read CSV
		df = pd.read_csv(f, header=0)

		# Update maximum 'total step size'?
		max_val = df['Step'].max() if df['Step'].max() < max_val else max_val
	return max_val

# ----------------------------------------------------------------------

def get_color_list(size):
	# Color List
	colors = plt.rcParams['axes.prop_cycle']
	colors = colors.by_key()['color']

	# Cycle Color List (Length: #CSV Files)
	colors = list(islice(cycle(colors), size))
	return colors

# ----------------------------------------------------------------------

# Plots individual graphics and chart set
def plot_graphics(folder, max):
	# Chart Set Configuration
	fig_set, ax_set = plt.subplots()
	plt.xlabel(XLABEL)
	plt.ylabel(YLABEL)

	# List all CSV files in folder
	files = glob.glob(folder + '*.csv')

	# Color List
	colors = get_color_list(len(files))
	color_gray = 'gray'
	color_best = 'tab:red' #'tab:blue'

	for i,f in enumerate(files):
		# Read & Clip CSV
		color = color_gray
		#color = colors[i]
		alpha = 0.4
		width = 2
		if BEST_FILE in f:
			color = color_best
			alpha = 1.0
			width = 4.0
		print(f)
		print(color)
		df = pd.read_csv(f, header=0)
		df['smooth_y'] = df['Value'].rolling(window=ROLL_WINDOW,center=False).mean()
		df_subset = df[df['Step'] <= max]

		# Individual Plot
		fig, ax = plt.subplots()
		plt.xlabel(XLABEL)
		plt.ylabel(YLABEL)
		#df.plot(x='Step', y='smooth_y', label='DF #' + str(i), ax=ax, color=colors[i])
		df.plot(x='Step', y='smooth_y', label=None, ax=ax, color=color_best)
		ax.legend(loc='upper right')
		ax.get_legend().remove()
		fig.savefig(f[:-4] + '.png', dpi=DPI_FIGURE)
		plt.close(fig)

		# Chart Set Append
		df_subset.plot(x='Step', y='smooth_y', label=None, ax=ax_set, color=color, alpha=alpha, lw=width, grid=True)
	# Save & Close Chart Set
	ax_set.get_legend().remove()
	ax.set_axisbelow(True)
	fig_set.savefig(folder + 'combined.png', dpi=DPI_FIGURE)
	plt.close(fig_set)

# ----------------------------------------------------------------------

# Find maximum 'total step size' between datasets
max_steps = find_max_steps(BASE_FOLDER)
print('Max Steps: ' + str(max_steps))

# Plot individual fraphics and chart set
plot_graphics(BASE_FOLDER, max_steps)
