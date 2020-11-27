import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


matrix = np.array( [[255,    0,   5408,    0,    0],
					[ 13,    0,   3439,    0,    0],
					[  0,    0,  39628,    0,    0],
					[  0,    0,      0,    0,    0],
					[  0,    0,      0,    0,  264]] )

matrix = np.transpose(matrix).astype('float')

for i in range(matrix.shape[0]):
	sum_row = 1.0
	if matrix[i,:].sum() > 0.0:
		sum_row = matrix[i,:].sum()
	matrix[i,:] = (matrix[i,:] / sum_row) * 100.0

#matrix = (matrix. / matrix.sum(axis=1)[:, np.newaxis]) * 100.0

sn.set(font_scale=1.4)#for label size
sn.heatmap(matrix, annot=True,annot_kws={"size": 13}, fmt='g', cmap="Blues")# font size
figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig('conf_matrix/conf_matrix.png', dpi=100)