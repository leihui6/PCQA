from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap('rainbow')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
d2c = scalarMap.to_rgba
res = np.loadtxt('./data/sample_quality_score_ours.txt')
quality_score = 1/res[:,3]
pc_colors = np.concatenate([res[:,0:3], quality_score.reshape(-1, 1)], axis = 1)
saveFilename = f'./data/sample_quality_score_colored.txt'
np.savetxt(saveFilename, pc_colors, fmt='%1.6f')