import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pc1 = np.load('/home/graphics/git/SmartLoader/saved_experts/test/PointCloud22-04-H10_39_18-1.gzip.npz')['arr_0']
pc2 = np.load('/home/graphics/git/SmartLoader/saved_experts/test/PointCloud22-04-H10_39_22-58.gzip.npz')['arr_0']
pc3 = np.load('/home/graphics/git/SmartLoader/saved_experts/test/PointCloud22-04-H10_39_31-130.gzip.npz')['arr_0']

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
xs = pc2[:,0]
ys = pc2[:,1]
zs = pc2[:,2]
ax.scatter(xs, ys, zs, s=0.1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()