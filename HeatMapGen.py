import numpy as np
from matplotlib import pyplot as plt
import time
from keras.models import load_model

show_cropped_point_cloud = False
show_height_map = True

def HeatMap(p_cloud, scan_y_range=[-0.85,0.85], x_res = 100):
    global a
    episode_heatmaps = []
    actions = []
    states = []
    ep_counter = 1

    # --- define the number of scan stripes according to scan_y_range:
    stripe_range = np.arange(scan_y_range[0], scan_y_range[1], 0.1)
    num_of_stripes = len(stripe_range) - 1

    start_time = time.time()

    point_cloud = p_cloud
    if len(point_cloud) < 10000:
        print('Sparse Point Cloud, no Heatmap generated')
        return

    # --- convert to env coordinate system:
    z = -point_cloud[:, 0]
    x = point_cloud[:, 1]
    y = -point_cloud[:, 2]

    # --- cut frame according to arena size
    ind = np.where((x < -1.5) | (x > 1.5) | (z > -2) | (y > 0.8) | (y < -0.8))
    x = np.delete(x, ind)
    y = np.delete(y, ind)
    z = np.delete(z, ind)

    # ---normalize height [0,1]
    z = (z - np.min(z)) / np.ptp(z)

    if show_cropped_point_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, s=0.1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    pc_len = len(x)

    h_map = np.zeros([x_res,num_of_stripes])

    x_A_coeff = (x_res-1)/3
    x_B_coeff = (x_res-1)/2

    y_A_coeff = (num_of_stripes-1)/1.6
    y_B_coeff = (num_of_stripes-1)/2

    for point in range(pc_len):
        x_ind = int(np.round(x[point]*x_A_coeff+x_B_coeff))
        y_ind = int(np.round(y[point]*y_A_coeff+y_B_coeff))
        if z[point] > h_map[x_ind,y_ind]:
            h_map[x_ind,y_ind] = z[point]

    if show_height_map:
        plt.imshow(h_map)
        plt.show(block=False)
        plt.pause(0.001)

    frame_time = time.time() - start_time

    return h_map, frame_time
