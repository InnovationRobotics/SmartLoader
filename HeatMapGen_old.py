import numpy as np
from matplotlib import pyplot as plt
import time
from keras.models import load_model

show_cropped_point_cloud = False
show_selected_stripes = True
show_total_height_map = False
show_height_map = False

a=0

def HeatMap(p_cloud, scan_y_range=[-0.85,0.85], x_res = 100):
    global a
    episode_heatmaps = []
    actions = []
    states = []
    ep_counter = 1

    # --- define the number of scan stripes according to scan_y_range:
    stripe_range = np.arange(scan_y_range[0], scan_y_range[1], 0.1)
    num_of_stripes = len(stripe_range) - 1
    x_resolution = x_res

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
    ind = np.where((x < -1.4) | (x > 1.5) | (z > -2))
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

    # ---seperate point_cloud to stripe arrays
    st_ind = []  # ---seperate to stripes
    for stripe in range(num_of_stripes):
        st_ind.append(np.where(np.logical_and(y > stripe_range[stripe], y < stripe_range[stripe + 1])))

    shortest = np.inf
    for stripe in range(num_of_stripes):
        stripe_len = len(st_ind[stripe][0])
        if stripe_len < shortest:
            shortest = stripe_len

    # ---populate an array of xyz coordinates for each stripe
    st = np.zeros([num_of_stripes, 3, shortest])
    for i in range(num_of_stripes):
        st[i, 0, :] = x[st_ind[i][0][0:shortest]]
        st[i, 1, :] = y[st_ind[i][0][0:shortest]]
        st[i, 2, :] = z[st_ind[i][0][0:shortest]]

    # ---sort stripes according to x values
    for j in range(num_of_stripes):
        st_test_ind = np.argsort(st[j, 0])
        st[j][0][:] = st[j][0][st_test_ind]
        st[j][1][:] = st[j][1][st_test_ind]
        st[j][2][:] = st[j][2][st_test_ind]

    if show_selected_stripes:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(st[0:num_of_stripes, 0, :], st[0:num_of_stripes, 1, :], st[0:num_of_stripes, 2, :], s=0.1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    if show_total_height_map:

        # rows, cols = 1, 5
        # fig, splot = plt.subplots(rows, cols)
        #
        # for subp in range(cols):
        map_len = len(st[0,0,:])
        full_map = np.zeros([map_len,num_of_stripes])
        for stri in range(num_of_stripes):
            full_map[:,stri]=st[stri][2][:]
        #
        #     splot[subp].imshow(full_map,aspect=0.01)

        plt.imshow(full_map,aspect=0.01)
        # plt.scatter(pos[0],pos[1],s=100,c='red',marker='o')
        plt.show(block=False)

        plt.pause(0.01)

    # # ----create the avareged heatmap:
    h_map = np.zeros([x_resolution, num_of_stripes])
    jump = int(shortest / x_resolution)
    for jj in range(num_of_stripes):  ## iterate over each stripe
        for k in range(x_resolution):  ## crate a 100*3 heatmap
            max_val = np.max(st[jj, 2, k * jump:((k + 1) * jump)])
            h_map[k, jj] = max_val

    # ----create full heatmap:
    # map_len = len(st[0, 0, :])
    # h_map = np.zeros([map_len, num_of_stripes])
    # for stri in range(num_of_stripes):
    #     h_map[:, stri] = st[stri][2][:]

    if show_height_map:
        # model = load_model('/home/sload/Downloads/pos_est_model')
        # pos = model.predict(h_map.reshape(1, 1, 100, 7))[0]

        plt.imshow(h_map)
        # plt.scatter(pos[0],pos[1],s=100,c='red',marker='o')
        plt.show(block=False)

        plt.pause(0.01)
        # # plt.close()


    frame_time = time.time() - start_time

    return h_map, frame_time