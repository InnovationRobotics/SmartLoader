import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion

from os import walk
import csv
import time



def joy_to_agent(joy_actions):

    agent_action = np.zeros(4)

    agent_action[0] = joy_actions[0]  # vehicle turn
    agent_action[2] = joy_actions[3]  # blade pitch     ##### reduced state space
    agent_action[3] = joy_actions[4]  # arm up/down
    # translate 5 dim joystick actions to 4 dim agent action
    # agent action: [steer, speed, blade_pitch, arm_height]
    # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]
    # all [-1, 1]
    agent_action[1] = 0.5 * (joy_actions[2] - 1) + 0.5 * (1 - joy_actions[5])  ## forward backward

    return agent_action


jobs = ['heat_map_generator', 'concat_recodrings', 'pos_aprox']
job = jobs[0]

if job == 'heat_map_generator':

    recordings_path = '/home/graphics/push_28_04_all/'
    show_point_cloud = False
    show_height_map = True

    recording_heatmaps = []
    recording_states = []
    recording_actions = []

    ep_counter = 1

    stripe_limits = [-0.8, 0.8]

    stripe_range = np.arange(stripe_limits[0], stripe_limits[1]+0.1, 0.1)
    num_of_stripes = len(stripe_range) - 1
    x_res = 100

    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break
    for ep_dir in dirnames:   ## iterate over each
        for (_, _, filenames) in walk(dirpath+ep_dir):  # create a list of pointcloud steps for each episode
            break

        actions = []
        states = []

        sorter = []
        popper = []
        for file_num, file_name in enumerate(filenames):
            if 'bag' in file_name:
                popper.append(file_num)
                continue
            if 'csv' in file_name:  # read csv and create state-action trajectories
                with open(dirpath+ep_dir+'/'+file_name) as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        act = row[2]
                        act = act.replace('(', '')
                        act = act.replace(')', '')
                        act = act.split(',')
                        act = [float(i) for i in act]
                        actions.append(joy_to_agent(act))
                        states.append(row[5:9])
                popper.append(file_num)
                continue

            indexer = []
            for str_val in filenames[file_num][26:-1]:  # sort pointcloud files
                if not str_val.isdigit():
                    sorter.append(int("".join(indexer)))
                    break
                else:
                    indexer.append(str_val)

        for index in sorted(popper, reverse=True):
            del filenames[index]

        sorted_files = list.copy(filenames)

        for jj in range(len(sorter)):  # create a new sorted list of pointcloud files
            # print('insert: ', filenames[jj], 'into place: ', sorter[jj]-1)
            sorted_files[sorter[jj]-1] = filenames[jj]

        skipped_pc = 0
        episode_heatmaps = []

        for file in sorted_files:

            start_time = time.time()

            point_cloud = np.load(dirpath+ep_dir+'/'+file)['arr_0']
            if len(point_cloud) < 10000:
                skipped_pc += 1
                continue

            z = -point_cloud[:, 0]
            x = point_cloud[:, 1]
            y = -point_cloud[:, 2]

            ind = np.where((x < -1.4) | (x > 1.5) | (z > -2.2))

            x = np.delete(x, ind)
            y = np.delete(y, ind)
            z = np.delete(z, ind)

            z = (z - np.min(z))/np.ptp(z)   ## normalize height [0,1]

            if show_point_cloud:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(x, y, z, s=0.1)

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()

            pc_len = len(x)

            h_map = np.zeros([x_res, num_of_stripes])

            x_A_coeff = (x_res - 1) / 3
            x_B_coeff = (x_res - 1) / 2

            # y_A_coeff = 9.375
            # y_B_coeff = 7.5
            #
            # # y_A_coeff = (num_of_stripes - 1) / (num_of_stripes*0.1)
            # # y_B_coeff = (num_of_stripes - 1) / 2
            #
            y_A_coeff = (num_of_stripes-1)/(stripe_limits[1]-stripe_limits[0])
            y_B_coeff = -stripe_limits[0]*y_A_coeff

            for point in range(pc_len):
                if (y[point] < stripe_limits[0]) | (y[point] > stripe_limits[1]+0.1):
                    continue
                x_ind = int(x[point] * x_A_coeff + x_B_coeff)
                y_ind = int(y[point] * y_A_coeff + y_B_coeff)
                if z[point] > h_map[x_ind, y_ind]:
                    h_map[x_ind, y_ind] = z[point]

            if show_height_map:
                plt.imshow(h_map, aspect=0.1)
                plt.show(block=False)
                plt.pause(0.001)

            episode_heatmaps.append(h_map)  # create a list of heatmaps for each episode
            frame_time = time.time() - start_time
        print('episode {} appended, {} low quality point_cloud_files'.format(ep_counter, skipped_pc))
        ep_counter += 1

        recording_heatmaps.append(episode_heatmaps)  # create a list of episodes (each episode containts a list of heatmaps, resulting in a 4-dim array)
        recording_states.append(states)
        recording_actions.append(actions)

    np.save(recordings_path + 'heatmap', recording_heatmaps)
    np.save(recordings_path + 'states', recording_states)
    np.save(recordings_path + 'actions', recording_actions)



if job == 'concat_recodrings':
    act1 = np.load('/home/graphics/git/SmartLoader/saved_experts/saved_ep/ob.npy')
    obs1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/obs.npy')
    rew1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/rew.npy')
    ep_ret1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/ep_ret.npy')
    ep_str1 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/80_ep_5_hist/ep_str.npy')

    act2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/act.npy')
    obs2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/obs.npy')
    rew2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/rew.npy')
    ep_ret2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/ep_ret.npy')
    ep_str2 = np.load('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/20_ep_5_hist_diff/ep_str.npy')

    act = np.concatenate((act1, act2))
    obs = np.concatenate((obs1, obs2))
    rew = np.concatenate((rew1,rew2))
    ep_ret = np.concatenate((ep_ret1,ep_ret2))
    ep_str = np.concatenate((ep_str1,ep_str2))

    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/rew', rew)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_ret', ep_ret)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_str', ep_str)

if job == 'pos_aprox':

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/'

    heat_maps = np.load(expert_path+'heatmap.npy')
    
    num_of_labels = 40

    global labels

    inputs = []
    labels = []

    for k in range(num_of_labels):

        def onclick(event):
            global labels
            xd, yd = event.xdata, event.ydata
            print(xd, yd)
            labels.append([int(xd), int(yd)])


        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.mpl_connect('button_press_event', onclick)

        index = np.random.randint(len(heat_maps))
        h_map = heat_maps[index, :, :].reshape(100, 7)
        # ax.figure(figsize=(10, 10))

        ax.imshow(h_map)
        plt.show()

        inputs.append(h_map)
        print(labels)

    t_labels = []
    # for k in range(0, len(labels), 2): # for 2 coordinates - body and shovle locations
    #     t_labels.append([labels[k],labels[k+1]])
    for k in range(0, len(labels), 1):
        t_labels.append([labels[k]])  # for 1 coordinatge - body locatino

    np.save(expert_path+'pos_map', inputs)
    np.save(expert_path+'pos_label', t_labels)

