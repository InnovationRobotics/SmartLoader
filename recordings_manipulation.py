import numpy as np
import matplotlib.pyplot as plt
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


jobs = ['clip_point_cloud_values', 'concat_recodrings']
job = jobs[0]

# for ep_dir in range(len(dirnames)):
#     for (_, _, filenames) in walk(dirpath + dirnames[ep_dir]):
#         for file in filenames:
#             if 'bag' in file:
#                 break

if job == 'clip_point_cloud_values':

    recordings_path = '/home/graphics/Desktop/HeatMap/lift/'
    show_cropped_point_cloud = False
    show_selected_stripes = False
    show_height_map = True

    episode_heatmaps = []
    actions = []
    states = []
    ep_counter = 1

    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break
    for ep_dir in dirnames:   ## iterate over each
        for (_, _, filenames) in walk(dirpath+ep_dir):  # create a list of pointcloud steps for each episode
            break

        sorter = []
        for file_num, file_name in enumerate(filenames):
            if 'bag' in file_name:
                filenames.pop(file_num)
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
                filenames.pop(file_num)

            indexer = []
            for str_val in filenames[file_num][26:-1]:  # sort pointcloud files
                if not str_val.isdigit():
                    sorter.append(int("".join(indexer)))
                    break
                else:
                    # indexer.append(int(str_val))
                    indexer.append(str_val)
        sorted_files = list.copy(filenames)

        for jj in range(len(sorter)):  # create a new sorted list of pointcloud files
            # print('insert: ', filenames[jj], 'into place: ', sorter[jj]-1)
            sorted_files[sorter[jj]-1] = filenames[jj]

        stripe_range = np.arange(-0.5,0.2,0.1)
        num_of_stripes = len(stripe_range)-1
        x_resolution = 100
        last_max_st = []
        last_max_st.append(np.zeros([x_resolution, num_of_stripes]))
        last_max_st.append(np.zeros([x_resolution, num_of_stripes]))
        last_max_st.append(np.zeros([x_resolution, num_of_stripes]))
        last_max_st.append(np.zeros([x_resolution, num_of_stripes]))
        last_max_st.append(np.zeros([x_resolution, num_of_stripes]))
        step_count = 0

        skipped_pc = 0
        for file in sorted_files:

            start_time = time.time()

            point_cloud = np.load(dirpath+ep_dir+'/'+file)['arr_0']
            if len(point_cloud) < 10000:
                skipped_pc += 1
                continue

            z = -point_cloud[:, 0]
            x = point_cloud[:, 1]
            y = -point_cloud[:, 2]

            ind = np.where((x < -1.4) | (x > 1.5) | (z > -2))

            x = np.delete(x, ind)
            y = np.delete(y, ind)
            z = np.delete(z, ind)

            # [x, y, z] = [y, -z, -x]
            z = (z - np.min(z))/np.ptp(z)   ## normalize height [0,1]

            if show_cropped_point_cloud:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(x, y, z, s=0.1)

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()


            st_ind = [] #### seperate to stripes
            for stripe in range(num_of_stripes):
                st_ind.append(np.where(np.logical_and(y > stripe_range[stripe], y < stripe_range[stripe+1])))


            # shortest = np.min( [len(st_ind[0][0]), len(st_ind[1][0]), len(st_ind[2][0])] )
            shortest = np.inf
            for stripe in range(num_of_stripes):
                stripe_len = len(st_ind[stripe][0])
                if stripe_len < shortest:
                    shortest = stripe_len


            st = np.zeros([num_of_stripes, 3, shortest])
            # for i in range(np.min( [len(st1_ind), len(st2_ind), len(st3_ind)] )):
            for i in range(num_of_stripes):    ## populate an array of xyz coordinates for each stripe
                st[i, 0, :] = x[st_ind[i][0][0:shortest]]
                st[i, 1, :] = y[st_ind[i][0][0:shortest]]
                st[i, 2, :] = z[st_ind[i][0][0:shortest]]

            for pp in range(num_of_stripes):
                st_test_ind = np.argsort(st[pp,0])
                st[pp][0][:] = st[pp][0][st_test_ind]
                st[pp][1][:] = st[pp][1][st_test_ind]
                st[pp][2][:] = st[pp][2][st_test_ind]

            if show_selected_stripes:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(st[0:num_of_stripes,0,:], st[0:num_of_stripes,1,:], st[0:num_of_stripes,2,:], s=0.1)

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()

            max_st = np.zeros([x_resolution,num_of_stripes])
            jump = int(shortest/x_resolution)
            for jj in range(num_of_stripes):   ## iterate over each stripe
                for k in range(x_resolution):   ## crate a 100*3 heatmap
                    max_val = np.max( st[jj, 2, k*jump:((k+1)*jump)] )
                    max_st[k, jj] = max_val

            if show_height_map:
                if step_count < 5:
                    diff_st = max_st - last_max_st[0]
                else:
                    diff_st = max_st - last_max_st[step_count-5]
                plt.matshow(max_st)
                # plt.colorbar()
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                step_count+=1
                last_max_st.append(max_st)

            episode_heatmaps.append(max_st)
            frame_time = time.time() - start_time
        print('episode {} appended, {} low quality point_cloud_files'.format(ep_counter, skipped_pc))
        ep_counter += 1

    np.save(recordings_path + 'heatmap', episode_heatmaps)
    np.save(recordings_path + 'states', states)
    np.save(recordings_path + 'actions', actions)



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
