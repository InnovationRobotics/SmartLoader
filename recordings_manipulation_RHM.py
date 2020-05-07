import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from keras.models import Model, load_model
import re
from os import walk
import csv
import sys
import time

csv.field_size_limit(sys.maxsize)
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


jobs = ['heat_map_generator', 'concat_recodrings', 'pos_aprox', 'pos_deductor']
job = jobs[0]

if job == 'heat_map_generator':

    # dirnames = ['/home/graphics/height_test/rec07-05-H16:03:57/']
    recordings_path = '/home/graphics/recordings/push_49_RHM_no_peel/'
    # recordings_path = '/home/graphics/height_test/'

    # dirnames = ['/home/test/']
    # dirpath = '/home/test/'

    y_map_clip = [20, 100]
    # y_map_clip = [20, 140]
    x_map_front_offset = 20
    x_map_front_clip = 70

    show_height_map = False

    actions = []
    states = []
    heatmaps = []
    starts = []
    prev_body_pose = [0, 0, 0]

    ep_counter = 1
    sp_skipped_data = 0
    nd_skipped_data = 0
    hm_skipped_data = 0

    for (dirpath, dirnames, _) in walk(recordings_path):  # crate a list of episodes
        break

    for ep_dir in dirnames:  ## iterate over each
        print(ep_dir)
        for (_, _, filenames) in walk(recordings_path+ep_dir):  # create a list of pointcloud steps for each episode
            break

        # first remove bag file from list and go over csv:
        csv_index = [n for n, x in enumerate(filenames) if 'csv' in x][0]
        bag_index = [n for n, x in enumerate(filenames) if 'bag' in x][0]

        # csv_file = filenames[csv_index][7:-1]  ## if csv is open
        csv_file = filenames[csv_index]


        with open(recordings_path+ep_dir+'/'+csv_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row_num, row in enumerate(reader):
                # print('evaluations row: ')
                # read actions
                act = row[2]
                if len(act) == 0:  # if no imu data, skip line
                    nd_skipped_data += 1
                    continue
                act = act.replace('(', '').replace(')', '')
                act = act.split(',')
                act = [float(i) for i in act]

                # read states (i.e. arm heigh and pitch and body and shovle orientation)

                body_pos_str = row[3][0:86].split()
                if len(body_pos_str) == 0:  # if no imu data, skip line
                    nd_skipped_data += 1
                    continue
                body_pos = []
                for cord in body_pos_str:
                    try:
                        body_pos.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                if prev_body_pose == body_pos:
                    print('same pos, skipping row {}', format(row_num+1))
                    sp_skipped_data += 1
                    continue

                if (body_pos[0]-prev_body_pose[0]) < -0.7:
                    starts.append(True)
                else:
                    starts.append(False)

                prev_body_pose = body_pos

                shovle_pos_str = row[4][0:86].split()
                if len(shovle_pos_str) == 0:  # if no imu data, skip line
                    nd_skipped_data += 1
                    continue
                shovle_pos = []
                for cord in shovle_pos_str:
                    try:
                        shovle_pos.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                arm_height = int(row[5])
                arm_pitch = int(row[6])

                blade_orien_str = row[7][0:120].split()
                if len(blade_orien_str) == 0:  # if no imu data, skip line
                    nd_skipped_data += 1
                    continue
                blade_orien = []
                for cord in blade_orien_str:
                    try:
                        blade_orien.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                body_orien_str = row[8][0:120].split()
                if len(body_orien_str) == 0:  # if no imu data, skip line
                    nd_skipped_data += 1
                    continue
                body_orien = []
                for cord in body_orien_str:
                    try:
                        body_orien.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                heat_map_str = row[9][178:-1].split(',')
                heat_map_size = [int(row[9][64:67]),int(row[9][132:135])]
                if len(heat_map_str) == 0:  # if no data, skip line
                    nd_skipped_data += 1
                    continue
                heat_map = []
                for cord in heat_map_str:
                    try:
                        heat_map.append(float(re.sub("[^\d\.\-\'e']", "", cord)))
                    except:
                        continue

                hm_array = np.array(heat_map)
                usb_cable = np.where(hm_array > 0.5)
                for arr_cell in usb_cable[0]:
                    heat_map[arr_cell] = hm_array.min()

                ## normalize heat_map
                heat_map = (heat_map - np.min(heat_map)) / np.ptp(heat_map)

                ## clip heat_map
                heat_map = np.array(heat_map).reshape(heat_map_size)
                if x_map_front_clip:
                    # x_map_min = int(max(body_pos[0]*100+50, 0))
                    # x_map_max = int(min(body_pos[0]*100+x_map_clip, 300))
                    x_map_min = int(shovle_pos[0]*100+x_map_front_offset)
                    x_map_max = int(shovle_pos[0]*100+x_map_front_offset+x_map_front_clip)
                    if x_map_max > 260:
                        hm_skipped_data += 1
                        continue
                else:
                    x_map_min, x_map_max = 0, 260

                heat_map = heat_map[x_map_min:x_map_max, y_map_clip[0]:y_map_clip[1]]


                ## normalize states
                body_pos = [body_pos[0] / 3, (body_pos[1] - (y_map_clip[0]/100)) / (y_map_clip[1]/100 - y_map_clip[0]/100), body_pos[2]]
                shovle_pos = [shovle_pos[0] / 3, (shovle_pos[1] - (y_map_clip[0]/100)) / (y_map_clip[1]/100 - y_map_clip[0]/100), shovle_pos[2]]
                arm_height = (arm_height - 150) / 50  #full span = [150:280]
                arm_pitch = (arm_pitch-50) / 230 #full span = [50:280]



                state = [body_pos, shovle_pos, body_orien, blade_orien, arm_height, arm_pitch]


                ######################################################################
                if show_height_map:
                    anim = True
                    if anim:
                        plt.imshow(heat_map, aspect=1)
                        plt.text(40, 5, arm_height, fontsize=15)
                        plt.show(block=False)
                        # plt.show()
                        plt.pause(0.01)
                        try:
                            del gca().texts[-1]
                        except:
                            pass
                        # draw()
                    else:
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(heat_map)
                        print(arm_height)

                        # est_pos = [shovle_pos[0]*260, shovle_pos[1]*(y_map_clip[1]-y_map_clip[0])]

                        # plt.scatter(est_pos[1], est_pos[0], s=300, c='red', marker='o')

                        plt.show()

                ######################################################################

                heatmaps.append(heat_map)  # create a list of clipped heatmaps for each episode
                actions.append(joy_to_agent(act))
                states.append(state)
                # print('appended {} steps'.format(row_num))


            print('episode {} appended, {} skipped frames due to missing data, {} due to duplicate data and {} due to heat map limits'.format(
                ep_counter, nd_skipped_data, sp_skipped_data, hm_skipped_data))
            skipped_data = 0
            ep_counter += 1

    np.save(dirpath + 'heatmaps', heatmaps)
    np.save(dirpath + 'states', states)
    np.save(dirpath + 'starts', starts)
    np.save(dirpath + 'actions', actions)

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
    rew = np.concatenate((rew1, rew2))
    ep_ret = np.concatenate((ep_ret1, ep_ret2))
    ep_str = np.concatenate((ep_str1, ep_str2))

    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/rew', rew)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_ret', ep_ret)
    np.save('/home/graphics/git/SmartLoader/saved_experts/Push/1_rock/100_ep_full/ep_str', ep_str)

if job == 'pos_aprox':

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/'

    heat_maps = np.load(expert_path + 'heatmap.npy')

    num_of_labels = 50

    global pos

    inputs = []
    labels = []
    pos = []

    for k in range(num_of_labels):
        def onclick(event):
            global pos
            xd, yd = event.xdata, event.ydata
            print(xd, yd)
            pos.append([int(xd), int(yd)])


        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.mpl_connect('button_press_event', onclick)

        index = np.random.randint(len(heat_maps))
        h_map = heat_maps[index, :, :].reshape(100, 16)
        # ax.figure(figsize=(10, 10))

        ax.imshow(h_map)
        plt.show()

        inputs.append(h_map)
        print(pos)


        # for k in range(0, len(labels), 2): # for 2 coordinates - body and shovle locations
        #     t_labels.append([labels[k],labels[k+1]])

        labels.append(pos[k])  # for 1 coordinatge - body location

    np.save(expert_path + 'pos_approx_map', inputs)
    np.save(expert_path + 'pos_approx_label', labels)

if job == 'pos_deductor':

    expert_path = '/home/graphics/git/SmartLoader/saved_experts/HeatMap/real_life/Push_49_ep/full_map/'
    heat_maps = np.load(expert_path + 'heatmap.npy')
    states = np.load(expert_path + 'states.npy', allow_pickle=True)
    starts = np.load(expert_path + 'starts.npy')
    model = load_model(expert_path+'KERAS_pos_est_model_08_val_loss')

    positions = []
    for step, h_map in enumerate(heat_maps):
        positions.append(model.predict(h_map.reshape(1, 1, 100, 16)))

    np.save(expert_path + 'positions', positions)





