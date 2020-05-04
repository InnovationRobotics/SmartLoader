from keras.models import load_model
from SmartLoaderIRL import SmartLoader
import numpy as np
from matplotlib import pyplot as plt
from LLC import LLC_pid

if __name__ == '__main__':

    env = SmartLoader()
    jobs = ['BC', 'PD', 'dump']
    job = jobs[0]

    # Behaviour Cloning TEST:
    if job == 'BC':

        # model = load_model('/home/sload/Downloads/SsmartLoader-master (1)/SmartLoader-master/saved_models/Heatmap/push_49_ep_012_loss')
        # model = load_model('/home/sload/Downloads/talos_001_model_keras_v2')
        # model = load_model('/home/sload/git/SmartLoader/saved_experts/pos_est_model')

        pos_model = load_model('/home/sload/Downloads/KERAS_pos_est_model_0.8_val_loss')
        conf_model = load_model('/home/sload/Downloads/KERAS_conf_model_1.7_loss')

        # for _ in range(10000):
        obs = env.reset() # [h_map, lift, pitch]
        hmap = obs[0]

        pid = LLC_pid.LLC

        # from heatmap predict xy location
        pos = pos_model.predict(hmap.reshape(1, 1, 100, 16))[0]
        obs = obs.append(pos) # [h_map, lift, pitch, x, y]

        # from model predict desired configuration
        # des =
        des = conf_model.predict(hmap.reshape(1, 1, 100, 16))[0]

        while True:
            pid.step(obs, des)
            obs = env.step(action)
            hmap = obs[0]

            pos = pos_model.predict(hmap.reshape(1, 1, 100, 16))[0]
            plt.imshow(hmap, aspect=0.1)
            # plt.scatter(pos[0],pos[1],s=100,c='red',marker='o')
            plt.show(block=False)
            plt.pause(0.01)

            des_conf = conf_model.predict(hmap.reshape(1, 1, 100, 16))[0]

            k_p=3

            # action[1] = k_p*(des_conf[1]-pos[1])
            print('pos : ', pos[1], ' des: ',  des_conf[1])


            # action[1]=des_conf
            # print(des_conf[0:4])
            # action = [0, 0, 0, 0]
            # action = []
            # xy_location = model.predict(hmap.reshape(1,1,100,16))[0]
            # print(xy_location)
            # print(pos)
            # action[0]=0
            # action[0]=0

    # model = load_model()

    # plt.matshow(h_map)
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()
    # print(ob[0])

    # elif job == 'PD':
    # # PID TEST:
    #     for i in range(500):
    #         action = env.LLC.step(obs, i)
    #         obs = env.step(action)
    #
    #     env.LLC.save_plot(name='step response')
    #
    # elif job == 'dump':
    # # dump test
    #     stop = False
    #     i = 0
    #     while not stop:
    #         action = env.LLC.step(obs, i)
    #         obs = env.step(action)
    #         if obs[2] >= env.LLC.pitch_pid.SetPoint:
    #             stop = True
    #         i += 1
    #
    #     env.LLC.save_plot(name='dump test')



