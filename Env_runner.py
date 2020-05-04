from keras.models import load_model
from SmartLoaderIRL import SmartLoader
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    env = SmartLoader()
    jobs = ['BC', 'PD', 'dump']
    job = jobs[0]
    obs = env.reset(job)

    if job == 'BC':
    # Behaviour Cloning TEST:
        action = np.array([])
        # model = load_model('/home/sload/Downloads/SmartLoader-master (1)/SmartLoader-master/saved_models/Heatmap/push_49_ep_012_loss')
        model = load_model('/home/sload/Downloads/talos_001_model_keras_v2')

        # for _ in range(10000):
        while True:
            obs = env.step(action)
            hmap = obs[0]

            # plt.imshow(hmap)
            # # plt.scatter(pos[0],pos[1],s=100,c='red',marker='o')
            # plt.show(block=False)
            # plt.pause(0.01)
            # action = [0, 0, 0, 0]
            action = model.predict(hmap.reshape(1,1,100,6))[0]
            # print(pos)
            # action[0]=0
            # action[0]=0

    # model = load_model()

    # plt.matshow(h_map)
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()
    # print(ob[0])

    elif job == 'PD':
    # PID TEST:
        for i in range(500):
            action = env.LLC.step(obs, i)
            obs = env.step(action)

        env.LLC.save_plot(name='step response')

    elif job == 'dump':
    # dump test
        stop = False
        i = 0
        while not stop:
            action = env.LLC.step(obs, i)
            obs = env.step(action)
            if obs[2] >= env.LLC.pitch_pid.SetPoint:
                stop = True
            i += 1

        env.LLC.save_plot(name='dump test')



