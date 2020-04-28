from keras.models import load_model
from SmartLoaderIRL import SmartLoader
import numpy as np

if __name__ == '__main__':

    env = SmartLoader()
    jobs = ['BC', 'PD', 'dump']
    job = jobs[2]
    obs = env.reset(job)

    if job == 'BC':
    # Behaviour Cloning TEST:
        action = np.array([])
        model = load_model('/home/sload/Downloads/SmartLoader-master/saved_models/Heatmap/lift_23_ep_005_loss')
        for _ in range(10000):
            obs = env.step(action)
            hmap = obs[0]
            # action = [0, 0, 0, 0]
            # action = model.predict(hmap.reshape(1,1,100,6))[0]

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

        env.LLC.save_plot(name='step response 2')

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



