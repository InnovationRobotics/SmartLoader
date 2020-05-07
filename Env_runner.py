from keras.models import load_model
from SmartLoaderIRL import SmartLoader
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from LLC import LLC_pid
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def quatToEuler(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

if __name__ == '__main__':

    env = SmartLoader()
    jobs = ['BC', 'PD', 'dump']
    job = jobs[0]

    # Behaviour Cloning TEST:
    if job == 'BC':

        # model = load_model('/home/sload/Downloads/SsmartLoader-master (1)/SmartLoader-master/saved_models/Heatmap/push_49_ep_012_loss')
        # model = load_model('/home/sload/Downloads/talos_001_model_keras_v2')
        # model = load_model('/home/sload/git/SmartLoader/saved_experts/pos_est_model')

        # conf_model = load_model('/home/sload/Downloads/KERAS_conf_model_1.7_loss')
        # while True:
        #     env.reset()

        # obs = {h_map, x_blade, y_blade, lift, pitch, x_vehicle, y_vehicle, vehicle_orien}
        obs = env.reset()
        # hmap = obs['h_map']

        des = [obs['x_blade']+1, obs['y_blade']+0.1]

        pid = LLC_pid.LLC()

        x, y, x_des, y_des = [], [], [], []
        counter = 0

        # while True:
        #     obs = env.reset()
            # euler = quatToEuler(obs['vehicle_orien'])
            # print(euler)

        for i in range(200):
            # from model predict desired configuration
            # des = [x_blade, y_blade, lift, pitch]
            # des = list(conf_model.predict(hmap.reshape(1, 1, 100, 16))[0])
            # if des from conf_model switch x,y places
            # des_x, des_y = des[1], des[0]
            # des[0], des[1] = des_x, des_y

            # test - drive forwards
            # if i % 20 == 0 and bool(i) and i < 100:
            #     xdes += 0.05
            # des = [xdes, obs['y_blade']]

            action = pid.step(obs, des)
            obs = env.step(action)
            print(action)
            # hmap = obs['h_map']

            error = np.linalg.norm([des[0] - obs['x_blade'], des[1] - obs['y_blade']])
            print('error =', error)
            # print('x vehicle = ', obs['x_vehicle'], 'y vehicle = ', obs['y_vehicle'])
            if abs(error) < 0.1:
                counter += 1
                if counter > 20:
                    break
                else:
                    counter = 0

            # save locations
            x.append(obs['x_blade'])
            y.append(obs['y_blade'])
            x_des.append(des[0])
            y_des.append(des[1])

            # plt.imshow(hmap, aspect=0.1)
            # # plt.scatter(pos[0],pos[1],s=100,c='red',marker='o')
            # plt.show(block=False)
            # plt.pause(0.01)

            # k_p=3

            # action[1] = k_p*(des_conf[1]-pos[1])
            # print('pos : ', pos[1], ' des: ',  des[1])


            # action[1]=des_conf
            # print(des_conf[0:4])
            # action = [0, 0, 0, 0]
            # action = []
            # xy_location = model.predict(hmap.reshape(1,1,100,16))[0]
            # print(xy_location)
            # print(pos)
            # action[0]=0
            # action[0]=0

        # stop moving
        action = [0, 0, 0, 0]
        obs = env.step(action)
        # plots
        # pid.lift_pid.save_plot('lift_test', 'lift')
        # pid.pitch_pid.save_plot('pitch_test', 'pitch')
        pid.steer_pid.save_plot('steer_test', 'steer')
        pid.speed_pid.save_plot('speed_test', 'speed')

        length = len(x)
        t = np.linspace(0, length, length)
        fig, (ax_x, ax_y) = plt.subplots(2)
        ax_x.set_title('x pos')
        ax_y.set_title('y pos')

        ax_x.plot(t, x_des, color='red')
        ax_y.plot(t, y_des, color='red')
        ax_x.plot(t, x, color='blue')
        ax_y.plot(t, y, color='blue')

        # save
        fig.savefig('/home/sload/git/SmartLoader/LLC/plots/locations')
        print('figure saved!')


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



