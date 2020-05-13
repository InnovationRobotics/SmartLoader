#!/usr/bin/python

import rospy
from std_msgs.msg import Int32, Bool
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
import ros_numpy
from grid_map_msgs.msg import GridMap
from keras.models import load_model

import os
import time
import numpy as np
import math

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


class SmartLoader:

    # CALLBACKS
    def VehiclePositionCB(self,stamped_pose):
        x = stamped_pose.pose.pose.position.x
        y = stamped_pose.pose.pose.position.y
        z = stamped_pose.pose.pose.position.z
        self.world_state['VehiclePos'] = np.array([x,y,z])

        qx = stamped_pose.pose.pose.orientation.x
        qy = stamped_pose.pose.pose.orientation.y
        qz = stamped_pose.pose.pose.orientation.z
        qw = stamped_pose.pose.pose.orientation.w
        self.world_state['VehicleOrien'] = np.array([qx,qy,qz,qw])

    def ShovelPositionCB(self,stamped_pose):
        x = stamped_pose.pose.pose.position.x
        y = stamped_pose.pose.pose.position.y
        z = stamped_pose.pose.pose.position.z
        self.world_state['ShovelPos'] = np.array([x,y,z])

        qx = stamped_pose.pose.pose.orientation.x
        qy = stamped_pose.pose.pose.orientation.y
        qz = stamped_pose.pose.pose.orientation.z
        qw = stamped_pose.pose.pose.orientation.w
        self.world_state['ShovelOrien'] = np.array([qx,qy,qz,qw])

    def ArmHeightCB(self, data):
        height = data.data
        cur_height = np.array([height])
        # self.world_state['ArmHeight'] = np.array([height])
        self.blade_lift_hist.append(cur_height)
        if len(self.blade_lift_hist) > 5:
            self.blade_lift_hist.pop(0)
        self.world_state['ArmHeight'] = np.mean(self.blade_lift_hist)


    def ArmShortHeightCB(self, data):
        height = data.data
        cur_pitch = np.array([height])
        # self.world_state['BladePitch'] = np.array([height])
        self.blade_pitch_hist.append(cur_pitch)
        if len(self.blade_pitch_hist) > 5:
            self.blade_pitch_hist.pop(0)
        self.world_state['BladePitch'] = np.mean(self.blade_pitch_hist)


    def BladeImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['BladeOrien'] = np.array([qx,qy,qz,qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['BladeAngularVel'] = np.array([wx,wy,wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['BladeLinearAcc'] = np.array([ax,ay,az])

    def VehicleImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['VehicleOrienIMU'] = np.array([qx,qy,qz,qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['VehicleAngularVelIMU'] = np.array([wx,wy,wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['VehicleLinearAccIMU'] = np.array([ax,ay,az])

    def PointCloudCB(self, data):
        ### based on velodyne FOV [330:25]
        np_pc = ros_numpy.numpify(data)
        xyz_array = ros_numpy.point_cloud2.get_xyz_points(np_pc)
        # self.heat_map = HeatMap(xyz_array)[0]
        # self.heatmap_arr.append(self.heat_map)
        # if len(self.heatmap_arr) > 3:
        #     # self.upd_heat_map = self.heatmap_arr[-1]
        #     self.upd_heat_map = np.mean(self.heatmap_arr, axis=0)
        #     self.heatmap_arr=[]

    def GridMapCB(self, data):
        map_size = [int(data.info.length_y/data.info.resolution),int(data.info.length_x/data.info.resolution)]

        hm_array = np.array(data.data[1].data)
        clean_hm = np.copy(hm_array)

        usb_cable = np.where(hm_array > 0.5)
        for arr_cell in usb_cable[0]:
            # clean_hm[arr_cell] = hm_array.min()
            # clean_hm[arr_cell] = np.mean(hm_array)
            clean_hm[arr_cell] = hm_array[max(usb_cable[0])+2]

        self.heat_map = np.array(clean_hm).reshape(map_size)[:,:]
        # plt.imshow(self.heat_map)
        # plt.show(block=False)
        # plt.pause(0.01)
        # plt.close

    def do_action(self, agent_action):
        joymessage = Joy()

        joyactions = self.AgentToJoyAction(agent_action)  # clip actions to fit action_size

        joymessage.axes = [joyactions[0], 0., joyactions[2], joyactions[3], joyactions[4], joyactions[5], 0., 0.]

        joymessage.buttons = 11*[0]
        joymessage.buttons[7] = 1 ## activation of hydraulic pump

        self.joypub.publish(joymessage)
        rospy.logdebug(joymessage)

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[2] = joyactions[5] = 1

        joyactions[0] = agent_action[0] # vehicle turn
        joyactions[3] = agent_action[2] # blade pitch
        joyactions[4] = agent_action[3] # arm up/down

        if agent_action[1] < 0: # drive backwards
            joyactions[2] = 2 * agent_action[1] + 1
            # joyactions[2] = -2*agent_action[1] - 1

        elif agent_action[1] > 0: # drive forwards
            joyactions[5] = -2*agent_action[1] + 1

        return joyactions

    def lift_est(self):
        x_blade = self.world_state['ShovelPos'].item(0)
        y_blade = self.world_state['ShovelPos'].item(1)
        z_blade = self.world_state['ShovelPos'].item(2)
        pitch = self.world_state['BladePitch'].item(0)

        x_blade = x_blade / 2.6
        y_blade = y_blade / 1.6
        z_blade = (z_blade - 0.097) / (0.323 - 0.097)
        pitch = (pitch - 50) / 230

        data = np.array([x_blade, y_blade, z_blade, pitch])

        lift = self.lift_est_model.predict(data.reshape(1,4))
        lift = lift*100 + 150
        return lift.item(0)


    def __init__(self):

        self._output_folder = os.getcwd()

        self.world_state = {}
        self.arm_lift = []
        self.arm_pitch = []
        self.heat_map = []
        self.heatmap_arr = []
        self.upd_heat_map = []

        self.blade_pitch_hist = []
        self.blade_lift_hist = []

        self.obs = {}
        # self.keys = ['ArmHeight', 'BladePitch', 'VehiclePos', 'ShovelPos']
        self.keys = ['BladePitch', 'VehiclePos', 'ShovelPos']

        # For time step
        self.current_time = time.time()
        self.last_time = self.current_time
        self.time_step = []
        self.last_obs = np.array([])
        self.TIME_STEP = 0.01

        self.lift_est_model = load_model('/home/sload/Downloads/lift_est_model_corrected')

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('sl_pose', PoseWithCovarianceStamped, self.VehiclePositionCB)
        self.shovelPositionSub = rospy.Subscriber('shovel_pose', PoseWithCovarianceStamped, self.ShovelPositionCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.shortHeightSub = rospy.Subscriber('arm/shortHeight', Int32, self.ArmShortHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.PointCloudSub = rospy.Subscriber('/velodyne_points', PointCloud2, self.PointCloudCB)
        self.vehicleImu = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)
        self.mapSub = rospy.Subscriber('/sl_map', GridMap, self.GridMapCB)

        # Define Publisher
        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)

    def get_obs(self):

        # wait for topics to update
        while True:  # wait for all topics to arrive
            if all(key in self.world_state for key in self.keys) and (len(self.heat_map) > 0):
                break

        # current state
        h_map = self.heat_map
        # arm_lift = self.world_state['ArmHeight'].item(0)
        arm_pitch = self.world_state['BladePitch'].item(0)
        x_vehicle = self.world_state['VehiclePos'].item(0)
        y_vehicle = self.world_state['VehiclePos'].item(1)
        z_vehicle = self.world_state['VehiclePos'].item(2)
        # vehicle_orien = quatToEuler(np.around (self.world_state['VehicleOrienIMU'],2))[2]
        x_blade = self.world_state['ShovelPos'].item(0)
        y_blade = self.world_state['ShovelPos'].item(1)
        z_blade = self.world_state['ShovelPos'].item(2)
        # blade_orien = self.world_state['BladeOrien']

        arm_lift = self.lift_est()

        obs = {'h_map':h_map, 'x_vehicle':x_vehicle, 'y_vehicle':y_vehicle, 'z_vehicle': z_vehicle,
               'x_blade':x_blade, 'y_blade':y_blade, 'z_blade': z_blade, 'lift': arm_lift, 'pitch': arm_pitch}

        return obs

    def step(self, action):

        # wait for topics to update
        start_time = time.time()
        # while True: # wait for all topics to arrive
        #     if all(key in self.world_state for key in self.keys) and self.heat_map:
        #         break
        # print(time.time() - start_time)

        # current state
        h_map = self.heat_map
        # arm_lift = self.world_state['ArmHeight'].item(0)
        arm_pitch = self.world_state['BladePitch'].item(0)
        x_vehicle = self.world_state['VehiclePos'].item(0)
        y_vehicle = self.world_state['VehiclePos'].item(1)
        z_vehicle = self.world_state['VehiclePos'].item(2)
        # vehicle_orien = quatToEuler(self.world_state['VehicleOrienIMU'])[2]
        x_blade = self.world_state['ShovelPos'].item(0)
        y_blade = self.world_state['ShovelPos'].item(1)
        z_blade = self.world_state['ShovelPos'].item(2)
        # blade_orien = self.world_state['BladeOrien']

        arm_lift = self.lift_est()

        obs = {'h_map':h_map, 'x_vehicle':x_vehicle, 'y_vehicle':y_vehicle, 'z_vehicle': z_vehicle,
               'x_blade':x_blade, 'y_blade':y_blade, 'z_blade': z_blade, 'lift': arm_lift, 'pitch': arm_pitch}

        step_time = time.time() - start_time
        if step_time < self.TIME_STEP:
            time.sleep(self.TIME_STEP - step_time)

        self.do_action(action)

        return obs
#
# if __name__ == '__main__':
#
#     env = SmartLoader()
#     for i in range(500):
#         joy = [] # option to control by joystick
#         obs = env.step(joy, i)
#         h_map = obs[0]
#
#     env.LLC.save_plot(name='step response 2')

        # plt.matshow(h_map)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # print(ob[0])