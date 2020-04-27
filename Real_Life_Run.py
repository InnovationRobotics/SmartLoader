#!/usr/bin/python

import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32, Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import PointCloud2
import ros_numpy
# from grid_map_msgs.msg import GridMap
from HeatMapGen import HeatMap
from matplotlib import pyplot as plt
from LLC import LLC_pid

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
        x = stamped_pose.pose.position.x
        y = stamped_pose.pose.position.y
        z = stamped_pose.pose.position.z
        self.world_state['VehiclePos'] = np.array([x,y,z])

        qx = stamped_pose.pose.orientation.x
        qy = stamped_pose.pose.orientation.y
        qz = stamped_pose.pose.orientation.z
        qw = stamped_pose.pose.orientation.w
        self.world_state['VehicleOrien'] = np.array([qx,qy,qz,qw])

    def ShovelPositionCB(self,stamped_pose):
        x = stamped_pose.pose.position.x
        y = stamped_pose.pose.position.y
        z = stamped_pose.pose.position.z
        self.world_state['ShovelPos'] = np.array([x,y,z])

        qx = stamped_pose.pose.orientation.x
        qy = stamped_pose.pose.orientation.y
        qz = stamped_pose.pose.orientation.z
        qw = stamped_pose.pose.orientation.w
        self.world_state['ShovelOrien'] = np.array([qx,qy,qz,qw])

    def ArmHeightCB(self, data):
        height = data.data
        self.world_state['ArmHeight'] = np.array([height])

    def ArmShortHeightCB(self, data):
        height = data.data
        self.world_state['BladePitch'] = np.array([height])

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
        self.heat_map = HeatMap(xyz_array)

    def do_action(self, agent_action):

        joymessage = Joy()

        joyactions = self.AgentToJoyAction(agent_action)  # clip actions to fit action_size

        joymessage.axes = [joyactions[0], 0., joyactions[2], joyactions[3], joyactions[4], joyactions[5], 0., 0.]

        joymessage.buttons = 11*[0]
        joymessage.buttons[7] = 1

        self.joypub.publish(joymessage)
        rospy.logdebug(joymessage)

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[0] = agent_action[0] # vehicle turn
        joyactions[3] = agent_action[2] # blade pitch
        joyactions[4] = agent_action[3] # arm up/down

        if agent_action[1] < 0: # drive backwards
            joyactions[2] = -2*agent_action[1] - 1

        elif agent_action[1] > 0: # drive forwards
            joyactions[5] = -2*agent_action[1] + 1

        return joyactions

    def __init__(self):

        self._output_folder = os.getcwd()

        self.world_state = {}
        self.arm_lift = []
        self.arm_pitch = []
        self.heat_map = []

        self.LLC = LLC_pid.LLC()

        # For time step
        self.current_time = time.time()
        self.last_time = self.current_time
        self.time_step = []
        self.last_obs = np.array([])
        self.TIME_STEP = 0.05

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('Sl_pose', PoseStamped, self.VehiclePositionCB)
        self.shovelPositionSub = rospy.Subscriber('shovel_pose', PoseStamped, self.ShovelPositionCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.shortHeightSub = rospy.Subscriber('arm/shortHeight', Int32, self.ArmShortHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.PointCloudSub = rospy.Subscriber('/velodyne_points', PointCloud2, self.PointCloudCB)
        # self.mapSub = rospy.Subscriber('Sl_map', GridMap, self.GridMapCB)
        # self.vehicleImu = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

        # Define Publisher
        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)

        time.sleep(1)


    def step(self):

        # for even time steps
        self.current_time = time.time()
        time_step = self.current_time - self.last_time

        if time_step < self.TIME_STEP:
            time.sleep(self.TIME_STEP - time_step)
            self.current_time = time.time()
            time_step = self.current_time - self.last_time

        self.time_step.append(time_step)
        self.last_time = self.current_time

        # current state
        h_map = self.heat_map
        arm_lift = self.world_state['ArmHeight'].item(0)
        arm_pitch = self.world_state['BladePitch'].item(0)
        print('height = ', arm_lift, 'pitch = ', arm_pitch)

        obs = [h_map, arm_lift, arm_pitch]

        pd_action = self.LLC.step(obs)
        action = np.concatenate(([0., 0.], pd_action))

        # do action
        if action:
            self.do_action(action)

        return obs

if __name__ == '__main__':

    env = SmartLoader()
    for i in range(100000):
        action = []
        obs = env.step()
        h_map = obs[0]

        # plt.matshow(h_map)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # print(ob[0])