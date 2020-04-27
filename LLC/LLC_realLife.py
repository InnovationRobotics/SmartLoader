#!/usr/bin/python

# Implement PD on real vehicle
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32, Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
# from grid_map_msgs.msg import GridMap

import os
import time
import numpy as np
import math
from LLC import pid
from matplotlib import pyplot as plt

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

class LLCEnv:

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

    # def GridMapCB(self):

    def do_action(self, pd_action):
        joymessage = Joy()

        joyactions = self.PDToJoyAction(pd_action)

        joymessage.axes = [joyactions[0], 0., joyactions[2], joyactions[3], joyactions[4], joyactions[5], 0., 0.]

        joymessage.buttons = 11*[0]
        joymessage.buttons[7] = 1

        self.joypub.publish(joymessage)
        rospy.logdebug(joymessage)

    def PDToJoyAction(self, pd_action):
        # translate chosen action (array) to joystick action (dict)
        joyactions = np.zeros(6)

        # ONLY LIFT AND PITCH
        joyactions[3] = pd_action[0] # blade pitch
        joyactions[4] = pd_action[1] # blade lift

        return joyactions


    def __init__(self):
        self._output_folder = os.getcwd()

        self.world_state = {}
        self.keys = ['ArmHeight', 'BladePitch']
        self.lift = []
        self.pitch = []

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
        # self.mapSub = rospy.Subscriber('Sl_map', GridMap, self.GridMapCB)
        # self.vehicleImu = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

        # Define Publisher
        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)

        # wait for set up
        # time.sleep(1)
        while True: # wait for all topics to arrive
            if all(key in self.world_state for key in self.keys):
                break

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self._kp_kd = 'realLife_lift_P=0.002_I=0.0001_D=0.00001_pitch_P=-0.008_D=0.001_st=0.05'
        self.lift_pid = pid.PID(P=0.002, I=0.0001, D=0.00001, saturation=True)
        self.lift_pid.SetPoint = 200.
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = pid.PID(P=-0.008, I=0, D=0.001, saturation=True)
        self.pitch_pid.SetPoint = 150.
        self.pitch_pid.setSampleTime(self.TIME_STEP)


    def step(self, i, stop):

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
        current_lift = self.world_state['ArmHeight'].item(0)
        current_pitch = self.world_state['BladePitch'].item(0)
        print('{}. lift = '.format(str(i)), current_lift, 'pitch = ', current_pitch)

        # check if done
        if (current_lift == self.lift_pid.SetPoint and current_pitch == self.pitch_pid.SetPoint):
            print('Success!')
            stop = True

        # pid update
        lift_output = self.lift_pid.update(current_lift)
        pitch_output = self.pitch_pid.update(current_pitch)

        # do action
        ## both actions together
        pd_action = np.array([pitch_output, lift_output])
        self.do_action(pd_action)
        ## alternating actions
        # self.do_action([lift_output, 0])
        # self.do_action([0, pitch_output])
        print('lift action = ', lift_output, 'pitch action = ', pitch_output)

        # save data
        self.lift.append(current_lift)
        self.pitch.append(current_pitch)

        return stop

    def save_plot(self):
        # init plot
        length = len(self.lift)
        x = np.linspace(0, length, length)
        fig, (ax_lift, ax_pitch) = plt.subplots(2)
        ax_lift.set_title('lift')
        ax_pitch.set_title('pitch')

        # plot set points
        ax_lift.plot(x, np.array(x.size * [self.lift_pid.SetPoint]))
        ax_pitch.plot(x, np.array(x.size *[self.pitch_pid.SetPoint]))

        # plot data
        ax_lift.scatter(x, self.lift, color='red')
        ax_pitch.scatter(x, self.pitch, color='red')

        # create plot folder if it does not exist
        try:
            plot_folder = "{}/plots".format(self._output_folder)
        except FileNotFoundError:
            os.makedirs(plot_folder)
        fig.savefig('{}/{}.png'.format(plot_folder, self._kp_kd))
        print('figure saved!')


if __name__ == '__main__':
    L = 500
    stop = False
    LLC = LLCEnv()
    for i in range(L):
        stop = LLC.step(i, stop)
        if i == L-1:
            LLC.save_plot()
        # if stop:
        #     LLC.save_plot()
        #     break
