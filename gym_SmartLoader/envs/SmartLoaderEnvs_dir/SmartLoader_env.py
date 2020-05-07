# !/usr/bin/env python3
# building custom gym environment:
# # https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5

import sys
import time
from src.EpisodeManager import *
import src.Unity2RealWorld as urw
import gym
from gym import spaces
import numpy as np
import math
from math import pi as pi
from scipy.spatial.transform import Rotation as R
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32, Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from grid_map_msgs.msg import GridMap


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


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


class BaseEnv(gym.Env):

    def VehiclePositionCB(self, stamped_pose):
        # new_stamped_pose = urw.positionROS2RW(stamped_pose)
        x = stamped_pose.pose.position.x
        y = stamped_pose.pose.position.y
        z = stamped_pose.pose.position.z
        self.world_state['VehiclePos'] = np.array([x, y, z])

        qx = stamped_pose.pose.orientation.x
        qy = stamped_pose.pose.orientation.y
        qz = stamped_pose.pose.orientation.z
        qw = stamped_pose.pose.orientation.w
        orientation = np.array([qx, qy, qz, qw])
        # self.world_state['VehicleOrien'] = np.array([qx, qy, qz, qw])
        yaw, pitch, roll = quaternion_to_euler(qx, qy, qz, qw)
        self.world_state['euler_ypr'] = np.array([yaw, pitch, roll])
        # rospy.loginfo('position is:' + str(stamped_pose.pose))

    def VehicleVelocityCB(self, stamped_twist):
        vx = stamped_twist.twist.linear.x
        vy = stamped_twist.twist.linear.y
        vz = stamped_twist.twist.linear.z
        self.world_state['VehicleLinearVel'] = np.array([vx, vy, vz])

        wx = stamped_twist.twist.angular.x
        wy = stamped_twist.twist.angular.y
        wz = stamped_twist.twist.angular.z
        self.world_state['VehicleAngularVel'] = np.array([wx, wy, wz])

        # rospy.loginfo('velocity is:' + str(stamped_twist.twist))

    def ArmHeightCB(self, data):
        height = data.data
        self.world_state['ArmHeight'] = np.array([height])

        # rospy.loginfo('arm height is:' + str(height))

    def BladeImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['BladeOrien'] = np.array([qx, qy, qz, qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['BladeAngularVel'] = np.array([wx, wy, wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['BladeLinearAcc'] = np.array([ax, ay, az])

        # rospy.loginfo('blade imu is:' + str(imu))

    def VehicleImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['VehicleOrienIMU'] = np.array([qx, qy, qz, qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['VehicleAngularVelIMU'] = np.array([wx, wy, wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['VehicleLinearAccIMU'] = np.array([ax, ay, az])

        # rospy.loginfo('vehicle imu is:' + str(imu))

    def StonePositionCB(self, data, arg):
        position = data.pose.position
        stone = arg

        x = position.x
        y = position.y
        z = position.z
        self.stones['StonePos' + str(stone)] = np.array([x, y, z])

        # rospy.loginfo('stone ' + str(stone) + ' position is:' + str(position))

    def joyCB(self, data):
        self.joycon = data.axes

    def do_action(self, agent_action):

        joymessage = Joy()

        # self.setDebugAction(action) # DEBUG
        joyactions = self.AgentToJoyAction(agent_action)  # clip actions to fit action_size

        joymessage.axes = [joyactions[0], 0., joyactions[2], joyactions[3], joyactions[4], joyactions[5], 0., 0.]

        self.joypub.publish(joymessage)
        rospy.logdebug(joymessage)

    def debugAction(self):

        actionValues = [0, 0, 1, 0, 0, -1, 0, 0]  # drive forwards
        # actionValues = [0,0,1,0,0,1,0,0]  # don't move
        return actionValues

    def __init__(self, numStones=1):
        super(BaseEnv, self).__init__()

        print('environment created!')

        self.world_state = {}
        self.stones = {}
        self.keys = {}
        self.simOn = False

        self.numStones = numStones
        self.reduced_state_space = True

        self.hist_size = 1

        # For time step
        self.current_time = time.time()
        self.last_time = self.current_time
        self.time_step = []
        self.last_obs = np.array([])
        self.TIME_STEP = 0.05  # 10 mili-seconds

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        self.subscribe_to_topics()

        self.stonePoseSubList = []
        self.stoneIsLoadedSubList = []

        for i in range(1, self.numStones + 2):
            topicName = 'stone/' + str(i) + '/Pose'
            self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, i))
        # if self.marker:
        #     topicName = 'stone/' + str(self.numStones+1) + '/Pose'
        #     self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, self.numStones+1))

        self.joysub = rospy.Subscriber('joy', Joy, self.joyCB)

        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)

        ## Define gym space - in sub envs

        # self.action_size = 4  # all actions
        # self.action_size = 3  # no pitch
        # self.action_size = 2  # without arm actions
        # self.action_size = 1  # drive only forwards

    def subscribe_to_topics(self):
        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.VehiclePositionCB)
        self.vehicleVelocitySub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped,
                                                   self.VehicleVelocityCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.vehicleImuSub = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

    def obs_space_init(self):

        self.min_pos = np.array(3 * [-500.])
        self.max_pos = np.array(3 * [500.])  # size of ground in Unity - TODO: update to room size
        min_quat = np.array(4 * [-1.])
        max_quat = np.array(4 * [1.])
        min_lin_vel = np.array(3 * [-5.])
        max_lin_vel = np.array(3 * [5.])
        min_ang_vel = np.array(3 * [-pi / 2])
        max_ang_vel = np.array(3 * [pi / 2])
        min_lin_acc = np.array(3 * [-1])
        max_lin_acc = np.array(3 * [1])
        self.min_arm_height = np.array([0.])
        self.max_arm_height = np.array([300.])
        self.min_yaw = np.array([-180.])
        self.max_yaw = np.array([180.])

        if self.reduced_state_space:
            # vehicle [x,y] pose, orientation yaw [deg] normalized by ref, linear velocity size, yaw rate,
            # linear acceleration size, arm height
            # low  = np.array([-500., -500., -180., -5., -180., -3.,   0.])
            # high = np.array([ 500.,  500.,  180.,  5.,  180.,  3., 300.])

            # delete velocities
            # vehicle [x,y] pose, orientation yaw [deg] normalized by ref, arm height
            low = np.array([self.min_pos[0], self.min_pos[1], self.min_yaw, self.min_arm_height])
            high = np.array([self.max_pos[0], self.max_pos[1], self.max_yaw, self.max_arm_height])

        else:
            # full state space
            # vehicle pose [x,y,z], vehicle quaternion [x,y,z,w], linear velocity,  angular velocity,
            # linear acceleration, arm height, blade quaternion
            low = np.concatenate(
                (self.min_pos, min_quat, min_lin_vel, min_ang_vel, min_lin_acc, self.min_arm_height, min_quat))
            high = np.concatenate(
                (self.max_pos, max_quat, max_lin_vel, max_ang_vel, max_lin_acc, self.max_arm_height, max_quat))

        # add stones depending on mission
        low, high = self._add_stones_to_state_space(low, high)

        obsSpace = spaces.Box(low=np.array([low] * self.hist_size).flatten(),
                              high=np.array([high] * self.hist_size).flatten())

        return obsSpace

    def _current_obs(self):
        # self.keys defined in each sub env

        while True:  # wait for all topics to arrive
            if all(key in self.world_state for key in self.keys):
                break

        if self.reduced_state_space:
            obs = np.array([self.world_state['VehiclePos'][0] - self.ref_pos[0],  # vehicle x pos normalized [m]
                            self.world_state['VehiclePos'][1] - self.ref_pos[1],  # vehicle y pos normalized [m]
                            self.normalize_orientation(quatToEuler(self.world_state['VehicleOrien'])[2]),
                            # yaw normalized [deg]
                            # np.linalg.norm(self.world_state['VehicleLinearVel']),                         # linear velocity size [m/s]
                            # self.world_state['VehicleAngularVel'][2]*180/pi,                              # yaw rate [deg/s]
                            # np.linalg.norm(self.world_state['VehicleLinearAccIMU']),                      # linear acceleration size [m/s^2]
                            self.world_state['ArmHeight'][0]])  # arm height [m]

        else:
            obs = np.array([])
            for key in self.keys:
                item = np.copy(self.world_state[key])
                if key == 'VehiclePos':
                    item -= self.ref_pos
                obs = np.concatenate((obs, item), axis=None)

        # add stones obs depending on mission
        obs = self._add_stones_to_obs(obs)

        return obs

    def current_obs(self):
        # wait for sim to update and obs to be different than last obs

        obs = self._current_obs()
        while True:
            if self.reduced_state_space:
                cond = np.array_equal(obs[0:3], self.last_obs[0:3])
            else:
                cond = np.array_equal(obs[0:7], self.last_obs[0:7])

            if cond:  # vehicle position and orientation
                obs = self._current_obs()
            else:
                break

        self.last_obs = obs

        return obs

    def normalize_orientation(self, yaw):
        # normalize vehicle orientation with regards to reference

        vec = self.ref_pos - self.world_state['VehiclePos']
        ref_angle = math.degrees(math.atan2(vec[1], vec[0]))
        norm_yaw = yaw - ref_angle

        return norm_yaw

    def init_env(self):
        if self.simOn:
            self.episode.killSimulation()

        self.episode = EpisodeManager()
        # self.episode.generateAndRunWholeEpisode(typeOfRand="verybasic") # for NUM_STONES = 1
        self.episode.generateAndRunWholeEpisode(typeOfRand="MultipleRocks", numstones=self.numStones,
                                                marker=self.marker)
        self.simOn = True

    def reset(self):
        # what happens when episode is done

        # clear all
        self.world_state = {}
        self.stones = {}
        self.steps = 0
        self.total_reward = 0
        self.boarders = []
        self.obs = []

        # initial state depends on environment (mission)
        self.init_env()

        # wait for simulation to set up
        while True:  # wait for all topics to arrive
            if bool(self.world_state) and bool(self.stones):  # and len(self.stones) == self.numStones + 1:
                break

        # wait for simulation to stabilize, stones stop moving
        time.sleep(5)

        if self.marker:  # push stones mission, ref = target
            self.ref_pos = self.stones['StonePos{}'.format(self.numStones + 1)]
        else:  # pick up mission, ref = stone pos
            self.ref_pos = self.stones['StonePos1']

        # # blade down near ground
        # for _ in range(30000):
        #     self.blade_down()
        DESIRED_ARM_HEIGHT = 28
        while self.world_state['ArmHeight'] > DESIRED_ARM_HEIGHT:
            self.blade_down()

        # get observation from simulation
        for _ in range(self.hist_size):
            self.obs.append(self.current_obs())

        # initial distance vehicle ref
        self.init_dis = np.linalg.norm(self.current_obs()[0:2])

        self.boarders = self.scene_boarders()

        self.joycon = 'waiting'

        return np.array(self.obs).flatten()

    def step(self, action):
        # rospy.loginfo('step func called')

        self.current_time = time.time()
        time_step = self.current_time - self.last_time

        if time_step < self.TIME_STEP:
            time.sleep(self.TIME_STEP - time_step)
            self.current_time = time.time()
            time_step = self.current_time - self.last_time

        self.time_step.append(time_step)
        self.last_time = self.current_time

        if action == 'recording':
            while self.joycon == 'waiting':  # get action from controller
                time.sleep(0.1)
            joy_action = self.joycon
            action = self.JoyToAgentAction(joy_action)
        else:
            # send action to simulation
            self.do_action(action)

        if not self.marker:  # pickup
            self.ref_pos = self.stones['StonePos1']  # update reference to current stone pose

        # get observation from simulation
        self.obs.pop(0)
        self.obs.append(self.current_obs())

        # calc step reward and add to total
        r_t = self.reward_func()

        # check if done
        done, final_reward, reset = self.end_of_episode()

        step_reward = r_t + final_reward
        self.total_reward = self.total_reward + step_reward

        if done:
            self.world_state = {}
            self.stones = {}
            print('initial distance = ', self.init_dis, ' total reward = ', self.total_reward)

        info = {"state": self.obs, "action": action, "reward": self.total_reward, "step": self.steps,
                "reset reason": reset}

        return np.array(self.obs).flatten(), step_reward, done, info

    def blade_down(self):
        # take blade down near ground at beginning of episode
        joymessage = Joy()
        joymessage.axes = [0., 0., 1., 0., -0.3, 1., 0., 0.]
        self.joypub.publish(joymessage)

    def scene_boarders(self):
        # define scene boarders depending on vehicle and stone initial positions and desired pose
        init_vehicle_pose = self.world_state['VehiclePos']
        vehicle_box = self.pose_to_box(init_vehicle_pose, box=5)

        stones_box = []
        for stone in range(1, self.numStones + 1):
            init_stone_pose = self.stones['StonePos' + str(stone)]
            stones_box = self.containing_box(stones_box, self.pose_to_box(init_stone_pose, box=5))

        scene_boarders = self.containing_box(vehicle_box, stones_box)
        if self.marker:  # push stones mission
            scene_boarders = self.containing_box(scene_boarders, self.pose_to_box(self.ref_pos[0:2], box=1))

        return scene_boarders

    def pose_to_box(self, pose, box):
        # define a box of boarders around pose (2 dim)

        return [pose[0] - box, pose[0] + box, pose[1] - box, pose[1] + box]

    def containing_box(self, box1, box2):
        # input 2 boxes and return box containing both
        if not box1:
            return box2
        else:
            x = [box1[0], box1[1], box2[0], box2[1]]
            y = [box1[2], box1[3], box2[2], box2[3]]

            return [min(x), max(x), min(y), max(y)]

    def out_of_boarders(self):
        # check if vehicle is out of scene boarders
        boarders = self.boarders
        curr_vehicle_pose = np.copy(self.world_state['VehiclePos'])

        if (curr_vehicle_pose[0] < boarders[0] or curr_vehicle_pose[0] > boarders[1] or
                curr_vehicle_pose[1] < boarders[2] or curr_vehicle_pose[1] > boarders[3]):
            return True
        else:
            return False

    def dis_stone_desired_pose(self):
        # list of stones distances from desired pose
        dis = []
        for stone in range(1, self.numStones + 1):
            current_pos = self.stones['StonePos' + str(stone)][0:2]
            dis.append(np.linalg.norm(current_pos - self.ref_pos[0:2]))

        return dis

    def dis_blade_stone(self):
        # list of distances from blade to stones
        dis = []
        blade_pose = self.blade_pose()
        for stone in range(1, self.numStones + 1):
            stone_pose = self.stones['StonePos' + str(stone)]
            dis.append(np.linalg.norm(blade_pose - stone_pose))

        return dis

    def blade_pose(self):
        L = 0.75  # distance from center of vehicle to blade BOBCAT
        r = R.from_quat(self.world_state['VehicleOrien'])

        blade_pose = self.world_state['VehiclePos'] + L * r.as_rotvec()

        return blade_pose

    def got_to_desired_pose(self):
        # check if all stones within tolerance from desired pose
        success = False
        dis = np.array(self.dis_stone_desired_pose())

        TOLERANCE = 0.75
        if all(dis < TOLERANCE):
            success = True

        return success

    def reward_func(self):
        raise NotImplementedError

    def end_of_episode(self):
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def AgentToJoyAction(self, agent_action):
        raise NotImplementedError

    def JoyToAgentAction(self, joy_action):
        raise NotImplementedError

    def _marker(self):
        raise NotImplementedError

    def _add_stones_to_obs(self, obs):
        raise NotImplementedError

    def _add_stones_to_state_space(self, low, high):
        raise NotImplementedError

    def run(self):
        # DEBUG
        obs = self.reset()
        done = False
        for _ in range(10000):
            while not done:
                action = [0, 0, 1, 0, 0, -1, 0, 0]
                obs, _, done, _ = self.step(action)


class PushAlgoryx(BaseEnv):
    MAX_STEPS = 200
    STEP_REWARD = 1 / MAX_STEPS
    FINAL_REWARD = 1

    def __init__(self):
        super(PushAlgoryx, self).__init__()
        self.hist_size = 1
        # self.world_state = {}
        # self.keys = {}
        # self.simOn = False
        # self.obs = []
        self.reduced_state_space = False
        # # For time step
        # self.current_time = time.time()
        # self.last_time = self.current_time
        # self.time_step = []
        # self.last_obs = np.array([])
        self.TIME_STEP = 0.1  # 100 mili-seconds
        self.ref_pos = np.array([15.5, -22.8, 1.5])
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1] * 4), high=np.array([1] * 4), dtype=np.float16)
        # spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        N_CHANNELS = 1
        HEIGHT = 251 # 260
        WIDTH = 150 # 160
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.world_state['GridMap'] = np.zeros(shape=(HEIGHT, WIDTH, N_CHANNELS))

        self.keys = ['GridMap','VehiclePos', 'euler_ypr',
                     'ArmHeight', 'BladePitch']

        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        self.subscribe_to_topics()

        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)



    # CALLBACKS

    def subscribe_to_topics(self):
        self.vehiclePositionSub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.VehiclePositionCB)
        # #self.vehicleVelocitySub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.VehicleVelocityCB)
        self.armHeightSub = rospy.Subscriber('/arm/height', Int32, self.ArmHeightCB)
        # #self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        # self.vehicleImuSub = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)
        self.gridMapSub = rospy.Subscriber('sl_map', GridMap, self.GridMapCB)
        self.armShortHeightSub = rospy.Subscriber('arm/shortHeight', Int32, self.ArmShortHeightCB)

    def ArmHeightCB(self, data):
        height = data.data
        self.world_state['ArmHeight'] = np.array([height])

    def ArmShortHeightCB(self, data):
        height = data.data
        self.world_state['BladePitch'] = np.array([height])

    def GridMapCB(self, data):
        raw_map = data
        hmap = np.array(data.data[1].data).reshape(
            [int(data.info.length_y / data.info.resolution), int(data.info.length_x / data.info.resolution)])
        self.world_state['GridMap'] = hmap

    def reset(self):

        # wait for topics to update
        time.sleep(1)

        # clear all
        self.world_state = {}
        self.steps = 0
        self.total_reward = 0
        self.obs = []

        # initial state depends on environment (mission)
        self.init_env()

        # wait for simulation to set up
        while True:  # wait for all topics to arrive
            if all(key in self.world_state for key in self.keys):
                break

        # DESIRED_ARM_HEIGHT = 28
        # while self.world_state['ArmHeight'] > DESIRED_ARM_HEIGHT:
        #     self.blade_down()

        # get observation from simulation
        for _ in range(self.hist_size):
            self.obs.append(self.current_obs())
            # self.obs.append(self.current_obs())

        # Since we are observing only grid_map
        # initial distance vehicle ref
        # self.init_dis = np.linalg.norm(self.current_obs()[0:2])

        # self.joycon = 'waiting'

        # current state
        self.obs = self.update_state()

        # return np.array(self.obs).flatten()
        return np.array(self.obs['h_map'])

    def current_obs(self):
        # wait for sim to update and obs to be different than last obs

        obs = self._current_obs()
        # while True:
        #     if self.reduced_state_space:
        #         cond = np.array_equal(obs[0:3], self.last_obs[0:3])
        #     else:
        #         cond = np.array_equal(obs[0:7], self.last_obs[0:7])
        #
        #     if cond:  # vehicle position and orientation
        #         obs = self._current_obs()
        #     else:
        #         break

        self.last_obs = obs

        return obs

    def update_state(self):
        h_map = self.world_state['GridMap']
        arm_lift = self.world_state['ArmHeight']
        arm_pitch = self.world_state['BladePitch']
        x_vehicle = self.world_state['VehiclePos'].item(0)
        y_vehicle = self.world_state['VehiclePos'].item(1)
        yaw = self.world_state['euler_ypr'].item(0)
        pitch = self.world_state['euler_ypr'].item(1)
        roll = self.world_state['euler_ypr'].item(2)
        # obs = {'h_map': h_map, 'x_vehicle': x_vehicle, 'y_vehicle': y_vehicle, 'arm_lift': arm_lift,
        #        'arm_pitch': arm_pitch, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
        obs = {'h_map': h_map}
        return obs

    def reward_func(self):
        arm_lift = self.world_state['ArmHeight']
        arm_pitch = self.world_state['BladePitch']
        current_pos = self.world_state['VehiclePos']
        dist = np.linalg.norm(current_pos[0:2] - self.ref_pos[0:2])
        max_dist  = 5
        normalized_dist = dist / max_dist
        arm_lift_max = 240
        arm_lift_min = 227
        arm_lift_interval = arm_lift_max - arm_lift_min
        arm_pitch_max = 135
        arm_pitch_min = 0
        arm_pitch_interval = arm_pitch_max - arm_pitch_min
        normalized_pitch = (arm_pitch_max - arm_pitch) / arm_pitch_interval
        normalized_lift = (arm_lift_max - arm_lift) / arm_lift_interval
        mse = np.mean(np.square([normalized_lift,normalized_pitch, normalized_dist]))
        return -mse/PushAlgoryx.MAX_STEPS

    def step(self, action):
        # if action:
        self.do_action(action)

        # for even time steps
        self.time_stuff()

        obs = self.update_state()

        # calc step reward and add to total
        r_t = self.reward_func()
        #always negative assert r_t > 0
        r_t= max(-0.2, self.total_reward)
        # check if done
        done, final_reward, reset = self.end_of_episode()

        step_reward = r_t + final_reward
        # self.total_reward = min(0.2, self.total_reward + step_reward)
        # self.total_reward = max(-0.2, self.total_reward)
        self.total_reward = self.total_reward + step_reward

        if done:
            self.world_state = {}
#           print('initial distance = ', self.init_dis, ' total reward = ', self.total_reward)

        info = {"state": self.obs, "action": action, "reward": self.total_reward, "step": self.steps,
                "reset reason": reset}

        return np.array(self.obs['h_map']), step_reward, done, info
        # return np.array(self.obs).flatten(), step_reward, done, info

    def time_stuff(self):
        self.current_time = time.time()
        time_step = self.current_time - self.last_time
        if time_step < self.TIME_STEP:
            time.sleep(self.TIME_STEP - time_step)
            self.current_time = time.time()
            time_step = self.current_time - self.last_time
        self.time_step.append(time_step)
        self.last_time = self.current_time

    def init_env(self):
        if self.simOn:
            self.episode.killSimulation()

        self.episode = EpisodeManager()
        # self.episode.generateAndRunWholeEpisode(typeOfRand="verybasic") # for NUM_STONES = 1
        self.episode.generateAndRunWholeEpisode(typeOfRand="AlgxVeryBasic")
        self.simOn = True

    def out_of_boarders(self):
        # check if vehicle is out of scene boarders
        boarders = [9, 20, -30, -13]
        curr_vehicle_pose = np.copy(self.world_state['VehiclePos'])
        return curr_vehicle_pose[0] < boarders[0] or curr_vehicle_pose[0] > boarders[1] or curr_vehicle_pose[1] < \
               boarders[2] or curr_vehicle_pose[1] > boarders[3]

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        if self.out_of_boarders():
            done = True
            reset = 'out of boarders' + self.world_state['VehiclePos']
            print('---------- ------', reset, '----------------')
            final_reward = - PushAlgoryx.FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        if self.steps > PushAlgoryx.MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False

        current_pos = self.world_state['VehiclePos']

        threshold = 1
        if np.linalg.norm(current_pos[0:2] - self.ref_pos[0:2]) < threshold:
            done = True
            reset = 'goal achieved'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False
            final_reward = PushAlgoryx.FINAL_REWARD

        self.steps += 1

        return done, final_reward, reset

    def _current_obs(self):
        # self.keys defined in each sub env

        while True:  # wait for all topics to arrive
            if all(key in self.world_state for key in self.keys):
                break

        if self.reduced_state_space:
            obs = np.array([self.world_state['VehiclePos'][0] - self.ref_pos[0],  # vehicle x pos normalized [m]
                            self.world_state['VehiclePos'][1] - self.ref_pos[1],  # vehicle y pos normalized [m]
                            # self.normalize_orientation(quatToEuler(self.world_state['VehicleOrien'])[2]),
                            # yaw normalized [deg]
                            # np.linalg.norm(self.world_state['VehicleLinearVel']),                         # linear velocity size [m/s]
                            # self.world_state['VehicleAngularVel'][2]*180/pi,                              # yaw rate [deg/s]
                            # np.linalg.norm(self.world_state['VehicleLinearAccIMU']),                      # linear acceleration size [m/s^2]
                            self.world_state['ArmHeight'][0]])  # arm height [m]

        else:
            obs = np.array([])
            for key in self.keys:
                item = np.copy(self.world_state[key])
                if key == 'VehiclePos':
                    item -= self.ref_pos
                obs = np.concatenate((obs, item), axis=None)



        return obs

    def do_action(self, agent_action):

        joymessage = Joy()

        joyactions = self.AgentToJoyAction(agent_action)  # clip actions to fit action_size

        joymessage.axes = [joyactions[0], 0., joyactions[2], joyactions[3], joyactions[4], joyactions[5], 0., 0.]

        joymessage.buttons = 11 * [0]
        joymessage.buttons[7] = 1  ## activation of hydraulic pump

        self.joypub.publish(joymessage)
        rospy.logdebug(joymessage)

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[2] = joyactions[5] = 1

        joyactions[0] = agent_action[0]  # vehicle turn
        joyactions[3] = agent_action[2]  # blade pitch
        joyactions[4] = agent_action[3]  # arm up/down

        if agent_action[1] < 0:  # drive backwards
            joyactions[2] = 2 * agent_action[1] + 1
            # joyactions[2] = -2*agent_action[1] - 1

        elif agent_action[1] > 0:  # drive forwards
            joyactions[5] = -2 * agent_action[1] + 1

        return joyactions


class PickUpEnv(BaseEnv):
    def __init__(self, numStones=1):  #### Number of stones ####
        BaseEnv.__init__(self, numStones)

        self.marker = False

        self._prev_stone_height = 0
        self._prev_orien = 0
        self._prev_sqr_dis_blade_stone = 0

        self.min_action = np.array(4 * [-1.])
        self.max_action = np.array(4 * [1.])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = self.obs_space_init()

        self.keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'VehicleLinearAccIMU',
                     'ArmHeight', 'BladeOrien']

    def reward_func(self):
        # reward per step
        reward = 0

        # negative reward for blade further from stone
        BLADE_CLOSER = 0.1
        self.current_sqr_dis_blade_stone = np.mean(np.power(self.dis_blade_stone(), 2))
        # if self.current_dis_blade_stone > self._prev_dis_blade_stone:
        reward += BLADE_CLOSER * (self._prev_sqr_dis_blade_stone - self.current_sqr_dis_blade_stone)

        # negative reward for orientation away from stone
        if self.reduced_state_space:
            ORIEN_CLOSER = 0.1
            self.current_orien = abs(self.current_obs()[2])
            # if self.current_orien > self._prev_orien:
            reward += ORIEN_CLOSER * (self._prev_orien - self.current_orien)

        # positive reward for lifting stone
        STONE_UP = 1.0
        self.current_stone_height = self.stones['StonePos1'][2]
        reward += STONE_UP * (self.current_stone_height - self._prev_stone_height)

        # negative reward for blade too high
        BLADE_OVER_STONE = 1.0
        MAX_BLADE_HEIGHT = 100
        if self.world_state['ArmHeight'] > MAX_BLADE_HEIGHT:
            reward -= BLADE_OVER_STONE

        # negative reward for blade over stone
        if self.stones['StonePos1'][2] < 30 and self.world_state['ArmHeight'] > 50:  # for stone scale 0.25
            reward -= BLADE_OVER_STONE

        # update for next step
        self._prev_sqr_dis_blade_stone = self.current_sqr_dis_blade_stone
        self._prev_orien = self.current_orien
        self._prev_stone_height = self.current_stone_height

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        FINAL_REWARD = 1000
        if self.out_of_boarders():
            done = True
            reset = 'out of boarders'
            print('----------------', reset, '----------------')
            final_reward = - FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        MAX_STEPS = 1000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False

        # Stone height
        HEIGHT_LIMIT = 31  # for stone size 0.25
        if self.current_stone_height >= HEIGHT_LIMIT:
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        self.steps += 1

        return done, final_reward, reset

    def _add_stones_to_obs(self, obs):
        # add stones
        if self.reduced_state_space:
            obs = np.concatenate((obs, quatToEuler(self.world_state['BladeOrien'])[0]), axis=None)  # blade pitch [deg]
            obs = np.concatenate((obs, self.stones['StonePos1'][2]), axis=None)  # stone's height
        else:
            obs = np.concatenate((obs, self.stones['StonePos1']), axis=None)  # stone's pose

        return obs

    def _add_stones_to_state_space(self, low, high):

        if self.reduced_state_space:  # add pitch and stone's height
            low = np.concatenate((low, self.min_yaw, self.min_pos[2]), axis=None)
            high = np.concatenate((high, self.max_yaw, self.max_pos[2]), axis=None)
        else:  # add stone's pose
            low = np.concatenate((low, self.min_pos), axis=None)
            high = np.concatenate((high, self.max_pos), axis=None)

        return low, high

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[0] = agent_action[0]  # vehicle turn
        joyactions[3] = agent_action[2]  # blade pitch
        joyactions[4] = agent_action[3]  # arm up/down

        # translate 4 dim agent action to 5 dim simulation action
        # agent action: [steer, speed, blade_pitch, arm_height]
        # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]

        joyactions[2] = 1.  # default value
        joyactions[5] = 1.  # default value

        if agent_action[1] < 0:  # drive backwards
            joyactions[2] = -2 * agent_action[1] - 1

        elif agent_action[1] > 0:  # drive forwards
            joyactions[5] = -2 * agent_action[1] + 1

        return joyactions

    def JoyToAgentAction(self, joy_actions):
        # translate chosen action (array) to joystick action (dict)

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


class PutDownEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_stone_pose = [250, 250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones * [False]

    def reward_func(self):
        # reward per step
        reward = -0.1

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')

        if all(self.stones_on_ground):
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = self.succ_reward()

        self.steps += 1

        return done, final_reward, reset

    def succ_reward(self):
        # end of episode reward depending on distance of stones from desired location
        reward = 1000

        for ind in range(1, self.numStones + 1):
            curret_pos = self.stones['StonePos' + str(ind)][0:2]
            dis = np.linalg.norm(curret_pos - self.desired_stone_pose)
            reward -= dis

        return reward


class MoveWithStonesEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_vehicle_pose = [250, 250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones * [False]

    def reward_func(self):
        # reward per step
        reward = -0.1

        SINGLE_STONE_FALL = 1000
        for stone in range(1, self.numStones + 1):
            if not self.stones_on_ground[stone]:
                if not self.stones['StoneIsLoaded' + str(stone)]:
                    reward -= SINGLE_STONE_FALL
                    self.stones_on_ground[stone] = True

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000
        SUCC_REWARD = 1000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')

        if self.got_to_desired_pose():
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = SUCC_REWARD

        self.steps += 1

        return done, final_reward, reset

    def got_to_desired_pose(self):
        # check if vehicle got within tolerance of desired position
        success = False

        current_pos = self.world_state['VehiclePos'][0:2]
        dis = np.linalg.norm(current_pos - self.desired_vehicle_pose)
        TOLERANCE = 0.1
        if dis < TOLERANCE:
            success = True

        return success


class PushStonesEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)

        self.marker = True

        # self._prev_mean_sqr_blade_dis = 9
        self._prev_mean_sqr_stone_dis = 16
        # self._prev_orien = 0

        self.min_action = np.array(3 * [-1.])
        self.max_action = np.array(3 * [1.])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.observation_space = self.obs_space_init()

        # self.keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel',
        #         'ArmHeight', 'BladeOrien', 'BladeAngularVel', 'BladeLinearAcc']
        self.keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'VehicleLinearAccIMU',
                     'ArmHeight']  ### reduced state space
        # self.keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'ArmHeight']  ### reduced state space no accel

    def reward_func(self):
        # reward per step
        # reward = -0.01
        reward = 0

        # reward for getting the blade closer to stone
        # BLADE_CLOSER = 0.1
        # mean_sqr_blade_dis = np.mean(np.power(self.dis_blade_stone(), 2))
        # # reward = BLADE_CLOSER / mean_sqr_blade_dis
        # reward = BLADE_CLOSER * (self._prev_mean_sqr_blade_dis - mean_sqr_blade_dis)

        # reward for getting the stone closer to target
        STONE_CLOSER = 1
        mean_sqr_stone_dis = np.mean(np.power(self.dis_stone_desired_pose(), 2))
        # reward += STONE_CLOSER / mean_sqr_stone_dis
        reward += STONE_CLOSER * (self._prev_mean_sqr_stone_dis - mean_sqr_stone_dis)

        # negative reward for orientation away from reference
        # if self.reduced_state_space:
        #     ORIEN_CLOSER = 0.1
        #     self.current_orien = abs(self.current_obs()[2])
        #     # if self.current_orien > self._prev_orien:
        #     # reward += ORIEN_CLOSER * (self._prev_orien - self.current_orien)
        #     reward -= ORIEN_CLOSER * self.current_orien

        # update prevs
        # self._prev_mean_sqr_blade_dis = mean_sqr_blade_dis
        self._prev_mean_sqr_stone_dis = mean_sqr_stone_dis
        # self._prev_orien = self.current_orien

        # STONE_CLOSER = 0.1
        # diff_from_init_dis = self.init_dis_stone_desired_pose - np.mean(self.sqr_dis_stone_desired_pose())
        # reward += STONE_CLOSER*diff_from_init_dis

        # for number of stones = 1
        # STONE_MIDDLE_BLADE = 0.5
        # reward += STONE_MIDDLE_BLADE / self.sqr_dis_optimal_stone_pose()

        # # positive reward if stone is closer to desired pose, negative if further away
        # STONE_CLOSER = 10
        # self.current_stone_dis = self.sqr_dis_stone_desired_pose()
        # if bool(self.last_stone_dis): # don't enter first time when last_stone_dis is empty
        #     diff = [curr - last for curr, last in zip(self.current_stone_dis, self.last_stone_dis)]
        #     if any(item < 0 for item in diff): # stone closer
        #         reward += STONE_CLOSER / np.mean(self.current_stone_dis)
        #     # if any(item > 0 for item in diff): # stone further away
        #     #     reward -= STONE_CLOSER / np.mean(self.current_stone_dis)
        #     # reward -= STONE_CLOSER*np.mean(diff)
        #
        # self.last_stone_dis = self.current_stone_dis
        # #
        # #     # if any(True for curr, last in zip(self.current_stone_dis, self.last_stone_dis) if curr < last):
        # #     #     reward += STONE_CLOSER
        # #         # rospy.loginfo('---------------- STONE closer, positive reward +10 ! ----------------')
        #
        # # # positive reward if blade is closer to stone's current pose, negative if further away
        # BLADE_CLOSER = 1
        # self.current_blade_dis = self.sqr_dis_blade_stone()
        # if bool(self.last_blade_dis): # don't enter first time when last_blade_dis is empty
        #     diff = [curr - last for curr, last in zip(self.current_blade_dis, self.last_blade_dis)]
        #     if any(item < 0 for item in diff): # blade closer
        #         reward += BLADE_CLOSER / np.mean(self.current_blade_dis)
        #     if any(item > 0 for item in diff): # blade further away
        #         reward -= BLADE_CLOSER / np.mean(self.current_blade_dis)
        #     # reward -= BLADE_CLOSER*np.mean(diff)
        #
        # self.last_blade_dis = self.current_blade_dis
        # #
        # #     if any(True for curr, last in zip(self.current_blade_dis, self.last_blade_dis) if curr < last):
        # #         reward += BLADE_CLOSER
        # #         # rospy.loginfo('----------------  BLADE closer, positive reward +1 ! ----------------')
        #
        # # for number of stones = 1
        # STONE_MIDDLE_BLADE = 0.5
        # self.current_stone_middle_blade_dis = self.sqr_dis_optimal_stone_pose()
        # diff = self.current_stone_middle_blade_dis - self.last_stone_middle_blade_dis
        # if diff < 0:
        #     reward += STONE_MIDDLE_BLADE / self.current_stone_middle_blade_dis
        # if diff > 0:
        #     reward -= STONE_MIDDLE_BLADE / self.current_stone_middle_blade_dis
        #
        # self.last_stone_middle_blade_dis = self.current_stone_middle_blade_dis

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        FINAL_REWARD = 5000
        if self.out_of_boarders():
            done = True
            reset = 'out of boarders'
            print('----------------', reset, '----------------')
            final_reward = - FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        MAX_STEPS = 250 * self.init_dis
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')
            # final_reward = - FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        if self.got_to_desired_pose():
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = FINAL_REWARD * MAX_STEPS / self.steps
            # final_reward = FINAL_REWARD
            # print('----------------', str(final_reward), '----------------')
            self.episode.killSimulation()
            self.simOn = False

        self.steps += 1

        return done, final_reward, reset

    def _add_stones_to_obs(self, obs):
        # add stones
        for ind in range(1, self.numStones + 1):
            item = np.copy(self.stones['StonePos' + str(ind)])
            item -= self.ref_pos
            obs = np.concatenate((obs, item), axis=None)

        return obs

    def _add_stones_to_state_space(self, low, high):
        # add stones' positions
        for ind in range(1, self.numStones + 1):
            low = np.concatenate((low, self.min_pos), axis=None)
            high = np.concatenate((high, self.max_pos), axis=None)

        return low, high

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[0] = agent_action[0]  # vehicle turn

        joyactions[4] = agent_action[2]  # arm up/down

        # translate 4 dim agent action to 5 dim simulation action
        # agent action: [steer, speed, blade_pitch, arm_height]
        # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]

        joyactions[2] = 1.  # default value
        joyactions[5] = 1.  # default value

        if agent_action[1] < 0:  # drive backwards
            joyactions[2] = -2 * agent_action[1] - 1

        elif agent_action[1] > 0:  # drive forwards
            joyactions[5] = -2 * agent_action[1] + 1

        return joyactions

    def JoyToAgentAction(self, joy_actions):
        # translate chosen action (array) to joystick action (dict)

        agent_action = np.zeros(3)

        agent_action[0] = joy_actions[0]  # vehicle turn
        agent_action[2] = joy_actions[4]  # arm up/down

        # translate 5 dim joystick actions to 4 dim agent action
        # agent action: [steer, speed, blade_pitch, arm_height]
        # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]
        # all [-1, 1]

        agent_action[1] = 0.5 * (joy_actions[2] - 1) + 0.5 * (1 - joy_actions[5])  ## forward backward

        return agent_action

# DEBUG
# if __name__ == '__main__':
#     # from stable_baselines.common.env_checker import check_env
#     #
#     # env = PickUpEnv()
#     # # It will check your custom environment and output additional warnings if needed
#     # check_env(env)
#
#     node = PushStonesEnv(1)
#     node.run()
