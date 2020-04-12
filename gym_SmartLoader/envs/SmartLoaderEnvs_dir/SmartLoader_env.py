# !/usr/bin/env python3
# building custom gym environment:
# # https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5
# # for testing
import gym

# class PickUpEnv(gym.Env):
#     def __init__(self):
#         print("Environment initialized")
#     def step(self):
#         print("Step successful!")
#     def reset(self):
#         print("Environment reset")

import sys
import time
from src.EpisodeManager import *
import src.Unity2RealWorld as urw
import gym
from gym import spaces
import numpy as np
from math import pi as pi
from scipy.spatial.transform import Rotation as R
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32, Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped



class BaseEnv(gym.Env):

    def VehiclePositionCB(self,stamped_pose):
        # new_stamped_pose = urw.positionROS2RW(stamped_pose)
        x = stamped_pose.pose.position.x
        y = stamped_pose.pose.position.y
        z = stamped_pose.pose.position.z
        self.world_state['VehiclePos'] = np.array([x,y,z])

        qx = stamped_pose.pose.orientation.x
        qy = stamped_pose.pose.orientation.y
        qz = stamped_pose.pose.orientation.z
        qw = stamped_pose.pose.orientation.w
        self.world_state['VehicleOrien'] = np.array([qx,qy,qz,qw])

        # rospy.loginfo('position is:' + str(stamped_pose.pose))

    def VehicleVelocityCB(self, stamped_twist):
        vx = stamped_twist.twist.linear.x
        vy = stamped_twist.twist.linear.y
        vz = stamped_twist.twist.linear.z
        self.world_state['VehicleLinearVel'] = np.array([vx,vy,vz])

        wx = stamped_twist.twist.angular.x
        wy = stamped_twist.twist.angular.y
        wz = stamped_twist.twist.angular.z
        self.world_state['VehicleAngularVel'] = np.array([wx,wy,wz])

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
        self.world_state['BladeOrien'] = np.array([qx,qy,qz,qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['BladeAngularVel'] = np.array([wx,wy,wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['BladeLinearAcc'] = np.array([ax,ay,az])

        # rospy.loginfo('blade imu is:' + str(imu))

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

        # rospy.loginfo('vehicle imu is:' + str(imu))

    def StonePositionCB(self, data, arg):
        position = data.pose.position
        stone = arg

        x = position.x
        y = position.y
        z = position.z
        self.stones['StonePos' + str(stone)] = np.array([x,y,z])

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

        actionValues = [0,0,1,0,0,-1,0,0]  # drive forwards
        # actionValues = [0,0,1,0,0,1,0,0]  # don't move
        return actionValues

    def AgentToJoyAction(self, agent_action):
        # translate chosen action (array) to joystick action (dict)

        joyactions = np.zeros(6)

        joyactions[0] = agent_action[0] # vehicle turn
        joyactions[3] = agent_action[3] # blade pitch
        joyactions[4] = agent_action[2] # arm up/down

        # translate 4 dim agent action to 5 dim simulation action
        # agent action: [steer, speed, blade_pitch, arm_height]
        # simulation joystick actions: [steer, speed backwards, blade pitch, arm height, speed forwards]

        joyactions[2] = 1. # default value
        joyactions[5] = 1. # default value

        if agent_action[1] < 0: # drive backwards
            joyactions[2] = -2*agent_action[1] - 1

        elif agent_action[1] > 0: # drive forwards
            joyactions[5] = -2*agent_action[1] + 1

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

        agent_action[1] = 0.5*(joy_actions[2] - 1) + 0.5*(1 - joy_actions[5])    ## forward backward

        return agent_action


    def __init__(self,numStones=1):
        super(BaseEnv, self).__init__()

        print('environment created!')

        self.world_state = {}
        self.stones = {}
        self.simOn = False

        self.numStones = numStones
        self.marker = False # True for Push Stones env, False for Pick Up env

        self.hist_size = 1

        # For time step
        self.current_time = time.time()
        self.last_time = self.current_time
        self.time_step = []
        self.last_obs = np.array([])
        self.TIME_STEP = 0.05 # 10 mili-seconds

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.VehiclePositionCB)
        self.vehicleVelocitySub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.VehicleVelocityCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.vehicleImuSub = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

        self.stonePoseSubList = []
        self.stoneIsLoadedSubList = []

        for i in range(1, self.numStones+1):
            topicName = 'stone/' + str(i) + '/Pose'
            self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, i))
        if self.marker:
            topicName = 'stone/' + str(self.numStones+1) + '/Pose'
            self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, self.numStones+1))

        self.joysub = rospy.Subscriber('joy', Joy, self.joyCB)

        self.joypub = rospy.Publisher("joy", Joy, queue_size=10)

        ## Define gym space
        min_action = np.array(4*[-1.])
        max_action = np.array(4*[ 1.])
        # self.action_size = 4  # all actions
        # self.action_size = 3  # no pitch
        # self.action_size = 2  # without arm actions
        # self.action_size = 1  # drive only forwards

        self.action_space = spaces.Box(low=min_action, high=max_action)
        self.observation_space = self.obs_space_init()

    def obs_space_init(self):
        # obs = [local_pose:(x,y,z), local_orien_quat:(x,y,z,w)
        #        velocity: linear:(vx,vy,vz), angular:(wx,wy,wz)
        #        arm_height: h
        #        arm_imu: orein_quat:(x,y,z,w), vel:(wx,wy,wz), acc:(ax,ay,az)
        #        stone<id>: pose:(x,y,z)]

        min_pos = np.array(3*[-500.])
        max_pos = np.array(3*[ 500.]) # size of ground in Unity - TODO: update to room size
        min_quat = np.array(4*[-1.])
        max_quat = np.array(4*[ 1.])
        min_lin_vel = np.array(3*[-5.])
        max_lin_vel = np.array(3*[ 5.])
        min_ang_vel = np.array(3*[-pi/2])
        max_ang_vel = np.array(3*[ pi/2])
        min_lin_acc = np.array(3*[-1])
        max_lin_acc = np.array(3*[ 1])
        min_arm_height = np.array([0.])
        max_arm_height = np.array([100.])

        # FIRST STATE SPACE
        # ["VehiclePos","VehicleOrien","VehicleLinearVel","VehicleAngularVel","ArmHeight","BladeOrien","BladeAngularVel","BladeLinearAcc","Stones"]
        # low  = np.concatenate((min_pos,min_quat,min_lin_vel,min_ang_vel,min_arm_height,min_quat,min_ang_vel,min_lin_acc), axis=None)
        # high = np.concatenate((max_pos,max_quat,max_lin_vel,max_ang_vel,max_arm_height,max_quat,max_ang_vel,max_lin_acc), axis=None)

        # NEW SMALLER STATE SPACE:
        # ["VehiclePos","VehicleOrien","VehicleLinearVel","VehicleAngularVel","VehicleLinearAccIMU","ArmHeight","Stones"]
        # low  = np.concatenate((min_pos, min_quat, min_lin_vel, min_ang_vel, min_lin_acc, min_arm_height))
        # high = np.concatenate((max_pos, max_quat, max_lin_vel, max_ang_vel, max_lin_acc, max_arm_height))

        # PICK UP ENV STATE SPACE
        # ["VehiclePos","VehicleOrien","VehicleLinearVel","VehicleAngularVel","VehicleLinearAccIMU","ArmHeight","BladeOrien","Stones"]
        low  = np.concatenate((min_pos, min_quat, min_lin_vel, min_ang_vel, min_lin_acc, min_arm_height, min_quat))
        high = np.concatenate((max_pos, max_quat, max_lin_vel, max_ang_vel, max_lin_acc, max_arm_height, max_quat))

        for ind in range(1, self.numStones + 1):
            low  = np.concatenate((low, min_pos), axis=None)
            high = np.concatenate((high, max_pos), axis=None)

        obsSpace = spaces.Box(low=np.array([low] * self.hist_size).flatten(),
                              high=np.array([low] * self.hist_size).flatten())

        return obsSpace

    def _current_obs(self):

        obs = np.array([])
        # keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel',
        #         'ArmHeight', 'BladeOrien', 'BladeAngularVel', 'BladeLinearAcc']
        # keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'VehicleLinearAccIMU', 'ArmHeight'] ### reduced state space
        # keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'ArmHeight']  ### reduced state space
        keys = ['VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'VehicleLinearAccIMU',
                'ArmHeight', 'BladeOrien'] # PICK UP ENV STATE SPACE

        while True: # wait for all topics to arrive
            if all(key in self.world_state for key in keys):
                break

        for key in keys:
            item = np.copy(self.world_state[key])
            if key == 'VehiclePos':
                item -= self.ref_pos
            obs = np.concatenate((obs, item), axis=None)

        for ind in range(1, self.numStones+1):
            item = np.copy(self.stones['StonePos' + str(ind)])
            item -= self.ref_pos
            obs = np.concatenate((obs, item), axis=None)

        return obs

    def current_obs(self):
        # wait for sim to update and obs to be different than last obs

        obs = self._current_obs()
        while True:
            if np.array_equal(obs[0:7], self.last_obs[0:7]): # vehicle position and orientation
                obs = self._current_obs()
            else:
                break

        self.last_obs = obs

        return obs

    def init_env(self):
        if self.simOn:
            self.episode.killSimulation()

        self.episode = EpisodeManager()
        # self.episode.generateAndRunWholeEpisode(typeOfRand="verybasic") # for NUM_STONES = 1
        self.episode.generateAndRunWholeEpisode(typeOfRand="MultipleRocks", numstones=self.numStones, marker=self.marker)
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
        while True: # wait for all topics to arrive
            if bool(self.world_state) and bool(self.stones): # and len(self.stones) == self.numStones + 1:
                break

        # wait for simulation to stabilize, stones stop moving
        time.sleep(5)

        if self.marker:
            self.ref_pos = self.stones['StonePos{}'.format(self.numStones + 1)]
        else:
            self.ref_pos = self.stones['StonePos1']

        # # blade down near ground
        # for _ in range(30000):
        #     self.blade_down()
        # DESIRED_ARM_HEIGHT = 22
        # while self.world_state['ArmHeight'] > DESIRED_ARM_HEIGHT:
        #     self.blade_down()

        # get observation from simulation
        for _ in range(self.hist_size):
            self.obs.append(self.current_obs())# without waiting for obs to updated

        self.init_dis = np.sqrt(np.sum(np.power(self.current_obs()[0:3], 2)))

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
            print('total reward = ', self.total_reward)

        info = {"state": self.obs, "action": action, "reward": self.total_reward, "step": self.steps, "reset reason": reset}

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
        if self.marker:
            scene_boarders = self.containing_box(scene_boarders, self.pose_to_box(self.ref_pos[0:2], box=5))

        return scene_boarders

    def pose_to_box(self, pose, box):
        # define a box of boarders around pose (2 dim)

        return [pose[0]-box, pose[0]+box, pose[1]-box, pose[1]+box]

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

    def reward_func(self):
        raise NotImplementedError

    def end_of_episode(self):
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def run(self):
        # DEBUG
        obs = self.reset()
        done = False
        for _ in range(10000):
            while not done:
                action = [0, 0, 1, 0, 0, -1, 0, 0]
                obs, _, done, _ = self.step(action)


class PickUpEnv(BaseEnv):
    def __init__(self, numStones=1): #### Number of stones ####
        BaseEnv.__init__(self, numStones)
        self.current_stone_height = 0
        self._prev_stone_height = 0
        # self.current_dis_blade_stone = 0
        # self._prev_dis_blade_stone = 0

    def reward_func(self):
        # reward per step
        reward = 0

        # reward for getting the blade closer to stone
        # BLADE_CLOSER = 0.1
        # self.current_dis_blade_stone = self.sqr_dis_blade_stone()
        # reward += BLADE_CLOSER * (self._prev_dis_blade_stone - self.current_dis_blade_stone)

        STONE_UP = 1.0
        self.current_stone_height = self.stones['StonePos1'][2]
        reward += STONE_UP * (self.current_stone_height - self._prev_stone_height)

        # BLADE_OVER_STONE = 1.0
        # MAX_BLADE_HEIGHT = 100
        # if self.world_state['ArmHeight'] > MAX_BLADE_HEIGHT:
        #     reward -= BLADE_OVER_STONE

        # self._prev_dis_blade_stone = self.current_dis_blade_stone
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
            print('----------------', reset ,'----------------')
            self.episode.killSimulation()
            self.simOn = False

        # Stone height
        HEIGHT_LIMIT = 31 # for stone size 0.25
        if self.current_stone_height >= HEIGHT_LIMIT:
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False

        self.steps += 1

        return done, final_reward, reset

    def blade_pose(self):
        L = 0.75 # distance from center of vehicle to blade BOBCAT
        r = R.from_quat(self.world_state['VehicleOrien'])

        blade_pose = self.world_state['VehiclePos'] + L*r.as_rotvec()

        return blade_pose

    def sqr_dis_blade_stone(self):
        # list of distances from blade to stones

        blade_pose = self.blade_pose()
        stone_pose = self.stones['StonePos1']
        sqr_dis = self.squared_dis(blade_pose, stone_pose)

        return sqr_dis

    def squared_dis(self, p1, p2):
        # calc distance between two points
        # p1,p2 = [x,y,z]

        squared_dis = pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2) + pow(p1[2]-p2[2], 2)

        return squared_dis


class PutDownEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_stone_pose = [250, 250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones*[False]

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
            print('----------------', reset ,'----------------')

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
        self.desired_vehicle_pose = [250,250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones*[False]

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
    def __init__(self, numStones=3):
        BaseEnv.__init__(self, numStones)

        # self._prev_mean_sqr_blade_dis = 9
        self._prev_mean_sqr_stone_dis = 16

    def reward_func(self):

        # reward per step
        reward = 0

        # reward for getting the blade closer to stone
        # BLADE_CLOSER = 0.1
        # mean_sqr_blade_dis = np.mean(self.sqr_dis_blade_stone())
        # # reward = BLADE_CLOSER / mean_sqr_blade_dis
        # reward = BLADE_CLOSER * (self._prev_mean_sqr_blade_dis - mean_sqr_blade_dis)

        # reward for getting the stone closer to target
        STONE_CLOSER = 1
        mean_sqr_stone_dis = np.mean(np.power(self.dis_stone_desired_pose(), 2))
        # reward += STONE_CLOSER / mean_sqr_stone_dis
        reward += STONE_CLOSER * (self._prev_mean_sqr_stone_dis - mean_sqr_stone_dis)

        # update prevs
        # self._prev_mean_sqr_blade_dis = mean_sqr_blade_dis
        self._prev_mean_sqr_stone_dis = mean_sqr_stone_dis

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

        MAX_STEPS = 200*self.init_dis
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
            final_reward = FINAL_REWARD*MAX_STEPS/self.steps
            # final_reward = FINAL_REWARD
            # print('----------------', str(final_reward), '----------------')
            self.episode.killSimulation()
            self.simOn = False

        self.steps += 1

        return done, final_reward, reset

    def dis_stone_desired_pose(self):
        # list of stones distances from desired pose

        dis = []
        for stone in range(1, self.numStones + 1):
            current_pos = self.stones['StonePos' + str(stone)][0:2]
            dis.append(np.linalg.norm(current_pos - self.ref_pos[0:2]))

        return dis

    def got_to_desired_pose(self):
        # check if all stones within tolerance from desired pose

        success = False
        dis = np.array(self.dis_stone_desired_pose())

        TOLERANCE = 0.75
        if all(dis < TOLERANCE):
            success = True

        return success

    # def blade_got_to_stone(self):
    #     # check if blade got to stone within tolerance
    #     success = False
    #     sqr_dis = self.sqr_dis_blade_stone()
    #
    #     TOLERANCE = 5
    #     if all(item < TOLERANCE for item in sqr_dis):
    #         success = True
    #
    #     return success

    # def blade_pose(self):
    #     L = 0.75 # distance from center of vehicle to blade BOBCAT
    #     r = R.from_quat(self.world_state['VehicleOrien'])
    #
    #     blade_pose = self.world_state['VehiclePos'] + L*r.as_rotvec()
    #
    #     return blade_pose

    # def stone_optimal_pose(self):
    #     # using current blade to stone distance to calc optimal position of stone for pushing from the middle of the blade
    #     L = pow(self.sqr_dis_blade_one_stone(), 0.5) # distance from center of vehicle to blade BOBCAT
    #     r = R.from_quat(self.world_state['VehicleOrien'])
    #
    #     optimal_pose = self.world_state['VehiclePos'] + L*r.as_rotvec()
    #
    #     return optimal_pose
    #
    # def sqr_dis_optimal_stone_pose(self):
    #     # for number of stones = 1
    #
    #     optimal_pose = self.stone_optimal_pose()[0:2]
    #     stone_pose = self.stones['StonePos1'][0:2]
    #     sqr_dis = self.squared_dis(optimal_pose, stone_pose)
    #
    #     return sqr_dis

    # def sqr_dis_blade_stone(self):
    #     # list of distances from blade to stones
    #     sqr_dis = []
    #     blade_pose = self.blade_pose()[0:2]
    #     for stone in range(1, self.numStones + 1):
    #         stone_pose = self.stones['StonePos' + str(stone)][0:2]
    #         sqr_dis.append(self.squared_dis(blade_pose, stone_pose))
    #
    #     return sqr_dis

    # def sqr_dis_blade_one_stone(self):
    #     # for number of stones = 1
    #     blade_pose = self.blade_pose()[0:2]
    #     stone_pose = self.stones['StonePos1'][0:2]
    #     sqr_dis = self.squared_dis(blade_pose, stone_pose)
    #
    #     return sqr_dis
    #
    # def squared_dis(self, p1, p2):
    #     # calc distance between two points
    #     # p1,p2 = [x,y]
    #
    #     squared_dis = pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2)
    #
    #     return squared_dis


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