import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap
from gym import spaces
from sensor_msgs.msg import Joy
from src.EpisodeManager import EpisodeManager
from std_msgs.msg import Int32

from gym_SmartLoader.envs.SmartLoaderEnvs_dir import BaseEnv


class PushAlgoryx(BaseEnv):
    MAX_STEPS = 200
    STEP_REWARD = 1 / MAX_STEPS
    FINAL_REWARD = 1.0

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
        self._start_pos = np.array([12.86, -23.23, 1.578])
        self.ref_pos = np.array([12.86, -14.66, 1.5])
        # self.ref_pos = np.array([15.5, -22.8, 1.5])
        self._boarders = [9, 16, -28, -10]

        # takeOne: boarders = [0.5, 20, -30, -13]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1] * 4), high=np.array([1] * 4), dtype=np.float16)
        # spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        N_CHANNELS = 1
        self.MAP_SIZE_Y = 291  # 260
        self.MAP_SIZE_X = 150  # 160
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(43657,), dtype=np.uint8)

        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(self.MAP_SIZE_Y, self.MAP_SIZE_X, N_CHANNELS), dtype=np.uint8)

        self.world_state['GridMap'] = np.zeros(shape=(self.MAP_SIZE_Y, self.MAP_SIZE_X, N_CHANNELS))

        self.keys = ['GridMap', 'VehiclePos', 'euler_ypr',
                     'ArmHeight', 'BladePitch']

        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        self.subscribe_to_topics()

        self.joypub = rospy.Publisher('joy', Joy, queue_size=10)

        self._obs = []

        self.total_reward = 0.0
        self.last_best_model = None

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
        if ((int(data.info.length_y / data.info.resolution) != self.MAP_SIZE_Y) or (
                int(data.info.length_x / data.info.resolution) != self.MAP_SIZE_X)):
            print(
                "Do not replace map" + "(" + self.MAP_SIZE_X.__str__() + ")" + "(" + self.MAP_SIZE_Y.__str__() + ") with (" + int(
                    data.info.length_x / data.info.resolution).__str__() + "," + int(
                    data.info.length_y / data.info.resolution).__str__())
        else:
            hmap = np.array(data.data[1].data).reshape([self.MAP_SIZE_Y, self.MAP_SIZE_X, 1])
            #            hmap = np.array(data.data[1].data).reshape([int(data.info.length_y / data.info.resolution), int(data.info.length_x / data.info.resolution)])

            self.world_state['GridMap'] = hmap

    def reset(self):

        # wait for topics to update
        time.sleep(1)

        # clear all
        self.world_state = {}
        self.steps = 0
        self.total_reward = 0.0
        self._obs = []

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
        # for _ in range(self.hist_size):
        #     self._obs.append(self.current_obs())
        #     # self.obs.append(self.current_obs())

        # Since we are observing only grid_map
        # initial distance vehicle ref
        # self.init_dis = np.linalg.norm(self.current_obs()[0:2])

        # self.joycon = 'waiting'

        # current state
        self._obs = self.update_state()

        # stack all observation for one very long input
        return np.array(np.hstack(self._obs.values()))

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
        h_map_flat = h_map.flatten()
        obs = {'h_map': h_map_flat, 'x_vehicle': x_vehicle, 'y_vehicle': y_vehicle, 'arm_lift': arm_lift,
               'arm_pitch': arm_pitch, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
        # obs = {'h_map': h_map}
        return obs

    def reward_func(self):
        arm_lift = self.world_state['ArmHeight']
        arm_pitch = self.world_state['BladePitch']
        current_pos = self.world_state['VehiclePos']
        dist = np.linalg.norm(current_pos[0:2] - self.ref_pos[0:2])
        max_dist = 10
        normalized_dist = dist / max_dist
        arm_lift_max = 240
        arm_lift_min = 227
        arm_lift_interval = arm_lift_max - arm_lift_min
        arm_pitch_max = 135
        arm_pitch_min = 0
        arm_pitch_interval = arm_pitch_max - arm_pitch_min

        # multiply by factor to make blade pose more important
        factor = 2.0
        normalized_pitch = factor * (arm_pitch_max - arm_pitch) / arm_pitch_interval
        normalized_lift = factor *  (arm_lift_max - arm_lift) / arm_lift_interval

        mse = np.mean(np.square([normalized_lift, normalized_pitch, normalized_dist])).squeeze()
        return -mse / PushAlgoryx.MAX_STEPS

    def step(self, action):
        # if action:
        self.do_action(action)

        # for even time steps
        self.time_stuff()

        self._obs = self.update_state()

        # calc step reward and add to total
        r_t = self.reward_func()
        assert r_t < 0
        # r_t= max(-0.2, r_t)
        r_t *= 0.1
        # check if done
        done, final_reward, reset = self.end_of_episode()

        step_reward = r_t + final_reward
        # self.total_reward = min(0.2, self.total_reward + step_reward)
        # self.total_reward = max(-0.2, self.total_reward)
        self.total_reward = self.total_reward + step_reward

        self.done = done
        if done:
            self.world_state = {}

        #           print('initial distance = ', self.init_dis, ' total reward = ', self.total_reward)

        info = {"action": action, "reward": self.total_reward, "step": self.steps,
                "reset reason": reset, "r_t": r_t, "final_reward": final_reward}

        self.last_rt = r_t
        self.last_final_reward = final_reward

        # return np.array(self._obs['h_map']), step_reward, done, info
        return np.array(np.hstack(self._obs.values())), step_reward, done, info
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
        # start local: x=12, y=-23, z=1.578

        curr_vehicle_pose = np.copy(self.world_state['VehiclePos'])
        return curr_vehicle_pose[0] < self._boarders[0] or curr_vehicle_pose[0] > self._boarders[1] or \
               curr_vehicle_pose[1] < \
               self._boarders[2] or curr_vehicle_pose[1] > self._boarders[3]

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0
        current_pos = self.world_state['VehiclePos']
        threshold = 4
        if self.out_of_boarders():
            done = True
            reset = 'out of boarders' + np.copy(self.world_state['VehiclePos']).__str__()
            print('---------- ------', reset, '----------------')
            final_reward = - PushAlgoryx.FINAL_REWARD
            self.episode.killSimulation()
            self.simOn = False
        elif self.steps > PushAlgoryx.MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False
        elif np.linalg.norm(current_pos[0:2] - self.ref_pos[0:2]) < threshold:
            done = True
            reset = 'goal achieved'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False
            final_reward = PushAlgoryx.FINAL_REWARD
        elif self.world_state['ArmHeight'] < 233 and self.steps > PushAlgoryx.MAX_STEPS / 2.0:
            done = True
            reset = 'Arm Too High For Too Long'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False
            final_reward = - PushAlgoryx.FINAL_REWARD / 2.0

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
