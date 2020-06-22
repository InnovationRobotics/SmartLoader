import time
import numpy as np
from matplotlib import pyplot as plt
import math


def quatToEuler(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return np.array([X, Y, Z])

class PushPidAlgoryx:
    def __init__(self):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.0005, I=0.0001, D=0, saturation=0.8)
        # self.lift_pid = PID(P=1, I=0, D=0, saturation=1)
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=0.005, I=0.0001, D=0, saturation=1)
        # self.pitch_pid = PID(P=1, I=0, D=0, saturation=1)
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # steer PID
        # self.steer_pid = PID(P=0.0002, I=0, D=-0.00005, saturation=0.008)
        # self.steer_pid.setSampleTime(self.TIME_STEP)

        # speed PID
        self.speed_pid = PID(P=0.05, I=0, D=0.001, saturation=0.6)
        self.speed_pid.setSampleTime(self.TIME_STEP)

    def step(self, obs, des):
        current_lift = obs['arm_lift'].item(0)
        current_pitch = obs['arm_pitch'].item(0)

        # des = [x_vehicle, y_vehicle, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

        # # for connecting lift and speed demands and actions
        # lift_error = abs(des[2] - current_lift)
        # lift_speed_factor = np.clip(5/abs(des[2] - current_lift), 0.35, 1)
        # print(lift_speed_factor)

        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = 30+obs['y_vehicle']
        self.speed_pid.SetPoint = x_des

        # steer error: angle between desired and current location - current blade orientation
        # self.steer_pid.SetPoint = np.math.degrees(np.math.atan2(y_des-obs['y_blade'], x_des-obs['x_blade']))
        # current_steer = np.math.degrees(np.math.atan2(obs['y_blade']-obs['y_vehicle'], obs['x_blade']-obs['x_vehicle']))

        # print('current steer = ', current_steer, 'desired steer = ', self.steer_pid.SetPoint)

        # pid update
        if current_lift < 150:
            lift_action = 0
        else:
            lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)
        # steer_action = self.steer_pid.update(current_steer)
        speed_action = self.speed_pid.update(current_speed)

        # fix lift action
        if -1 <= lift_action < 0:
            lift_action = -lift_action
        else:
            lift_action = lift_action - 1
        # for crazy stuff
        # if 0 < lift_action <= 1:
        #     lift_action = -lift_action
        # else:
        #     lift_action = lift_action - 1

        # do action
        action = np.array([0, speed_action, pitch_action, lift_action])

        # save data
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)
        # self.steer_pid.save_data(current_steer, steer_action, self.steer_pid.SetPoint)
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)

        return action


class DriveBackAndLiftPidAlgoryx:
    def __init__(self, des):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # speed PID
        self.speed_pid = PID(P=0.1, I=0.01, D=0.01, saturation=0.6)
        self.speed_pid.setSampleTime(self.TIME_STEP)

        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.005, I=0, D=0, saturation=1)
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.005, I=0, D=0, saturation=1)
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # des = [x_blade, y_blade, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

    def step(self, obs, des, steer=True):
        current_lift = obs['lift']
        current_pitch = obs['pitch']

        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = obs['x_vehicle']
        self.speed_pid.SetPoint = x_des

        # pid update
        lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)

        # if got to des pos, stop driving but continue blade movement
        # if obs['x_vehicle'] <= x_des:
        #     speed_action = 0
        # else:
        speed_action = self.speed_pid.update(current_speed)

        # do action
        action = np.array([0, speed_action, pitch_action, lift_action])

        # save data
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)

        return action


class PushPid:
    def __init__(self):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.00055, I=0.00015, D=0.00001, saturation=0.8)
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.0065, I=0, D=0.0002, saturation=0.5)
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # steer PID
        # self.steer_pid = PID(P=0.0002, I=0, D=-0.00005, saturation=0.008)
        # self.steer_pid.setSampleTime(self.TIME_STEP)

        # speed PID
        self.speed_pid = PID(P=1.2, I=0.1, D=0.1, saturation=0.6)
        self.speed_pid.setSampleTime(self.TIME_STEP)

    def step(self, obs, des):
        current_lift = obs['lift']
        current_pitch = obs['pitch']

        # des = [x_blade, y_blade, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

        # # for connecting lift and speed demands and actions
        # lift_error = abs(des[2] - current_lift)
        # lift_speed_factor = np.clip(5/abs(des[2] - current_lift), 0.35, 1)
        # print(lift_speed_factor)

        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = obs['x_blade']
        self.speed_pid.SetPoint = x_des

        # steer error: angle between desired and current location - current blade orientation
        # self.steer_pid.SetPoint = np.math.degrees(np.math.atan2(y_des-obs['y_blade'], x_des-obs['x_blade']))
        # current_steer = np.math.degrees(np.math.atan2(obs['y_blade']-obs['y_vehicle'], obs['x_blade']-obs['x_vehicle']))

        # print('current steer = ', current_steer, 'desired steer = ', self.steer_pid.SetPoint)

        # pid update
        lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)
        # steer_action = self.steer_pid.update(current_steer)
        speed_action = self.speed_pid.update(current_speed)

        # # for connecting lift and speed demands and actions
        # if lift_error > 10:
        #     speed_action = 0.5*speed_action

        # do action
        action = np.array([0, speed_action, pitch_action, lift_action])

        # save data
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)
        # self.steer_pid.save_data(current_steer, steer_action, self.steer_pid.SetPoint)
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)

        return action


class DumpPid:
    def __init__(self, des):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.00055, I=0.00015, D=0.00001, saturation=0.8)
        # self.lift_pid.SetPoint = 220.
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.0025, I=0, D=0.0005, saturation=0.5)
        # self.pitch_pid.SetPoint = 250.
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # des = [x_blade, y_blade, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

    def step(self, obs):
        current_lift = obs['lift']
        current_pitch = obs['pitch']

        # pid update
        lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)

        # do action
        action = np.array([0, 0, pitch_action, lift_action])

        # save data
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)

        return action


class DriveBackAndLowerBladePid:
    def __init__(self, des):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # speed PID
        self.speed_pid = PID(P=1.5, I=0.1, D=0.1, saturation=0.8)
        self.speed_pid.setSampleTime(self.TIME_STEP)

        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.0005, I=0.00015, D=0.00001, saturation=0.8)
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.003, I=0, D=0.0005, saturation=0.5)
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # des = [x_blade, y_blade, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

    def step(self, obs, des, steer=True):
        current_lift = obs['lift']
        current_pitch = obs['pitch']

        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = obs['x_vehicle']
        self.speed_pid.SetPoint = x_des

        # pid update
        lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)

        # if got to des pos, stop driving but continue blade movement
        if obs['x_vehicle'] <= x_des:
            speed_action = 0
        else:
            speed_action = self.speed_pid.update(current_speed)

        # do action
        action = np.array([0, speed_action, pitch_action, lift_action])

        # save data
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)

        return action


class DriveBackPid:
    def __init__(self):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # speed PID
        self.speed_pid = PID(P=1.5, I=0.1, D=0.1, saturation=0.8)
        self.speed_pid.setSampleTime(self.TIME_STEP)

    def step(self, obs, des, steer=True):
        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = obs['x_vehicle']
        self.speed_pid.SetPoint = x_des

        speed_action = self.speed_pid.update(current_speed)

        # do action
        action = np.array([0, speed_action, 0, 0])

        # save data
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)

        return action


class LoadPid:
    def __init__(self):
        self.TIME_STEP = 10e-6  # 10 mili

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.0004, I=0.00015, D=0.00001, saturation=0.8)
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.006, I=0, D=0.0002, saturation=0.5)
        self.pitch_pid.setSampleTime(self.TIME_STEP)

        # speed PID
        self.speed_pid = PID(P=1, I=0.1, D=0.1, saturation=0.6)
        self.speed_pid.setSampleTime(self.TIME_STEP)

    def step(self, obs, des, x_pile):
        current_lift = obs['lift']
        current_pitch = obs['pitch']

        # des = [x_blade, y_blade, lift, pitch]
        self.lift_pid.SetPoint = des[2]  # desired lift
        self.pitch_pid.SetPoint = des[3]  # desired pitch

        # speed error: x_des - x_current
        x_des, y_des = des[0], des[1]
        current_speed = obs['x_blade']
        self.speed_pid.SetPoint = x_des

        # pid update
        lift_action = self.lift_pid.update(current_lift)
        pitch_action = self.pitch_pid.update(current_pitch)
        speed_action = self.speed_pid.update(current_speed)

        # do action
        # if far away from pile, only drive, don't move blade
        # if obs['x_blade'] < x_pile - 0.5:
        #     action = np.array([0, speed_action, 0, 0])
        # else:
        action = np.array([0, speed_action, pitch_action, lift_action])

        # save data
        self.lift_pid.save_data(current_lift, lift_action, self.lift_pid.SetPoint)
        self.pitch_pid.save_data(current_pitch, pitch_action, self.pitch_pid.SetPoint)
        self.speed_pid.save_data(current_speed, speed_action, self.speed_pid.SetPoint)

        return action


class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, saturation=1.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.saturation = saturation

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self._state = []
        self._action = []
        self._setPoint = []

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.output = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if delta_time >= self.sample_time:
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            # if self.ITerm < -self.windup_guard:
            #     self.ITerm = -self.windup_guard
            # elif self.ITerm > self.windup_guard:
            #     self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

            # saturate output between [-saturation, saturation]t
            output = np.clip(output, -self.saturation, self.saturation)

            return output

        else:
            return 0

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sample time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

    def save_data(self, state, action, setPoint):
        self._state.append(state)
        self._action.append(action)
        self._setPoint.append(setPoint)

    def save_plot(self, fileName, stateName):
        # init plot
        length = len(self._state)
        x = np.linspace(0, length, length)
        fig, (ax_state, ax_action) = plt.subplots(2)
        ax_state.set_title(stateName)
        ax_action.set_title(stateName + ' action')

        # plot data
        ax_state.plot(x, self._state, color='blue')
        ax_action.plot(x, self._action, color='blue')
        # plot set points
        ax_state.plot(x, self._setPoint, color='red')

        # save
        fig.savefig('/home/iaiai/git/SmartLoader/LLC/plots/' + fileName)
        print('figure saved!')
