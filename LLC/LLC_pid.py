import os
import time
import numpy as np
import math
from matplotlib import pyplot as plt


class LLC:
    def __init__(self):
        self._output_folder = os.getcwd()

        self.lift = []
        self.pitch = []

        self.TIME_STEP = 0.05 # match env time step

        # Define PIDs
        # armHeight = lift = [down:145 - up:265]
        self.lift_pid = PID(P=0.002, I=0.0001, D=0.00001, saturation=True)
        self.lift_pid.SetPoint = 200.
        self.lift_pid.setSampleTime(self.TIME_STEP)

        # armShortHeight = pitch = [up:70 - down:265]
        self.pitch_pid = PID(P=-0.008, I=0, D=0.001, saturation=True)
        self.pitch_pid.SetPoint = 150.
        self.pitch_pid.setSampleTime(self.TIME_STEP)


    def step(self, obs):

        heat_map = obs[0]
        current_lift = obs[1]
        current_pitch = obs[2]

        # pid update
        lift_output = self.lift_pid.update(current_lift)
        pitch_output = self.pitch_pid.update(current_pitch)

        # do action (both actions together)
        action = np.array([pitch_output, lift_output])
        print('lift action = ', lift_output, 'pitch action = ', pitch_output)

        # save data
        self.lift.append(current_lift)
        self.pitch.append(current_pitch)

        return action


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



class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0, saturation=False, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.saturation = saturation

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

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

            if self.ITerm < -self.windup_guard:
                self.ITerm = -self.windup_guard
            elif self.ITerm > self.windup_guard:
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

            if self.saturation:
                # saturate output between [-1,1]
                if output > 1.0:
                    output = 1.0
                elif output < -1.0:
                    output = -1.0

            return output

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sample time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time