# SmartLoader

Welcome to SmartLoader!
Innovation project February-May 2020.

All code needed to train and implement imitation and reinforcement learning on simulation and real-life loader.

Includes:
gym environment for bobcat simulation (gym_SmartLoader) and for real-life scenario (SmartLoaderIRL)
Train_agent - train RL lationagent
Real-life episode runners: episode_runner_push_mission, episode_runner_lift_mission
Save data from real-life joystick recordings - recordings_concat and recordings_manipulations
Imitation learning from recordings - Pos_EstimatorAgent_CNN_Keras
Benny’s heatmap - HeatMapGen
LLC - PID for controlling all loader’s actions (or bobcat in simulation): steer, speed, lift, pitch. Different PIDs for different missions: push, dump, drive back, load.
