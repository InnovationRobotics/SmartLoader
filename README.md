# SmartLoader
Welcome to SmartLoader!
Innovation project February-May 2020.

All code needed to train and implement imitation and reinforcement learning on simulation and real-life loader.

Includes:
  - gym_SmartLoader - gym environment for:
    - bobcat simulation (SmartLoader_env)
    - Benny's heat map bobcat simulation (HeatMap_env)
    - Algoryx simulation (push_algoryx)
  - Sim_Agents: algorithms for RL (train_agent) and BC agents for simulation
  - SmartLoaderIRL - Environment for real-life scenario
  - Real-life episode runners: episode_runner_push_mission, episode_runner_lift_mission
  - recordings_manipulations - save data from real-life joystick recordings
  - Imitation learning from recordings - Pos_EstimatorAgent_CNN_Keras
  - Benny’s heatmap - HeatMapGen
  - LLC - PID for controlling all loader’s actions (or bobcat in simulation): steer, speed, lift, pitch. Different PIDs for different missions: push, dump, drive back, load.
  - Algorix training
