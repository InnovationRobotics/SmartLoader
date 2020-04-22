from gym.envs.registration import register
# add all custom envs here

register(
    id='BaseEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:BaseEnv'
)

register(
    id='PickUpEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:PickUpEnv'
)

register(
    id='PutDownEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:PutDownEnv'
)

register(
    id='MoveWithStonesEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:MoveWithStonesEnv'
)

register(
    id='PushStonesEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:PushStonesEnv'
)
register(
    id='PushStonesHeatMapEnv-v0',
    entry_point='gym_SmartLoader.envs.SmartLoaderEnvs_dir:PushStonesHeatMapEnv'
)
