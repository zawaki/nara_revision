from gym.envs.registration import register

register(
    id='network-packing-v0',
    entry_point='network_packing_test.envs:PackingEnv',
)