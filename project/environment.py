import gym
import roboschool


def make_env(env_name, seed=-1):

    if env_name.startswith("Roboschool"):
        env = gym.make(env_name)
    else:
        print('No env found, starting Robo-Pendulum')
        env = gym.make('RoboschoolInvertedPendulumSwingup-v1')

    if (seed >= 0):
        env.seed(seed)
    '''
    print("environment details")
    print("env.action_space", env.action_space)
    print("high, low", env.action_space.high, env.action_space.low)
    print("environment details")
    print("env.observation_space", env.observation_space)
    print("high, low", env.observation_space.high, env.observation_space.low)
    assert False
    '''
    return env
