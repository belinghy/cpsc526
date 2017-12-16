import gym
import roboschool
from OpenGL import GLU
import numpy as np

env = gym.make('RoboschoolReacher-v1')
while True:
    s = env.reset()
    for t in range(550):
        env.render()
        if t < 2:
            action = np.array([1., 1.])
        elif t == 200:
            action = np.array([-.5, -.5])
        else:
            action = np.array([0., 0.])

        s = env.step(action)
        print (s[0])
