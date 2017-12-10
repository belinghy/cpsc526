import cloudpickle
import gym, roboschool
import numpy as np
import tensorflow as tf
from OpenGL import GLU

from baselines.ppo2.ppo2 import Model
from baselines.ppo2.policies import MlpPolicy


policy = MlpPolicy
nsteps=2048
nminibatches=32
lam=0.95
gamma=0.99
noptepochs=10
log_interval=1
ent_coef=0.0
lr=3e-4
cliprange=0.2
save_interval=50
total_timesteps=1e6
max_grad_norm=0.5
vf_coef=0.5

env = gym.make('RoboschoolReacher-v1')
nenvs = 1
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * nsteps
nbatch_train = nbatch // nminibatches

config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
tf.Session(config=config).__enter__()

model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)


model.load('/home/belinghy/Desktop/UBC/git_cpsc526/openai/saved_models/ppo2_reacher/checkpoints/00200')

while True:
    obs = env.reset()
    obs = obs[np.newaxis, :]
    state = model.initial_state
    done = False
    for _ in range(200):
        env.render()
        action, _, state, _ = model.step(obs, state, done)
        obs, _, done, _ = env.step(action[0])
        obs = obs[np.newaxis, :]
        if done:
            break