import configs
import gym
import numpy as np
import time
from environment import make_env
from OpenGL import GLU

def train(env, agent, max_ep, steps_per_ep, model_path, train_from_model=False, show_plot=False):
    if train_from_model:
        agent.restore(model_path)

    if show_plot:
        import pyqtgraph as pg
        rewardplot = pg.plot(title='Episode Reward Graph')

    ep_reward_list = np.array([])
    for episode in range(max_ep):
        state = env.reset()
        ep_reward = 0.0
        for step in range(steps_per_ep):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            ep_reward = ep_reward + reward
            if agent.memory_full:
                # start to learn once memory is full
                agent.learn()

            state = next_state
            if done or step == steps_per_ep-1:
                print ('Episode: {} | Step: {} | Done?: {} | Reward: {}'.format(episode, step, 'Yes' if done else 'No', ep_reward))
                break

        if show_plot:
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            rewardplot.plot(ep_reward_list, pen='y')
            pg.QtGui.QApplication.processEvents()

    agent.save(model_path)


def replay(env, agent, model_path, sim_length=1000):
    agent.restore(model_path)

    while True:
        state = env.reset()
        done = False
        for _ in range(sim_length):
            env.render()
            action = agent.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', metavar='ENV', type=str, help="e.g. robo_pendulum, robo_reacher, or robo_walker")
    parser.add_argument('agent', metavar='AGENT', type=str, help="e.g. DDPG_TF, AC_Keras")
    parser.add_argument('-m', '--model_path', type=str, help="continue training from model")
    parser.add_argument("--replay", help="replay model", action="store_true")
    parser.add_argument("--visual", help="display reward plot", action="store_true")
    args = parser.parse_args()

    if args.env_name:
        GAME_NAME = args.env_name

    REPLAY_ONLY = args.replay
    VISUAL_ON = args.visual

    if REPLAY_ONLY and not args.model_path:
        print('Must specifiy a model to to play')
        exit(-1)

    game = configs.games[GAME_NAME]
    env = gym.make(game.env_name)

    state_dim = game.input_size
    action_dim = game.output_size
    action_bound = game.action_bound

    # Delay import
    if args.agent == 'DDPG_TF':
        from agents import DDPG_TF
        agent = DDPG_TF(action_dim, state_dim, action_bound)
    elif args.agent == 'AC_Keras':
        from agents import AC_Keras
        agent = AC_Keras(action_dim, state_dim, action_bound)
    else:
        print ('Unknown agent')
        exit(-1)


    if args.model_path:
        model_file = args.model_path
        train_from_model = True
    else:
        model_file = 'saved_models/{}_{}_{}'.format(GAME_NAME, args.agent, int(time.time()))
        train_from_model = False


    if not REPLAY_ONLY:
        train(env=env, agent=agent, max_ep=250, steps_per_ep=500, model_path=model_file, train_from_model=train_from_model, show_plot=VISUAL_ON)

    replay(env=env, agent=agent, model_path=model_file)
