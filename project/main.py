import configs
import gym
import time
from environment import make_env
from OpenGL import GLU

def train(env, agent, max_ep, steps_per_ep, model_path, train_from_model = False):
    if train_from_model:
        agent.restore(model_path)
    
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
    
    agent.save(model_path)
            

def replay(env, agent, model_path):
    agent.restore(model_path)

    while True:
        state = env.reset()
        for _ in range(200):
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
    parser.add_argument("-r", "--replay", help="replay model", action="store_true")
    args = parser.parse_args()

    if args.env_name:
        GAME_NAME = args.env_name
    
    REPLAY_ONLY = args.replay

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
        train(env=env, agent=agent, max_ep=500, steps_per_ep=200, model_path=model_file, train_from_model=train_from_model)
    
    replay(env=env, agent=agent, model_path=model_file)
    