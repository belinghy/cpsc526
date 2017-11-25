import gym
import numpy as np
import random
import roboschool
# Fast collection type for append() and pop() on both ends, O(n) near the middle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# Need this import to get rid of OpenGL error, when running roboschool
from OpenGL import GLU


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Using Keras
        model = Sequential()
        # hidden layer 1
        layer1_units = 24
        model.add(Dense(layer1_units, input_dim=self.state_size, activation='relu'))
        # hidden layer 2
        layer2_units = 24
        model.add(Dense(layer2_units, activation='relu'))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration stage
            # FIXME: only works if action space is Discrete?  Only returns one integer
            # return np.random.randint(self.action_size)
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # FIXME: Don't think this is how minibatch works...
        # FIXME: Should be shuffle self.memory, create minibatches (pop), and fit on each batch
        # minibatch = np.random.choice(self.memory, batch_size, replace=False)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            # TODO: What's happening here?
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    
if __name__ == '__main__':
    # GAME_NAME = 'RoboschoolInvertedPendulumSwingup-v1'
    GAME_NAME = 'CartPole-v1'
    env = gym.make(GAME_NAME)

    # Dimensions of observation space
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    EPISODES = 1000
    for episode in range(EPISODES):
        state = env.reset()
        # Turn into column vector
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            # Save for off line training
            agent.remember(state, action ,reward, next_state, done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {}, epsilon: {:.2}'.format(episode, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            # Actual training
            agent.replay(batch_size)
        
        if episode % 10 == 0:
            agent.save('./saved_models/cartpole-dqn.h5')