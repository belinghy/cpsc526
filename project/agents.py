import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam
import keras.backend as K

# ======================== DDPG_TF Hyper-parameters ===========================#
DDPG_LR_A = 0.001       # actor learning rate
DDPG_LR_C = 0.001       # critic learning rate
DDPG_TAU = 0.01         # soft replacement
DDPG_GAMMA = 0.95       # discount rate
DDPG_MEMORY_CAPACITY = 10_000
DDPG_BATCH_SIZE = 32

class DDPG_TF:
    def __init__(self, action_dim, state_dim, action_bound):
        # Why this shape? [ state, action, reward, next_state, done ]
        self.memory = np.zeros((DDPG_MEMORY_CAPACITY, state_dim * 2 + action_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = np.array(action_bound)
        self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
        self.next_state = tf.placeholder(tf.float32, [None, state_dim], 'next_state')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        with tf.variable_scope('Actor'):
            # Variable name doesn't matter
            self.action = self._build_actor(self.state, scope='eval', trainable=True)
            action_ = self._build_actor(self.next_state, scope='target', trainable=False)
        
        with tf.variable_scope('Critic'):
            # assign self.action = action in memory when calculating q for td_error
            # otherwise the self.action is from Actor when updating Actor
            q = self._build_critic(self.state, self.action, scope='eval', trainable=True)
            q_ = self._build_critic(self.next_state, action_, scope='target', trainable=False)

        # network parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1-DDPG_TAU)*ta + DDPG_TAU*ea), tf.assign(tc, (1-DDPG_TAU)*tc + DDPG_TAU*ec)]
                                for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        
        # define loss functions
        q_target = self.reward + DDPG_GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.critic_train = tf.train.AdamOptimizer(learning_rate=DDPG_LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(q) # maximize for q
        self.actor_train = tf.train.AdamOptimizer(learning_rate=DDPG_LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.action, { self.state: s[None, :] })[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        # FIXME: Why is replacement ok?  Minibatch wrong?  Soft-replace?
        indices = np.random.choice(DDPG_MEMORY_CAPACITY, size=DDPG_BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        bs = batch_trans[:, :self.state_dim]
        ba = batch_trans[:, self.state_dim : self.state_dim + self.action_dim]
        br = batch_trans[:, -self.state_dim-2 : -self.state_dim-1]
        bs_ = batch_trans[:, -self.state_dim-1:-1]

        self.sess.run(self.actor_train, { self.state: bs })
        self.sess.run(self.critic_train, { self.state: bs, self.action: ba, self.reward: br, self.next_state: bs_ })

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, a, [r], s_, [d])) # reward was a scalar
        index = self.pointer % DDPG_MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > DDPG_MEMORY_CAPACITY: # indicator for learning
            self.memory_full = True

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # one hidden layer 100 units
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='layer1', trainable=trainable)
            # tanh guarantees we are between [-1, 1]
            a = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='action_layer', trainable=trainable)
            return tf.multiply(a, self.action_bound, name='scaled_a')

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [ self.state_dim, n_l1 ], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [ self.action_dim, n_l1 ], trainable=trainable)
            bias1 = tf.get_variable('b1', [ 1, n_l1 ], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + bias1)
            return tf.layers.dense(net, 1, trainable=trainable) # Q(s, a)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

# ======================== ActorCritic-Keras Hyper-parameters ===========================#
AC_LR_A = 0.001       # actor learning rate
AC_LR_C = 0.001       # critic learning rate
AC_TAU = 0.01         # soft replacement
AC_GAMMA = 0.95       # discount rate
AC_EPSILON = 1.       # Exploration threshold
AC_EPSILON_DECAY = 0.995
AC_RANDOM_ACTION_SIGMA = 0.1
AC_MEMORY_CAPACITY = 10_000
AC_BATCH_SIZE = 1

class AC_Keras:
    def __init__(self, action_dim, state_dim, action_bound, epsilon=1.):
        # Why this shape? [ state, action, reward, next_state, done]
        self.memory = np.zeros((AC_MEMORY_CAPACITY, state_dim * 2 + action_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = np.array(action_bound)
        self.epsilon = epsilon



        # ============= Actor Model ============ #
        self.actor_state_input, self.actor_model = self._build_actor()
        _, self.target_actor_model = self._build_actor()

        # will feed de/dC
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_dim])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA
        grads = zip(self.actor_grads, actor_model_weights)

        self.optimize = tf.train.AdamOptimizer(AC_LR_A).apply_gradients(grads)


        # ============ Critic Model ============ #
        self.critic_state_input, self.critic_action_input, self.critic_model = self._build_critic()
        _, _, self.target_critic_model = self._build_critic()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) # This is de/dC


        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, state):
        self.epsilon = self.epsilon * AC_EPSILON_DECAY
        action = self.actor_model.predict(state[None, :])
        action = action[0]
        if np.random.random() < self.epsilon:
            uncapped = action + np.random.normal(0, AC_RANDOM_ACTION_SIGMA, action.shape)
            return np.clip(uncapped, -self.action_bound, self.action_bound)
        return action

    def learn(self):
        # FIXME: Why is replacement ok?  Minibatch wrong?  Soft-replace?
        indices = np.random.choice(AC_MEMORY_CAPACITY, size=AC_BATCH_SIZE, replace=False)
        # [ state, action, reward, next_state, done ]
        samples = self.memory[indices, :]

        self._train_critic(samples)
        self._train_actor(samples)


    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done])) # reward was a scalar, and python converts bool to float
        index = self.pointer % AC_MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > AC_MEMORY_CAPACITY: # indicator for learning
            self.memory_full = True


    def _build_actor(self):
        state = Input(shape=(self.state_dim, ))
        hidden1 = Dense(100, activation='relu')(state)
        action = Dense(self.action_dim, activation='tanh')(hidden1)
        
        model = Model(input=state, output=action)
        adam = Adam(lr=AC_LR_A)
        model.compile(loss='mse', optimizer=adam)
        return state, model


    def _build_critic(self):
        state = Input(shape=(self.state_dim, ))
        state_hidden1 = Dense(24, activation='relu')(state)
        state_hidden2 = Dense(48)(state_hidden1)

        action = Input(shape=(self.action_dim, ))
        action_hidden1 = Dense(48)(action)

        merged = Add()([state_hidden2, action_hidden1])
        merged_hidden1 = Dense(24, activation='relu')(merged)
        # Reward is always an interger, so output size 1
        output = Dense(1, activation='relu')(merged_hidden1)

        model = Model(input=[state, action], output=output)
        adam = Adam(lr=AC_LR_C)
        model.compile(loss='mse', optimizer=adam)
        return state, action, model

    
    def _train_actor(self, samples):
        for sample in samples:
            state = sample[:self.state_dim]
            action  = sample[self.state_dim : self.state_dim + self.action_dim]
            reward  = sample[-self.state_dim-2 : -self.state_dim-1]
            nstate = sample[-self.state_dim-1:-1]
            done  = sample[-1:]
        
            state = state[None, :]
            nstate = nstate[None, :]

            predicted_action = self.actor_model.predict(state)

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grad: grads
            })

    
    def _train_critic(self, samples):
        for sample in samples:
            state = sample[:self.state_dim]
            action  = sample[self.state_dim : self.state_dim + self.action_dim]
            reward  = sample[-self.state_dim-2 : -self.state_dim-1]
            nstate = sample[-self.state_dim-1:-1]
            done  = sample[-1:]

            state = state[None, :]
            nstate = nstate[None, :]

            if not done:
                target_action = self.target_actor_model.predict(nstate)
                future_reward = self.target_critic_model.predict([nstate, target_action])[0][0]
                reward += AC_GAMMA * future_reward
            
            self.critic_model.fit([state, action], reward, verbose=0)


    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)