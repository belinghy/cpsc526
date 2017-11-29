import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam
import keras.backend as K


PPO_LR_A = 0.0001
PPO_LR_C = 0.0002
PPO_A_UPDATE_STEPS = 10
PPO_C_UPDATE_STEPS = 10
PPO_BATCH_SIZE = 32
PPO_MEMORY_CAPACITY = PPO_BATCH_SIZE
PPO_GAMMA = 0.9

class PPO_TF:

    def __init__(self, action_dim, state_dim, action_bound):
        # [ state, action, reward, next_state, done ]
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = np.array(action_bound)
        self.memory = np.zeros((PPO_MEMORY_CAPACITY, state_dim * 2 + action_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
        self.method = [
            dict(name='kl_pen', kl_target=0.01, lamb=0.5),
            dict(name='clip', epsilon=0.2)
        ][1]

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.state, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advan = self.discounted_reward - self.v
            self.closs = tf.reduce_mean(tf.square(self.advan))
            self.ctrain_op = tf.train.AdamOptimizer(PPO_LR_C).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.action = tf.placeholder(tf.float32, [None, action_dim], 'action')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.action) / oldpi.prob(self.action)
                surr = ratio * self.advantage
            if self.method['name'] == 'kl_pen':
                self.lamb = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.lamb * kl))
            elif self.method['name'] == 'clip':
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-self.method['epsilon'], 1.+self.method['epsilon'])*self.advantage))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(PPO_LR_A).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advan, {self.state: s, self.discounted_reward: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.method['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.state: s, self.action: a, self.advantage: adv, self.lamb: self.method['lamb']})
                if kl > 4*self.method['kl_target']:  # this in in google's paper
                    break
            if kl < self.method['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.method['lamb'] /= 2
            elif kl > self.method['kl_target'] * 1.5:
                self.method['lamb'] *= 2
            self.method['lamb'] = np.clip(self.method['lamb'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        elif self.method['name'] == 'clip':   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.state: s, self.action: a, self.advantage: adv}) for _ in range(PPO_A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.state: s, self.discounted_reward: r}) for _ in range(PPO_C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.action_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.action_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.state: s})[0]
        return np.clip(a, -self.action_bound, self.action_bound)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.state: s})[0, 0]

    def store_transition(self, s, a, r, s_, d):
        transition = np.hstack((s, a, [r], s_, [d])) # reward was a scalar
        index = self.pointer % PPO_MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


    def train(self, env, max_ep, steps_per_ep, model_path, train_from_model=False, show_plot=False):
        if train_from_model:
            self.restore(model_path)

        if show_plot:
            import pyqtgraph as pg
            rewardplot = pg.plot(title='Episode Reward Graph')

        ep_reward_list = np.array([])
        for episode in range(max_ep):
            state = env.reset()
            ep_reward = 0.0
            for step in range(steps_per_ep):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                ep_reward = ep_reward + reward
                
                if (self.pointer + 1) % PPO_BATCH_SIZE == 0 or step == steps_per_ep - 1:
                    v_s_ = self.get_v(next_state)
                    buffer_r = self.memory[:, -self.state_dim-2 : -self.state_dim-1]
                    discounted_r = []

                    # reverse, calculate discount
                    for r in buffer_r[::-1]:
                        v_s_ = r + PPO_GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs = self.memory[:, :self.state_dim]
                    ba = self.memory[:, self.state_dim : self.state_dim + self.action_dim]
                    br = np.array(discounted_r)
                    self.update(bs, ba, br)

                    self.pointer = 0

                state = next_state
                if done or step == steps_per_ep-1:
                    print ('Episode: {} | Step: {} | Done?: {} | Reward: {}'.format(episode, step, 'Yes' if done else 'No', ep_reward))
                    break

            if show_plot:
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                rewardplot.plot(ep_reward_list, pen='y')
                pg.QtGui.QApplication.processEvents()

        self.save(model_path)

    def replay(self, env, model_path, sim_length=1000):
        self.restore(model_path)

        while True:
            state = env.reset()
            done = False
            for _ in range(sim_length):
                env.render()
                action = self.choose_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
                if done:
                    break



# ======================== DDPG_TF Hyper-parameters ===========================#
DDPG_LR_A = 0.0001       # actor learning rate
DDPG_LR_C = 0.0001       # critic learning rate
DDPG_TAU = 0.01         # soft replacement
DDPG_GAMMA = 0.95       # discount rate
DDPG_MEMORY_CAPACITY = 512
DDPG_BATCH_SIZE = 256

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
            h2 = tf.layers.dense(net, 100, activation=tf.nn.relu, name='layer2', trainable=trainable)
            # tanh guarantees we are between [-1, 1]
            a = tf.layers.dense(h2, self.action_dim, activation=tf.nn.tanh, name='action_layer', trainable=trainable)
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

    def train(self, env, max_ep, steps_per_ep, model_path, train_from_model=False, show_plot=False):
        if train_from_model:
            self.restore(model_path)

        if show_plot:
            import pyqtgraph as pg
            rewardplot = pg.plot(title='Episode Reward Graph')

        ep_reward_list = np.array([])
        for episode in range(max_ep):
            state = env.reset()
            ep_reward = 0.0
            for step in range(steps_per_ep):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                ep_reward = ep_reward + reward
                if self.memory_full:
                    # start to learn once memory is full
                    self.learn()

                state = next_state
                if done or step == steps_per_ep-1:
                    print ('Episode: {} | Step: {} | Done?: {} | Reward: {}'.format(episode, step, 'Yes' if done else 'No', ep_reward))
                    break

            if show_plot:
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                rewardplot.plot(ep_reward_list, pen='y')
                pg.QtGui.QApplication.processEvents()

        self.save(model_path)

    def replay(self, env, model_path, sim_length=1000):
        self.restore(model_path)

        while True:
            state = env.reset()
            done = False
            for _ in range(sim_length):
                env.render()
                action = self.choose_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
                if done:
                    break

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


    def train(self, env, max_ep, steps_per_ep, model_path, train_from_model=False, show_plot=False):
        if train_from_model:
            self.restore(model_path)

        if show_plot:
            import pyqtgraph as pg
            rewardplot = pg.plot(title='Episode Reward Graph')

        ep_reward_list = np.array([])
        for episode in range(max_ep):
            state = env.reset()
            ep_reward = 0.0
            for step in range(steps_per_ep):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                ep_reward = ep_reward + reward
                if self.memory_full:
                    # start to learn once memory is full
                    self.learn()

                state = next_state
                if done or step == steps_per_ep-1:
                    print ('Episode: {} | Step: {} | Done?: {} | Reward: {}'.format(episode, step, 'Yes' if done else 'No', ep_reward))
                    break

            if show_plot:
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                rewardplot.plot(ep_reward_list, pen='y')
                pg.QtGui.QApplication.processEvents()

        self.save(model_path)


    def replay(self, env, model_path, sim_length=1000):
        self.restore(model_path)

        while True:
            state = env.reset()
            done = False
            for _ in range(sim_length):
                env.render()
                action = self.choose_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
                if done:
                    break