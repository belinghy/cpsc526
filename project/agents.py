import numpy as np
import tensorflow as tf

######### Hyper-parameters ########
DDPG_LR_A = 0.001       # actor learning rate
DDPG_LR_C = 0.001       # critic learning rate
DDPG_TAU = 0.01         # soft replacement
DDPG_GAMMA = 0.95       # discount rate
DDPG_MEMORY_CAPACITY = 10_000
DDPG_BATCH_SIZE = 32

class DDPG_TF:
    def __init__(self, action_dim, state_dim, action_bound):
        # Why this shape? [ state, action, reward, next_state ]
        self.memory = np.zeros((DDPG_MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.actor_replace_counter = 0
        self.critic_replace_counter = 0

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
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
        br = batch_trans[:, -self.state_dim-1 : -self.state_dim]
        bs_ = batch_trans[:, -self.state_dim:]

        self.sess.run(self.actor_train, { self.state: bs })
        self.sess.run(self.critic_train, { self.state: bs, self.action: ba, self.reward: br, self.next_state: bs_ })

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # reward was a scalar
        index = self.pointer % DDPG_MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > DDPG_MEMORY_CAPACITY: # indicator for learning
            self.memory_full = True

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # one hidden layer 100 units
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='layer1', trainable=trainable)
            a = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='action_layer', trainable=trainable)
            # FIXME: We are assuming action_bound >= 1
            return tf.multiply(a, self.action_bound, name='scaed_a')

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