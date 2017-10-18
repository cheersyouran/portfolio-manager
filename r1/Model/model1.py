import tensorflow as tf

from r1 import config
from r1.Model.base import Base

TIME_SPACE = config.TIME_SPACE
STOCK_NUM = config.STOCK_NUM
HISTORY = config.HISTORY
ALL_HISTORY = config.ALL_HISTORY
PATH = config.PATH
MODEL1_PATH = config.MODEL1_PATH

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement

class Model1(Base):
    def __init__(self, load):
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.S = tf.placeholder(tf.float32, [None, STOCK_NUM, 3], 's')
        self.S_ = tf.placeholder(tf.float32, [None, STOCK_NUM, 3], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self.build_actor(self.S, scope='eval')
            self.a_ = self.build_actor(self.S_, scope='target')
        with tf.variable_scope('Critic'):
            self.q = self.build_critic(self.S, self.a, scope='eval')
            self.q_ = self.build_critic(self.S_, self.a_, scope='target')

        self.replace_params()
        self.init_loss_function()
        if(load==True):
            Base.restore_model(self.sess, MODEL1_PATH)

    def init_loss_function(self):
        q_target = self.R + GAMMA * self.q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())


    def replace_params(self):
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self, s, a, r, s_, iteration):
        # soft target replacement
        self.sess.run(self.soft_replace)
        self.sess.run(self.atrain, {self.S: s})
        self.sess.run(self.ctrain, {self.S: s, self.a: a, self.R: r, self.S_: s_})
        if(iteration%10000==0):
             Base.save_model(self.sess, MODEL1_PATH)

    def build_actor(self, s, scope):
        with tf.variable_scope(scope):
            # [1, TIME_SPACE, STOCK_NUM, 3]
            s_reshape = tf.reshape(s, [-1, TIME_SPACE, STOCK_NUM, 3])

            w_conv1 = Base.weight_varible([4, 1, 3, 2])
            b_conv1 = Base.bias_variable([2])
            # [1, 28, STOCK_NUM, 2]
            h_conv1 = tf.nn.relu(Base.conv2d(s_reshape, w_conv1) + b_conv1)

            cash_bais = Base.bias_variable([1, TIME_SPACE-3, 1, 2])
            # [1, 28, STOCK_NUM+1, 2]
            h_conv1 = tf.concat([h_conv1, cash_bais], 2)

            w_conv2 = Base.weight_varible([TIME_SPACE-3, 1, 2, 20])
            b_conv2 = Base.bias_variable([20])
            # [1, 1, STOCK_NUM+1, 20]
            h_conv2 = tf.nn.relu(Base.conv2d(h_conv1, w_conv2)+b_conv2)

            w_conv3 = Base.weight_varible([1, 1, 20, 1])
            b_conv3 = Base.bias_variable([1])
            # [1, 1, STOCK_NUM+1, 1]
            h_conv3 = tf.nn.relu(Base.conv2d(h_conv2, w_conv3)+b_conv3)

            a = tf.nn.softmax(h_conv3, dim=2)
            return a

    def build_critic(self, s, a, scope):
        with tf.variable_scope(scope):
            # [1, TIME_SPACE, STOCK_NUM, 3]
            s_reshape = tf.reshape(s, [-1, TIME_SPACE, STOCK_NUM, 3])

            W_conv1 = Base.weight_varible([4, 1, 3, 2])
            b_conv1 = Base.bias_variable([2])
            # [1, 28,STOCK_NUM,2]
            h_conv1 = tf.nn.relu(Base.conv2d(s_reshape, W_conv1)+b_conv1)

            cash_bais = Base.bias_variable([1, TIME_SPACE-3, 1, 2])
            # [1, 28, STOCK_NUM+1, 2]
            h_conv1 = tf.concat([h_conv1, cash_bais], 2)

            W_conv2 = Base.weight_varible([TIME_SPACE-3, 1, 2, 20])
            b_conv2 = Base.bias_variable([20])
            # [1, 1,STOCK_NUM+1, 20]
            h_conv2 = tf.nn.relu(Base.conv2d(h_conv1, W_conv2)+b_conv2)
            # [1, 1,STOCK_NUM+1, 21]
            h_conv2 = tf.concat([h_conv2, a], 3)

            W_conv3 = Base.weight_varible([1, STOCK_NUM + 1, 21, 1])
            b_conv3 = Base.bias_variable([1])
            # [1, 1, 1, 1]
            h_conv3 = tf.nn.relu(Base.conv2d(h_conv2, W_conv3)+b_conv3)

            q_sa = tf.reshape(h_conv3, [1,1])
            return q_sa

