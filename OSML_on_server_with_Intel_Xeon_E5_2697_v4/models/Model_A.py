import random
import sys
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from copy import deepcopy
import pickle
sys.path.append("../")
from configs import A_FEATURES, A_LABELS, ROOT
from utils import *

tf.disable_eager_execution()

# Configurations
model_dir = ROOT + "/models/checkpoints/Model_A/"
model_path = model_dir + "Model_A.ckpt"
log_dir = ROOT + "/models/logs/"
log_path = log_dir + "Model_A_loss.csv"
data_dir = ROOT + "/data/data_process/Model_A/"
data_path_train = data_dir + "Model_A_train.csv"
data_path_test = data_dir + "Model_A_test.csv"
data_path_valid = data_dir + "Model_A_valid.csv"
max_min_path = ROOT + "/data/data_process/max_min.txt"


class Model_A:
    def __init__(self, epoch=10, BATCH=256, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.BATCH = BATCH
        self.input_size = len(A_FEATURES)
        self.output_size = len(A_LABELS)
        self.data_train = []
        self.data_valid = []
        with open(max_min_path, "r") as f:
            max_min = eval(f.readline())
            self.max_min = {}
            self.max_min["input_max"] = [max_min["max"][feature] for feature in A_FEATURES]
            self.max_min["input_min"] = [max_min["min"][feature] for feature in A_FEATURES]
        tf.reset_default_graph()
        self.create_network()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
            self.saver.restore(self.sess, model_path)
            print_color("==> Model A load successfully.","green")
        else:
            print_color("==> Fail loading Model A.","red")

    def use_model(self, input_arr):
        """
        Use Model-A to infer OAA, RCliff and OAA bandwidth.
        :param input_arr: Model-A features
        :return: Model-A labels
        """
        input_matrix = np.asmatrix(self.normalize_for_input(input_arr))
        with tf.device("/gpu:0"):
            output_arr = self.label_conv.eval(feed_dict={self.state: input_matrix}, session=self.sess)[0]
        return output_arr

    def close_session(self):
        self.sess.close()

    def create_network(self):
        hidden_size_1 = 40
        hidden_size_2 = 40
        hidden_size_3 = 40

        # Input - Hidden layer 1
        w_fc1 = self.weight_variable([self.input_size, hidden_size_1])
        b_fc1 = self.bias_variable([hidden_size_1])
        # Hidden layer 1 - Hidden layer 2
        w_fc2 = self.weight_variable([hidden_size_1, hidden_size_2])
        b_fc2 = self.bias_variable([hidden_size_2])
        # Hidden layer 2 - Hidden layer 3
        w_fc3 = self.weight_variable([hidden_size_2, hidden_size_3])
        b_fc3 = self.bias_variable([hidden_size_3])
        # Hidden layer 3 - Output
        w_fc4 = self.weight_variable([hidden_size_3, self.output_size])
        b_fc4 = self.bias_variable([self.output_size])

        # Input
        self.state = tf.placeholder("float", [None, self.input_size])

        # Hidden layers
        h_fc1 = tf.nn.relu(tf.matmul(self.state, w_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

        # Output
        self.label_conv = tf.matmul(h_fc3, w_fc4) + b_fc4

        # Loss function
        self.label = tf.placeholder(tf.float32, shape=[None, self.output_size])
        self.loss = tf.reduce_mean(tf.square(self.label - self.label_conv))
        self.mae = tf.reduce_mean(self.label - self.label_conv)
        self.error = tf.abs(self.label - self.label_conv)
        # Train step
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def weight_variable(self, shape):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.01))

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape=shape) + 0.1)

    def normalize_for_input(self, input_arr):
        input_arr=deepcopy(input_arr)
        for i in range(len(input_arr)):
            input_arr[i] = (float(input_arr[i]) - self.max_min["input_min"][i]) / (
                        self.max_min["input_max"][i] - self.max_min["input_min"][i])
        return input_arr

    def handleData(self, train_state, train_label, valid_state = None, valid_label = None):

        if not(train_state is None or train_label is None):
            for i in range(train_state.shape[0]):
                self.data_train.append((np.asmatrix(train_state[i]), np.asmatrix(train_label[i])))

        if not(valid_state is None or valid_label is None):
            for i in range(valid_state.shape[0]):
                self.data_valid.append((np.asmatrix(valid_state[i]), np.asmatrix(valid_label[i])))

    def get_batch(self, memory, c, size):
        minibatch = memory[c * size % len(memory):(c + 1) * size % len(memory)]
        state_batch = [d[0] for d in minibatch]
        label_batch = [d[1] for d in minibatch]
        state_batch = np.reshape(state_batch, (-1, self.input_size))
        label_batch = np.reshape(label_batch, (-1, self.output_size))
        return state_batch, label_batch

    def get_batch_random(self, memory, size):
        minibatch = random.sample(memory, size)
        state_batch = [d[0] for d in minibatch]
        label_batch = [d[1] for d in minibatch]
        state_batch = np.reshape(state_batch, (-1, self.input_size))
        label_batch = np.reshape(label_batch, (-1, self.output_size))
        return state_batch, label_batch

    def trainNetwork(self):
        df_loss = pd.DataFrame(columns=["step", "loss_train"])
        self.iterations = round(len(self.data_train) / self.BATCH * self.epoch)
        print("Train data has {} tuples, iterations: {}".format(len(self.data_train), self.iterations))
        with tf.device("/gpu:0"):
            for c in tqdm(range(self.iterations)):
                state_batch, label_batch = self.get_batch(self.data_train, c, self.BATCH)
                self.train_step.run(feed_dict={self.state: state_batch, self.label: label_batch}, session=self.sess)
                if c % 100 == 0:
                    loss_train = self.sess.run([self.loss], feed_dict={self.state: state_batch, self.label: label_batch})
                    loss_arr = [c,loss_train[0]]
                    df_loss.loc[df_loss.shape[0]] = loss_arr
                    print(loss_arr)
                    self.saver.save(self.sess, model_path)
        df_loss.to_csv(log_path, index=False)


def train_model():
    model_a = Model_A()
    data_train=pd.read_csv(data_path_train)
    state_train = data_train.loc[:, A_FEATURES].values
    label_train = data_train.loc[:, A_LABELS].values
    model_a.handleData(state_train, label_train)
    model_a.trainNetwork()


def test_model():
    model_a = Model_A()
    data_test = pd.read_csv(data_path_test)
    state_test = data_test.loc[:, A_FEATURES].values
    label_test = data_test.loc[:, A_LABELS].values
    output=model_a.sess.run(model_a.error, feed_dict={model_a.state: state_test, model_a.label: label_test})
    error = pd.DataFrame(data=output,columns=A_LABELS)
    print(error.mean())


if __name__ == "__main__":
    train_model()
    test_model()
