import random
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm

sys.path.append("../")
from configs import B_FEATURES, B_LABELS, ROOT, MAX_VAL, MIN_VAL
from utils import *

tf.disable_eager_execution()

# Configurations
model_dir = ROOT + "/models/checkpoints/Model_B/"
model_path = model_dir + "Model_B.ckpt"
log_dir = ROOT + "/models/logs/"
log_path = log_dir + "Model_B_loss.csv"
data_dir = ROOT + "/data/data_process/Model_B/"
data_path_train = data_dir + "Model_B_train.csv"
data_path_test = data_dir + "Model_B_test.csv"
data_path_valid = data_dir + "Model_B_valid.csv"

class Model_B:
    def __init__(self, tl=False, epoch=10, BATCH=256, learning_rate=0.001, drop_rate=0.3, is_train=False):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.BATCH = BATCH
        self.drop_rate = drop_rate
        self.input_size = len(B_FEATURES)
        self.output_size = len(B_LABELS)
        self.data_train = []
        self.data_valid = []
        self.max_min = {}
        self.max_min["input_max"] = [MAX_VAL[key] for key in B_FEATURES]
        self.max_min["input_min"] = [MIN_VAL[key] for key in B_FEATURES]
        tf.reset_default_graph()
        self.create_network()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if not is_train and checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
            self.saver.restore(self.sess, model_path)
            print_color("Model B load successfully.","green")
        else:
            print_color("Model B is not loaded.","red")

    def use_model(self, input_arr):
        input_matrix = np.asmatrix(self.normalize_for_input(input_arr))
        with tf.device("/gpu:0"):
            output_arr = self.label_conv.eval(feed_dict={self.state: input_matrix},session=self.sess)[0]
        return output_arr

    def close_session(self):
        self.sess.close()

    def create_network(self):
        hidden_size_1 = 32
        hidden_size_2 = 64
        hidden_size_3 = 32

        # Input
        self.state = tf.placeholder(tf.float32, shape=[None, self.input_size])
        # Label
        self.label = tf.placeholder(tf.float32, shape=[None, self.output_size])

        # Hidden layers
        h1 = tf.layers.dense(self.state, hidden_size_1, activation = tf.nn.relu)
        h1 = tf.layers.dropout(h1, rate = self.drop_rate)

        h2 = tf.layers.dense(h1, hidden_size_2, activation = tf.nn.relu)
        h2 = tf.layers.dropout(h2, rate = self.drop_rate)

        h3 = tf.layers.dense(h2, hidden_size_3, activation = tf.nn.relu)
        h3 = tf.layers.dropout(h3, rate = self.drop_rate)

        self.label_conv = tf.layers.dense(h3, self.output_size)
        self.loss = tf.reduce_mean(tf.square(self.label - self.label_conv)) 
        self.error = tf.abs(self.label - self.label_conv)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def weight_variable(self, shape):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.01))

    def bias_variable(self, shape):
        return tf.Variable(tf.zeros(shape=shape) + 0.1)

    def normalize_for_input(self, input_arr):
        for i in range(len(input_arr)):
            input_arr[i] = (float(input_arr[i]) - self.max_min["input_min"][i]) / (
                        self.max_min["input_max"][i] - self.max_min["input_min"][i])
        return input_arr

    def handleData(self, train_state, train_label, valid_state=None, valid_label=None):
        if not (train_state is None or train_label is None):
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
        for c in tqdm(range(self.iterations)):
            state_batch, label_batch = self.get_batch(self.data_train, c, self.BATCH)
            self.train_step.run(feed_dict={self.state: state_batch, self.label: label_batch})
            if c % 100 == 0:
                loss_train = self.sess.run([self.loss], feed_dict={self.state: state_batch, self.label: label_batch})
                #state_batch_valid, y_batch_valid = self.get_batch_random(self.data_valid, self.BATCH)
                #loss_valid = self.sess.run([self.loss],
                #                           feed_dict={self.state: state_batch_valid, self.label: y_batch_valid})
                #loss_arr = [c, loss_train[0], loss_valid[0]]
                loss_arr = [c, loss_train[0]]
                df_loss.loc[df_loss.shape[0]] = loss_arr
                print(loss_arr)
                self.saver.save(self.sess, model_path)
        df_loss.to_csv(log_path, index=False)


def train_model(tl, is_train=True):
    model = Model_B(tl=tl, is_train=is_train)

    pkl_path_train=data_path_train.replace(".csv",".pkl")
    if os.path.exists(pkl_path_train):
        with open(pkl_path_train,"rb") as f:
            data_train=pickle.load(f)
    else:
        data_train = pd.read_csv(data_path_train)

    state_train = data_train.loc[:, B_FEATURES].values
    label_train = data_train.loc[:, B_LABELS].values

    model.handleData(state_train, label_train)
    model.trainNetwork()


def test_model():
    model = Model_B(is_train=False)

    pkl_path_test=data_path_test.replace(".csv",".pkl")
    if os.path.exists(pkl_path_test):
        with open(pkl_path_test,"rb") as f:
            data_test=pickle.load(f)
    else:
        data_test = pd.read_csv(data_path_test)
    state_test = data_test.loc[:, B_FEATURES].values
    label_test = data_test.loc[:, B_LABELS].values
    output=model.sess.run(model.label_conv, feed_dict={model.state: state_test, model.label: label_test})
    error = model.sess.run(model.error, feed_dict={model.state: state_test, model.label: label_test})
    error = pd.DataFrame(data=output,columns=B_LABELS)
    print("Error: {}".format(error.mean()))

    num_correct=0
    for i in range(len(label_test)):
        if (label_test[i]>=1 and output[i]>=1) or (label_test[i]<1 and output[i]<1):
            num_correct+=1
    print("Accuracy: {}".format(num_correct/len(label_test)))
    

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-t","--tl", help="Enabling transfer learning if the --tl parameter is provided", action="store_true")
    args=parser.parse_args()

    train_model(args.tl)
    test_model()
