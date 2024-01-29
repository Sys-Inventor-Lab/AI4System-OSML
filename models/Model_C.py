import sys

import numpy as np
import tensorflow.compat.v1 as tf
import csv
from tqdm import tqdm
import pandas as pd
sys.path.append("../")
from utils import *
from configs import *

model_dir = ROOT + "/models/Model_C/"
log_dir = ROOT + "/models/logs/"
export_filename=log_dir+"Model_C_data.csv"
reward_path = log_dir + "Model_C_reward.csv"
data_dir = ROOT + "/data/data_process/Model_C/"
data_path_train = data_dir + "Model_C_train.csv"
data_path_test = data_dir + "Model_C_test.csv"
data_path_valid = data_dir + "Model_C_valid.csv"
max_min_path = ROOT + "/data/data_process/max_min/max_min_Model_C.txt"
tf.disable_eager_execution()

class Model_C:
    def __init__(
            self,
            n_actions,
            n_features,
            name,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.95,
            replace_target_iter=100,
            memory_size=50,
            batch_size=5,
            e_greedy_increment=None,
            output_graph=False,
            load=True
    ):
        self.name = name
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        with tf.Graph().as_default() as locals()['G' + self.name]:
            self._build_net()
            self.saver = tf.train.Saver()
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [
                    tf.assign(t, e) for t, e in zip(t_params, e_params)
                ]
            self.sess = tf.Session(graph=locals()['G' + self.name])
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        with open(max_min_path, "r") as f:
            max_min = eval(f.readline())
            self.max_min = {}
            self.max_min["input_max"] = max_min["max"][:self.n_features]
            self.max_min["input_min"] = max_min["min"][:self.n_features]

        self.export_file=open(export_filename,"a")

        #self.sess.run(tf.local_variables_initializer())
        self.sess.run(init_op)
        if load:
            self.restore()
        self.cost_his = []

    def __del__(self):
        self.export_file.close()

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None,], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None,], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            e2 = tf.layers.dense(e1,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e2')
            e3 = tf.layers.dense(e2,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e3')
            self.q_eval = tf.layers.dense(e3,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t1')
            t2 = tf.layers.dense(t1,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t2')
            t3 = tf.layers.dense(t2,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t3')
            self.q_next = tf.layers.dense(t3,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_, save_to_file=False):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        if save_to_file:
            text_to_write=",".join([str(item) for item in list(transition)])
            self.export_file.write(text_to_write+"\n")

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def normalize_for_input(self, input_arr):
        for i in range(len(input_arr)):
            input_arr[i] = (input_arr[i] - self.max_min["input_min"][i]) / (
                        self.max_min["input_max"][i] - self.max_min["input_min"][i])
        return input_arr

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        #self.cost_his.append(cost)
        self.learn_step_counter += 1
        return cost


    def save(self):
        model_path=model_dir+"{}/{}".format(self.name,self.name)
        self.saver.save(self.sess,model_path,write_meta_graph=True)

    def restore(self):
        model_path=model_dir+"{}".format(self.name)
        path = tf.train.latest_checkpoint(model_path)
        if path is not None:
            self.saver.restore(self.sess, path)
            print_color("Model C load successfully.","green")
        else:
            print_color("Fail loading Model C.","red")

    def read_memory(self, transition):
        if len(transition.shape)==1:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
        elif len(transition.shape)==2:
            for i in range(len(transition)):
                index = self.memory_counter % self.memory_size
                self.memory[index, :]=transition[i]
                self.memory_counter+=1

def train_model():
    with tf.variable_scope('add'):
        model_c_add = Model_C(len(ACTION_SPACE_ADD), len(C_FEATURES["s"]), "model_c_add",output_graph=False, e_greedy_increment=True)
    with tf.variable_scope('sub'):
        model_c_sub = Model_C(len(ACTION_SPACE_SUB), len(C_FEATURES["s"]), "model_c_sub",output_graph=False, e_greedy_increment=True)    

    checkpoint=stored_data("Model_C.pkl")
    if checkpoint.empty():
        checkpoint.data={"add":{},"sub":{}}

    last_timestamp=time.time()
    paths=[]
    for root in ["single","multiple"]:
        data_root = ROOT+"/data/data_process/Model_C/tmp/{}/".format(root)
        for path_name in walk(data_root):
            for path_thread in walk(path_name):
                for path_rps in walk(path_thread):
                    paths.append(path_rps)

    for path_idx,path in tqdm(enumerate(paths)):
        if path in checkpoint.data["add"] and checkpoint.data["add"][path]=="done" and path in checkpoint.data["sub"] and checkpoint.data["sub"][path]=="done":
            continue
        else:
            try:
                df = pd.read_csv(path_rps)
                df["add"]=df.apply(lambda x: ACTION_SPACE[int(x[C_FEATURES["a"][0]])] in ACTION_SPACE_ADD,axis=1)
                df["sub"]=df.apply(lambda x: ACTION_SPACE[int(x[C_FEATURES["a"][0]])] in ACTION_SPACE_SUB,axis=1)
                add_df=df.loc[df["add"]==True]
                sub_df=df.loc[df["sub"]==True]
                add_df[C_FEATURES["a"][0]]=add_df.apply(lambda x:ACTION_ID_ADD[ACTION_SPACE[int(x[C_FEATURES["a"][0]])]],axis=1)
                sub_df[C_FEATURES["a"][0]]=sub_df.apply(lambda x:ACTION_ID_SUB[ACTION_SPACE[int(x[C_FEATURES["a"][0]])]],axis=1)
                add_df=add_df[C_FEATURES["s"]+C_FEATURES["a"]+C_FEATURES["r"]+C_FEATURES["s_"]]
                sub_df=sub_df[C_FEATURES["s"]+C_FEATURES["a"]+C_FEATURES["r"]+C_FEATURES["s_"]]
            except pd.errors.EmptyDataError as e:
                continue

            add_df=add_df.sample(frac=1)
            sub_df=sub_df.sample(frac=1)
            for _,feature in enumerate(C_FEATURES["s"]):
                add_df[feature] = (add_df[feature] - model_c_add.max_min["input_min"][_]) / (model_c_add.max_min["input_max"][_] - model_c_add.max_min["input_min"][_])
                sub_df[feature] = (sub_df[feature] - model_c_sub.max_min["input_min"][_]) / (model_c_sub.max_min["input_max"][_] - model_c_sub.max_min["input_min"][_])
            for _,feature in enumerate(C_FEATURES["s_"]):
                add_df[feature] = (add_df[feature] - model_c_add.max_min["input_min"][_]) / (model_c_add.max_min["input_max"][_] - model_c_add.max_min["input_min"][_])
                sub_df[feature] = (sub_df[feature] - model_c_sub.max_min["input_min"][_]) / (model_c_sub.max_min["input_max"][_] - model_c_sub.max_min["input_min"][_])

            add_df=add_df.dropna(axis=0,how="any")
            sub_df=sub_df.dropna(axis=0,how="any")

            if path not in checkpoint.data["add"] or checkpoint.data["add"][path]!="done":
                i=0
                while i < add_df.shape[0]:
                    model_c_add.read_memory(add_df.iloc[i:i+50, :].to_numpy())
                    i=i+50
                    model_c_add.choose_action(df.iloc[i, :8].to_numpy())
                    cost = model_c_add.learn()
                    if i%100000==0:
                        cur_timestamp=time.time()
                        duration=round(cur_timestamp-last_timestamp,2)
                        speed=round(100000/duration,2)
                        ETA=round((add_df.shape[0]-i)/speed,2)
                        print("n_files:{}, to go:{}, memory_counter: {}, duration: {}, speed:{}, ETA: {}, cost: {}".format(path_idx,len(paths)-path_idx,model_c_add.memory_counter,duration,speed,ETA,round(cost,2)))
                        last_timestamp=time.time()
                        model_c_add.save()
                checkpoint.data["add"][path]="done"
                checkpoint.store()

            if path not in checkpoint.data["sub"] or checkpoint.data["sub"][path]!="done":
                i=0
                while i < sub_df.shape[0]:
                    model_c_sub.read_memory(sub_df.iloc[i:i+50, :].to_numpy())
                    i=i+50
                    #model_c_sub.choose_action(df_sub.iloc[i, :8].to_numpy())
                    cost = model_c_sub.learn()
                    if i%100000==0:
                        cur_timestamp=time.time()
                        duration=round(cur_timestamp-last_timestamp,2)
                        speed=round(100000/duration,2)
                        ETA=round((sub_df.shape[0]-i)/speed,2)
                        print("n_files:{}, to go:{}, memory_counter: {}, duration: {}, speed:{}, ETA: {}, cost: {}".format(path_idx,len(paths)-path_idx,model_c_sub.memory_counter,duration,speed,ETA,round(cost,2)))
                        last_timestamp=time.time()
                        model_c_sub.save()
                checkpoint.data["sub"][path]="done"
                checkpoint.store()

if __name__ == '__main__':
    #train_model()
    with tf.variable_scope('add'):
        model_c_add = Model_C(len(ACTION_SPACE_ADD), len(C_FEATURES["s"]), "model_c_add",output_graph=False, e_greedy_increment=True)
    with tf.variable_scope('sub'):
        model_c_sub = Model_C(len(ACTION_SPACE_SUB), len(C_FEATURES["s"]), "model_c_sub",output_graph=False, e_greedy_increment=True)    

    #model_c_add.save()
    #model_c_sub.save()
