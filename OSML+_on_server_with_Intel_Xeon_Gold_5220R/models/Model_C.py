import numpy as np
import pandas as pd
import numpy.random as nr
from collections import deque
import argparse
import sys
sys.path.append("../")
from models.Model_C_critic_network import CriticNetwork 
from models.Model_C_actor_network import ActorNetwork
import random
import tensorflow.compat.v1 as tf
from utils import *
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from configs import ROOT, C_FEATURES, ACTION_SPACE

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log=logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()

REPLAY_BUFFER_SIZE = 76800
BATCH_SIZE = 256
GAMMA = 0.99

class Model_C(object):
    def __init__(self, action_dim, state_dim) -> None: 
        self.name = "DDPG"
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)
        
        os.system("mkdir -p "+ROOT+"/data/data_collection/Model_C")
        self.export_filename=ROOT+"/data/data_collection/Model_C/RL_DDPG_data.csv"
        self.export_filename_test=ROOT+"/data/data_collection/Model_C/RL_DDPG_data_test.csv"
        self.export_file=open(self.export_filename,"a")
        self.export_file_test=open(self.export_filename_test,"a")
        self.saver = tf.train.Saver()
        self.pointer = 0
        self.load_ckpt()
        self.load_memory_pool()

    def __del__(self):
        self.export_file.close()
        self.export_file_test.close()

    def learn(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

        #log.info("minibatch:{}".format(minibatch))
        #log.info("next_action_batch:{}".format(next_action_batch))
        #log.info("q_value_batch:{}".format(q_value_batch))
        #log.info("y_batch:{}".format(y_batch))
        #log.info("action_batch_for_gradients:{}".format(action_batch_for_gradients))
        #log.info("q_gradient_batch:{}".format(q_gradient_batch))

        if self.pointer%100==0:
            self.save_ckpt()

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action+self.exploration_noise.noise()

    def choose_action(self,state):
        action = self.actor_network.action(state)
        return action

    def store_transition(self,state,action,reward,next_state,save_to_file=True):
        # Store to file
        transition = np.hstack((state, action, [reward], next_state))
        if any(np.isnan(transition)) or None in transition:
            print("NaN in inputs of Model-C:", state,action,reward,next_state)
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state)
        self.pointer += 1
        if save_to_file:
            text_to_write=",".join([str(item) for item in list(transition)])
            self.export_file.write(text_to_write+"\n")

    def store_transition_test(self,state,action,reward,next_state,save_to_file=True):
        transition = np.hstack((state, action, [reward], next_state))
        if save_to_file:
            text_to_write=",".join([str(item) for item in list(transition)])
            self.export_file_test.write(text_to_write+"\n")

    def export_file_line_cnt(self):
        if os.path.exists(self.export_filename):
            outs,errs=shell_output("wc -l {}".format(self.export_filename))
            cnt=int(outs.split()[0])
            return cnt
        else:
            return 0

    def load_memory_pool(self):
        csv_path=ROOT+"/data/data_collection/Model_C/RL_DDPG_data.csv"
        if not os.path.exists(csv_path):
            print("Model-C memory pool csv_path does not exist: {}".format(csv_path))
            return
        try:
            arr=pd.read_csv(csv_path,header=None).to_numpy()
        except pd.errors.EmptyDataError:
            return
        for i in range(len(arr)):
            s = arr[i][:self.state_dim]
            a = arr[i][self.state_dim:self.state_dim + self.action_dim]
            r = arr[i][-self.state_dim - 1:-self.state_dim][0]
            s_ = arr[i][-self.state_dim:]
            self.store_transition(s,a,r,s_,save_to_file=False)
    
    def save_ckpt(self):
        dir_name=ROOT+"/models/checkpoints/Model_C/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_path=dir_name+"{}/{}".format(self.name,self.name)
        self.saver.save(self.sess,model_path,write_meta_graph=True)

        dir_name=ROOT+"/models/checkpoints/Model_C/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_path=dir_name+"{}.{}/{}".format(self.name,self.pointer,self.name)
        self.saver.save(self.sess,model_path,write_meta_graph=True)

    def load_ckpt(self):
        dir_name=ROOT+"/models/checkpoints/Model_C/"
        model_path=dir_name+"{}".format(self.name)
        path = tf.train.latest_checkpoint(model_path)
        if path is not None:
            self.saver.restore(self.sess, path)
            print_color("Model C load successfully.","green")


    def restore(self):
        saver = tf.train.Saver()
        path = tf.train.latest_checkpoint(model_dir)
        if path is not None:
            saver.restore(self.sess, path)

    def read_memory(self, transition):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition[0:self.transistion_lenth]
        self.memory_counter += 1


    def draw_reward(self,mode):
        mode=int(mode)
        # mode == 1: episodic
        # mode == 2: step
        # mode == 3: legal_actions's mean
        fig_name="reward_episodic.png"

        plt.rc('font', family='ARIAL')
        figsize = get_figsize(w=10, h=6, dpi=300)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams["figure.figsize"] = figsize

        fig, ax = plt.subplots(1, 1)
        MAX_EP_STEPS = 200 
        y = []
        if mode==1:
            episodes = len(self.replay_buffer.step_reward)//(MAX_EP_STEPS)
            for i in range(episodes):
                y.append(sum(self.replay_buffer.step_reward[i*MAX_EP_STEPS:(i+1)*MAX_EP_STEPS]))
        elif mode==2:
            episodes = len(self.replay_buffer.step_reward)//(MAX_EP_STEPS)
            for i in range(episodes):
                y.append(np.mean(self.replay_buffer.step_reward[i*MAX_EP_STEPS:(i+1)*MAX_EP_STEPS]))
        elif mode==3:
            y = self.replay_buffer.step_reward

        x = np.arange(0, len(y))
        if mode==3:
            ax.scatter(x,y,label="reward",s=1)
        else:
            ax.plot(x, y, label="reward")
        ax.set_xlabel('Episode Steps')
        ax.set_ylabel("Episodic Reward", fontdict={'size': 'medium', 'weight': 'bold'})
        ax.legend(labelspacing=0, loc="upper right")
        plt.savefig(fig_name)

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state
    

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.step_reward=[]
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        self.step_reward.append(reward)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


def draw_reward(param):
    model = Model_C(len(ACTION_SPACE), len(C_FEATURES["s"]))
    model.load_memory_pool()
    model.draw_reward(param)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--draw', default=1)

    args=parser.parse_args()
    if args.train:
        train_network()
    if args.test:
        test_network()
    draw_reward(args.draw)
