import tensorflow.compat.v1 as tf 
import numpy as np
import math
import sys
sys.path.append("../")


# Hyper Parameters
LEARNING_RATE = 0.0005
TAU = 0.001
BATCH_SIZE = 64
W_FINAL=0.003
VAR = 0.6

class ActorNetwork:
        """docstring for ActorNetwork"""
        def __init__(self,sess,state_dim,action_dim):

                self.sess = sess
                self.state_dim = state_dim
                self.action_dim = action_dim
                # create actor network
                self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim,action_dim)

                # create target actor network
                self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net)

                # define training rules
                self.create_training_method()

                self.sess.run(tf.initialize_all_variables())

                self.update_target()
                #self.load_network()

        def create_training_method(self):
                self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
                self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

        def create_network(self,state_dim,action_dim):
                layer1_size = 32
                layer2_size = 64
                layer3_size = 32

                state_input = tf.placeholder("float",[None,state_dim])
                is_training = tf.placeholder(tf.bool)

                W1 = self.variable([state_dim,layer1_size],state_dim)
                b1 = self.variable([layer1_size],state_dim)

                W2 = self.variable([layer1_size,layer2_size],layer1_size)
                b2 = self.variable([layer2_size],layer1_size)

                W3 = self.variable([layer2_size,layer3_size],layer2_size)
                b3 = self.variable([layer3_size],layer2_size)

                W4 = tf.Variable(tf.random_uniform([layer3_size,action_dim],-1*W_FINAL,W_FINAL))
                b4 = tf.Variable(tf.random_uniform([action_dim],-1*W_FINAL,W_FINAL))
                
                layer1 = tf.matmul(state_input,W1) + b1
                layer2 = tf.matmul(layer1,W2) + b2
                layer3 = tf.matmul(layer2,W3) + b3
                action_output = tf.nn.softmax(tf.matmul(layer3,W4) + b4)

                return state_input,action_output,[W1,b1,W2,b2,W3,b3,W4,b4],is_training

        def create_target_network(self,state_dim,action_dim,net):
                state_input = tf.placeholder("float",[None,state_dim])
                is_training = tf.placeholder(tf.bool)
                ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
                target_update = ema.apply(net)
                target_net = [ema.average(x) for x in net]

                layer1 = tf.matmul(state_input,target_net[0]) + target_net[1]
                layer2 = tf.matmul(layer1,target_net[2]) + target_net[3]
                layer3 = tf.matmul(layer2,target_net[4]) + target_net[5]

                action_output = tf.nn.softmax(tf.matmul(layer3,target_net[6]) + target_net[7])

                return state_input,action_output,target_update,is_training

        # f fan-in size
        def variable(self,shape,f):
                return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

        def update_target(self):
                self.sess.run(self.target_update)

        def train(self,q_gradient_batch,state_batch):
            self.sess.run(self.optimizer,feed_dict={
                self.q_gradient_input:q_gradient_batch,
                self.state_input:state_batch,
                self.is_training: True})
            '''
            print("==============Actor gradient")
            gradient=self.sess.run(self.parameters_gradients,feed_dict={self.q_gradient_input:q_gradient_batch,self.state_input:state_batch,self.is_training: True})
            gradient=[np.mean(item) for item in gradient]
            for i in zip(gradient,self.net):
                print(i)
            print("==============Actor gradient end")
            '''

        def actions(self,state_batch):
                return self.sess.run(self.action_output,feed_dict={
                        self.state_input:state_batch,
                        self.is_training: True
                        })

        def action(self,state):
                action_list = self.sess.run(self.action_output,feed_dict={
                        self.state_input:[state],
                        self.is_training: False
                        })
                noise_action = np.random.normal(action_list, VAR).tolist()[0]
                max_action=max(noise_action)
                max_index=noise_action.index(max_action)
                return max_index, noise_action


        def target_actions(self,state_batch):
                return self.sess.run(self.target_action_output,feed_dict={
                        self.target_state_input: state_batch,
                        self.target_is_training: True
                        })



        def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
                return tf.cond(training_phase, 
                lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
                lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
                updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))
'''
        def load_network(self):
                self.saver = tf.train.Saver()
                checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
                if checkpoint and checkpoint.model_checkpoint_path:
                        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                        print "Successfully loaded:", checkpoint.model_checkpoint_path
                else:
                        print "Could not find old network weights"
        def save_network(self,time_step):
                print 'save actor-network...',time_step
                self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

                
