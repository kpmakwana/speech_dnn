import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy
from scipy import io as sio
from scipy.io import loadmat
from scipy.io import savemat

###########################################  make directory ############################################
mainfolder = "/Users/kaushik/Documents/sppech_dnn"

directory_mask = mainfolder+'/'+'MASKSDNN'
if not os.path.exists(directory_mask):
    os.makedirs(directory_mask)

directory_model = mainfolder+'/'+'model_pathDNN'
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

loadtrainingpath = mainfolder+'/'+'features' # training features

data = loadmat(loadtrainingpath + "/" + "Batch_0.mat")
ip = np.transpose(data['noisy'])        # incoming total context of 11
op = np.transpose(np.log(data['clean'])) # b_s*64, # log-true labels

############################################## parameters #############################################
n_input = ip.shape[1]
print ("Input :" + str(n_input))
n_hidden1 = 512
n_hidden2 = 512
n_hidden3 = 512
n_output = op.shape[1]
print ("Input :" + str(n_output))
learning_rate = 0.0001

training_epochs = 35
training_batches = 2000

#####################################################
# Placeholders
x = tf.placeholder(tf.float32, [None,n_input], name="noisybatch") # b_s x ip
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralcleanframe") # b_s x op

# Create model
def FFN(x, weights, biases):
    layer_1 = tf.add(tf.matmul((x), weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    mask = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return mask

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])*.0001),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])*.0001),
    'h3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3])*.0001),
    'out': tf.Variable(tf.random_normal([n_hidden3, n_output])*.0001)
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])*0.01),
    'b2': tf.Variable(tf.random_normal([n_hidden2])*0.01),
    'b3': tf.Variable(tf.random_normal([n_hidden3])*0.01),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
y = FFN(x, weights, biases)

# compute cross entropy as our loss function
cost = 0.5*(tf.reduce_mean(tf.square(tf.subtract(y_, y))))

# use GD as optimizer to train network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize all variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

################################################### TRAINING ################################################################
k = 0
model_path = directory_model + "/" + "model" + str(k) + ".ckpt"

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, model_path)

    for epoch in range(0,training_epochs):
        saver.restore(sess, model_path)

        Rand_files = np.random.permutation(training_batches)
        batch_index = 0;
        for batch in Rand_files:

            data = loadmat(loadtrainingpath + "/" + "Batch_" + str(batch)+".mat")

            batch_noisy = np.transpose(data['noisy']) #ip
            batch_cleancentral = np.transpose(np.log(data['clean'])) # label

            costs,_ = sess.run([cost,optimizer], feed_dict={x: batch_noisy, y_: batch_cleancentral})

            print ("Epoch: "+str(epoch)+" Batch_index: "+str(batch_index)+" Cost= "+str(costs))
            batch_index = batch_index+1

        k = k+1
        model_path = directory_model + "/" + "model" + str(k) + ".ckpt"
        save_path = saver.save(sess, model_path)
