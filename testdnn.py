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
mainfolder = '/Users/kaushik/Documents/sppech_dnn'

directory_mask = mainfolder+'/'+'MASKSDNN'
if not os.path.exists(directory_mask):
    os.makedirs(directory_mask)

loadtestingpath = mainfolder+'/'+'test_samples' # training features

data = loadmat(loadtestingpath + "/" + "noise_3.mat")
ip = np.transpose(data['energy1'])           # incoming total context of 11
op = 64

############################################## parameters #############################################
n_input = ip.shape[1]
n_hidden1 = 512
n_hidden2 = 512
n_hidden3 = 512
n_output = op
learning_rate = 0.0001

#####################################################
# Placeholders
x = tf.placeholder(tf.float32, [None,n_input], name="noisybatch") # b_s x ip
#y_ = tf.placeholder(tf.float32, [None,n_output], name="centralcleanframe") # b_s x op

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

y = FFN(x, weights, biases)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

################################################### TRAINING ###############################################################
model_path = mainfolder + '/'+ 'model_pathDNN/model35.ckpt'


with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)

    data = loadmat(loadtestingpath + "/" + "noise_3.mat")
    batch_noisy = np.transpose(data['energy1'])
    outputspectrum = sess.run(y, feed_dict={x: batch_noisy})

    file = directory_mask + '/'+'File_3.mat'
    savemat(file, mdict={'Pred_spectrum': outputspectrum})
    print("Done")
