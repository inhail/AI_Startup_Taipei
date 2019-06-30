# -*- coding: utf-8 -*-
"""
LeNet 5 - 1998

Architecture
Layer 1: Convolutional. The output shape should be 28x28x6.

Activation. Your choice of activation function.

Pooling. The output shape should be 14x14x6.

Layer 2: Convolutional. The output shape should be 10x10x16.

Activation. Your choice of activation function.

Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Layer 3: Fully Connected. This should have 120 outputs.

Activation. Your choice of activation function.

Layer 4: Fully Connected. This should have 84 outputs.

Activation. Your choice of activation function.

Layer 5: Fully Connected (Logits). This should have 10 outputs.

@author: Jerry
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data#直接從路徑資料夾引入minst data
from tensorflow.contrib.layers import flatten   #使用tensorflow的API作Flatten，拉成ㄧ維vector Ex 28x28 -> 784

# Load MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

# Preprocessing Data
## Add Padding 2x2x2x2 (28x28 to 32x32)
X_train = np.pad(X_train,((0,0),(2,2),(2,2),(0,0)),'constant')
X_test = np.pad(X_test,((0,0),(2,2),(2,2),(0,0)),'constant')
X_validation = np.pad(X_validation,((0,0),(2,2),(2,2),(0,0)),'constant')
#print(X_train.shape)


# Create LeNet Convolution Neural Network
## Create parameters
EPOCHS = 10
BATCH_SIZE = 128
## LeNet main function
def LeNet(x):
    #Hyperparameters
    #Layer 1: Convolution (32x32x1 -> 28x28x6)
    conv1_w = tf.Variable(tf.random_normal([5,5,1,6]))
    conv1_b = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
    ## Activation function
    conv1 = tf.nn.sigmoid(conv1)
    
    #Layer 2: Max pooling (28x28x6 -> 14x14x6)
    conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #Layer 3: Convolution (14x14x6 -> 10x10x16)
    conv2_w = tf.Variable(tf.random_normal([5,5,6,16]))
    conv2_b = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    ## Activation function
    conv2 = tf.nn.sigmoid(conv2)
    
    #Layer 4: Max pooling (10x10x16 -> 5x5x16)
    conv2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #Layer 5: Fully Connected (5x5x16 = 400 -> 120)
    dense1_w = tf.Variable(tf.random_normal([400,120]))
    dense1_b = tf.Variable(tf.zeros([120]))
    conv2 = flatten(conv2)
    dense1 = tf.matmul(conv2, dense1_w) + dense1_b
    ## Activation function
    dense1 = tf.nn.sigmoid(dense1)
    
    #Layer 6: Fully Connected (120 -> 84)
    dense2_w = tf.Variable(tf.random_normal([120,84]))
    dense2_b = tf.Variable(tf.zeros([84]))
    dense2 = tf.matmul(dense1, dense2_w) + dense2_b
    ## Activation function
    dense2 = tf.nn.sigmoid(dense2)
    
    #Layer 7: Fully Connected (84 -> 10)
    dense3_w = tf.Variable(tf.random_normal([84,10]))
    dense3_b = tf.Variable(tf.zeros([10]))
    dense3 = tf.matmul(dense2, dense3_w) + dense3_b
    return dense3
    
## Main function
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        #X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")


# Test Accuracy
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.2f}".format(test_accuracy))