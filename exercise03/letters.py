'''
Created on Nov 22, 2016

@author: jonas
'''

import tensorflow as tf
import math
import time
import preprocessing
#from tensorflow.examples.tutorials.mnist import input_data
import os.path

IMAGE_PIXELS = 30*30
NUM_CLASSES = 52
BATCH_SIZE = 100
MAX_STEPS = 450
EVAL_STEPS = 50

class MNISTListIterator:
    def __init__(self, l, batchsize):
        self.i = 0
        self.l = l
        self.batchsize = batchsize
        
    def __iter__(self):
        return self
    
    def next(self):
        if self.i + self.batchsize - 1< len(self.l):
            i = self.i
            self.i += self.batchsize
            return self.l[i:i + self.batchsize]
        else:
            raise StopIteration()
        

def classify(images):
    '''with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, 128], stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([128], name='biases'))
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
   
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([128, 32], stddev = 1.0 / math.sqrt(float(128))), name='weights')
        biases = tf.Variable(tf.zeros([32], name='biases'))
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([32, NUM_CLASSES], stddev = 1.0 / math.sqrt(float(32))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES], name='biases'))
        logits = tf.matmul(hidden2, weights) + biases'''
    
    with tf.name_scope("conv1"):
        weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32), name='biases')
        conv1 = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
    
    with tf.name_scope("pool1"):
        hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    with tf.name_scope("conv2"):
        weights = tf.Variable(tf.truncated_normal([6, 6, 32, 64], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='biases')
        conv2 = tf.nn.conv2d(hidden2, weights, strides=[1, 1, 1, 1], padding='SAME')
        hidden3 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
    
    with tf.name_scope("pool2"):
        hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    '''with tf.name_scope("conv3"):
        weights = tf.Variable(tf.truncated_normal([5, 5, 32, 512], stddev=0.1, dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='biases')
        conv3 = tf.nn.conv2d(hidden4, weights, strides=[1, 1, 1, 1], padding='VALID')
        hidden5 = tf.nn.relu(tf.nn.bias_add(conv3, biases))'''
    
    with tf.name_scope('hidden6'):
        reshape = tf.reshape(hidden4, [BATCH_SIZE, 4096])
        weights = tf.Variable(tf.truncated_normal([4096, 512], stddev = 1.0 / math.sqrt(float(4096))), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32), name='biases')
        hidden6 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        #hidden6 = tf.nn.dropout(hidden6, 0.5)
    
    '''with tf.name_scope('hidden7'):
        weights = tf.Variable(tf.truncated_normal([512, 128], stddev = 1.0 / math.sqrt(float(512))), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32), name='biases')
        hidden7 = tf.nn.relu(tf.matmul(hidden6, weights) + biases)
        #hidden7 = tf.nn.dropout(hidden7, 0.5)'''
    
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev = 1.0 / math.sqrt(float(512))), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32), name='biases')
        logits = tf.matmul(hidden6, weights) + biases
        
    return logits

def calc_loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def train_network(loss, learning_rate, global_step):
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluate(logits, labels):
    labels = tf.to_int64(labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def eval_unlabeled(logits):
    return tf.nn.top_k(logits)

def run(basedata=None):
    print("Reading data...\n")
    if basedata is None:
        basedata = preprocessing.read_labeled_file('letters-dataset1-labeled.txt', normalize=False, letters=True)
    print("Data read.\n")
    image_iterator = MNISTListIterator(basedata['images'], BATCH_SIZE)
    label_iterator = MNISTListIterator(basedata['labels'], BATCH_SIZE)
    #mnist = input_data.read_data_sets("MNIST_data/")
    print("Starting classification...\n")
    g = tf.Graph()
    with g.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 30, 30, 1))
        labels_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE))
        logits = classify(images_placeholder)
        loss = calc_loss(logits, labels_placeholder)
        learning_rate = tf.train.exponential_decay(0.1, global_step, 450, 0.95, staircase=True)
        train_op = train_network(loss, learning_rate, global_step)
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        saver = tf.train.Saver()
        if os.path.isfile("./model.ckpt"):
            print("Restoring model...")
            saver.restore(sess, "./model.ckpt")
        else:
            sess.run(init_op)
        for step in range(MAX_STEPS):
            #images, labels = mnist.train.next_batch(BATCH_SIZE)
            #images = images.reshape(BATCH_SIZE, 28, 28, 1)
            feed_dict = {images_placeholder: image_iterator.next(), labels_placeholder: label_iterator.next()}
            start_time = time.time()
            _, lr, loss_value = sess.run([train_op, learning_rate, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            
            if step % 10 == 0:
                print(str(step) + ": loss = " + str(loss_value) + ", learning_rate=" + str(lr) + ", " + str(duration) + " sec/batch\n")
        saver.save(sess, "./model.ckpt")
        true_count = 0
        do_eval = evaluate(logits, labels_placeholder)
        print("Evaluating network...\n")
        for step in range(EVAL_STEPS):
            #images, labels = mnist.validation.next_batch(BATCH_SIZE)
            #images = images.reshape(BATCH_SIZE, 28, 28, 1)
            feed_dict = {images_placeholder: image_iterator.next(), labels_placeholder: label_iterator.next()}
            true_count += sess.run(do_eval, feed_dict=feed_dict)
        precision = true_count / float(EVAL_STEPS*BATCH_SIZE)
        print("Num examples: " + str(EVAL_STEPS*BATCH_SIZE) + ", num correct: " + str(true_count) + ", precision: " + str(precision))

def run_unlabeled(data, letters=False):
    print("Starting classification...\n")
    image_iterator = MNISTListIterator(data, BATCH_SIZE)
    g = tf.Graph()
    with g.as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 30, 30, 1))
        logits = classify(images_placeholder)
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        saver = tf.train.Saver()
        if os.path.isfile("./model.ckpt"):
            print("Restoring model...")
            saver.restore(sess, "./model.ckpt")
        else:
            sess.run(init_op)
        result = list()
        for i in range(500):
            feed_dict = {images_placeholder: image_iterator.next()}
            _, identified_class = sess.run(eval_unlabeled(logits), feed_dict=feed_dict)
            for j in range(BATCH_SIZE):
                if letters:
                    if (identified_class[j][0] < 26):
                        result.append(chr(identified_class[j][0] + 65))
                    else:
                        result.append(chr(identified_class[j][0] + 97 - 26))
                else:
                    result.append(str(identified_class[j][0])[0])
    
    return result
            
        
