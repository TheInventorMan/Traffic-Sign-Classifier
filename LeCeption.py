import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeCeption(x):    
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.leaky_relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2_1: Convolutional (Inception 1). Input = 14x14x6. Output = 10x10x16.
    conv21_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv21_b = tf.Variable(tf.zeros(16))
    conv21   = tf.nn.conv2d(conv1, conv21_W, strides=[1, 1, 1, 1], padding='VALID') + conv21_b
    
    # Layer 2_2: Convolutional (Inception 2). Input = 14x14x6. Output = 12x12x16.
    conv22_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma))
    conv22_b = tf.Variable(tf.zeros(16))
    conv22   = tf.nn.conv2d(conv1, conv22_W, strides=[1, 1, 1, 1], padding='VALID') + conv22_b
    
    # Layer 2_2: Double Max Pool. Input = 12x12x16. Output = 10x10x16
    conv22 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    conv22 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    
    # Inception Stack. Output = 10x10x32
    conv2 = tf.concat((conv21, conv22), 3)
    
    # Activation.
    conv2 = tf.nn.leaky_relu(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
    # Branch off bypass. Input = 5x5x32. Output = 800.
    bypBranch = flatten(conv2)
    
    # Main feedforward path
    # Layer 3: Convolutional. Input = 5x5x32. Output = 1x1x800.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 800), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(800))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Activation
    conv3 = tf.nn.leaky_relu(conv3)
    
    # Merge branches. Output = 1600.
    mainBranch = flatten(conv3)
    fc0 = tf.concat([mainBranch, bypBranch], 1)
    
    fc0 = tf.nn.dropout(fc0, keep_prob)

    # Layer 4: Fully Connected. Input = 1600. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    # Activation.
    fc1    = tf.nn.leaky_relu(fc1)
    
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # Layer 5: Fully Connected. Input = 400. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.leaky_relu(fc2)
    
    fc2    = tf.nn.dropout(fc2, keep_prob)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
