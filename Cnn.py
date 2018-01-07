
# coding: utf-8

# In[131]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[136]:


mnist.test.labels


# In[6]:


import numpy as np
import glob
import os
from PIL import Image
from skimage import data, io, filters, color
from skimage.transform import rescale, resize
import os
#from PIL import Image
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer


# In[7]:


import tensorflow as tf
sess = tf.InteractiveSession()


# In[8]:


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[9]:


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[10]:


sess.run(tf.global_variables_initializer())


# In[11]:


y = tf.matmul(x,W) + b


# In[12]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# In[13]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[16]:


for _ in range(1000):
batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# In[17]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[18]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[176]:


print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[102]:


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# In[103]:


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[104]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[105]:


x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[106]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[107]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[108]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[109]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[110]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[169]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[172]:


print('test accuracy %g' % accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0}))


# In[171]:


#Loading the images from USPS dataset and resizing them to 28*28
test_images = np.zeros((19999,784))
i = 0
test_labels = np.zeros((19999,1))
a = np.ones((28,28), dtype=np.int)
for image_path in glob.glob(os.path.join('/Users/issackoshypanicker/Desktop/MLHW3/proj3_images/Numerals','*','*.png')):
    dirname = os.path.basename(os.path.dirname(image_path))
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = resize(image, (28, 28))
    image = (a - image)*255
    #y = np.asarray(image.getdata(), dtype=np.float64).reshape((28, 28))
    test_images[i,:] = image.reshape(28*28)
#     test_images = 
    test_labels[i,:] = int(dirname)
    i += 1
lb = LabelBinarizer()
test_labels = lb.fit_transform(test_labels)
test_labels=test_labels.astype(np.int32)


# In[115]:


tf.initialize_all_variables().run()


# In[149]:


print('test accuracy usps %g' % accuracy.eval(feed_dict={
    x: test_images, y_: test_labels, keep_prob: 1}))

