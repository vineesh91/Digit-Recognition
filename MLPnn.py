
# coding: utf-8

# In[3]:


import numpy as np
from scipy.special import expit
from constants import *
import os, struct
import sys


# In[4]:


#Reference : http://www.zhanjunlang.com/resources/tutorial/Python%20Machine%20Learning.pdf
def load_mnist(path, kind='train'):
    labels_path = os.path.join( path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join( path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath,  dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


# In[14]:


def load_usps_file(path):
        X = []
        y = []
        with gzip.open(path, 'r') as f:
            for line in f.readlines():
                sample = line.strip().split()
                y.append(int(float(sample[0])))
                flat_img = [float(val) for val in sample[1:]]
                flat_img = np.array(flat_img, dtype=np.float32)
                X.append(flat_img.reshape((1, 1, 16, 16)))
        y = np.array(y).astype(np.int32)
        X = np.concatenate(X, axis=0).astype(np.float32)
        return X * 0.5 + 0.5, y


# In[85]:


X_train, y_train = load_mnist('data', kind='train')


# In[36]:


X_test, y_test = load_mnist('data', kind='t10k')


# In[47]:


class Network(object):
    def __init__(self, n_features, n_hidden, n_output, epochs, eta):
        np.random.seed(1)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.eta = eta
        self.minibatches = 30
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        print w1.shape
        self.w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        self.w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        W1 = np.random.randn(n_features, n_hidden) / np.sqrt(n_features) 
        B1 = np.zeros((1, n_hidden)) 
        W2 = np.random.randn(n_hidden, n_output) / np.sqrt(n_hidden) 
        B2 = np.zeros((1, n_output)) 
        print self.w1.shape
        print self.w2.shape
                
    def vectorized_result(self, y, k):
        vectorise = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            vectorise[val, idx] = 1
        return vectorise

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_gradient(self, z):
        sg = self.sigmoid(z)
        return sg * (1.0 - sg)

    def add_bias(self, X, how='b1'):
        if how == 'b1':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'b2':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

    def feedforward1(self, X, w1, w2):  
        z2 = X.dot(W1) + B1 
        a2 = self.sigmoid(z2) 
        z3 = a1.dot(W2) + B2 
        a3 = self.softmax(z3)
        return z2, a2, z3, a3
    
    def feedforward(self, X, w1, w2):
        a1 = self.add_bias(X, how='b1')
        z2 = w1.dot(a1.T)
        a2 = self.sigmoid(z2)
        a2 = self.add_bias(a2, how='b2')
        z3 = w2.dot(a2)
        a3 = self.softmax(z3)
        return a1, z2, a2, z3, a3
  
    def cross_entropy_error(self, yout, output, w1, w2):
        term1 = -yout * (np.log(output))
        return term1

    def gradientdescent(self, a1, a2, a3, z2, yout, w1, w2):
        # backpropagation
        sigma3 = a3 - yout
        z2 = self.add_bias(z2, how='b2')
        sigma2 = w2.T.dot(sigma3) * self.sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        return grad1, grad2

    def softmax(self,z):
#         return np.exp(z) / np.sum(np.exp(z))
        return expit(z)

    def predict(self, X):
        a1, z2, a2, z3, a3 = self.feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y):
        self.cost1 = []
        X_data, y_data = X.copy(), y.copy()
        yout = self.vectorized_result(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            idx = np.random.permutation(y_data.shape[0])
            X_data, yout = X_data[idx], yout[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:

                # feedforward
                a1, z2, a2, z3, a3 = self.feedforward(X_data[idx],self.w1,self.w2)
                cost = self.cross_entropy_error(yout[:, idx],a3,self.w1,self.w2)
                self.cost1.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self.gradientdescent(a1,a2,a3,z2,yout[:, idx],self.w1,self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 = self.w1 - delta_w1 
                self.w2 = self.w2 - delta_w2
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self


# In[81]:


net = Network(784,15,10,1000,0.001)


# In[82]:


net.fit(X_train, y_train)


# In[83]:


prediction = net.predict(X_test)

acc = ((np.sum(y_test == prediction, axis=0)).astype('float') / X_test.shape[0])

print('\nTraining accuracy: %.2f%%' % (acc * 100))


# In[84]:


prediction = net.predict(X_train)

acc = ((np.sum(y_train == prediction, axis=0)).astype('float') / X_train.shape[0])

print('\nTraining accuracy: %.2f%%' % (acc * 100))

