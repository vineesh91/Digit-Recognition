import numpy as np
import struct
from MultiLogReg import MultiLogReg
from MultiLogReg import test_log_model
from sklearn.model_selection import train_test_split
from skimage import data, io, filters, color
from skimage.transform import rescale, resize
import os
import glob

#Loading the MNIST training and testing labels and vectors

fpath = 'C:/Users/vinee/PycharmProjects/ML3/Data'
trainfeature = open(os.path.join(fpath,'train-images.idx3-ubyte'),'rb')
cr, size, rtrain, ctrain = struct.unpack(">IIII", trainfeature.read(16))
trainlabels = open(os.path.join(fpath,'train-labels.idx1-ubyte'),'rb')
cr, size = struct.unpack(">II", trainlabels.read(8))
testfeatures = open(os.path.join(fpath,'t10k-images.idx3-ubyte'),'rb')
cr, size, rtest, ctest = struct.unpack(">IIII", testfeatures.read(16))
testlabels = open(os.path.join(fpath,'t10k-labels.idx1-ubyte'),'rb')
cr, size = struct.unpack(">II", testlabels.read(8))

#extracting the data and reshaping the image data rom a 28*28 to a 784 vector
mnist_train_features = (np.fromfile(trainfeature, dtype=np.uint8).reshape(60000, rtrain*ctrain))/255.0
mnist_train_labels = np.fromfile(trainlabels, dtype=np.int8)
mnist_test_features = (np.fromfile(testfeatures, dtype=np.uint8).reshape(10000, rtest*ctest))/255.0
mnist_test_labels = np.fromfile(testlabels, dtype=np.int8)

#fetching the images corresponding to each label
#for i in range(60000):
   # mnist_train_images = array((img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\.reshape((rows, cols)))

# splitting the training data to training and validation sets

test_images = np.zeros((20000,784))
i = 0

#Loading the images and resizing them to 28*28
test_labels = np.zeros((20000))
for image_path in glob.glob(os.path.join('C:/Users/vinee/PycharmProjects/ML3/Data/proj3_images/Numerals','*','*.png')):
    dirname = os.path.basename(os.path.dirname(image_path))
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = resize(image, (28, 28))
    #y = np.asarray(image.getdata(), dtype=np.float64).reshape((28, 28))
    sft = np.ones(784)
    test_images[i,:] = sft -image.reshape(28*28)

    test_labels[i] = int(dirname)
    i += 1

#for bias
bs = np.ones((20000,1))
test_images = np.append(test_images,bs,axis=1)
mnist_test_features,mnist_val_features,mnist_test_labels,mnist_val_labels= train_test_split(mnist_test_features,mnist_test_labels,test_size = 0.5)



weights = MultiLogReg(mnist_train_features,mnist_val_features,mnist_test_features,mnist_train_labels,mnist_val_labels,mnist_test_labels)

test_log_model(weights,test_images,test_labels,20000)


