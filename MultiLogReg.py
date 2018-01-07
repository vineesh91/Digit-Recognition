import numpy as np
import glob
#import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

def MultiLogReg(trainIn,valIn,testIn, trainOut,valOut,testOut):
    #initializing weigths
    trainsz = 60000
    testsz = 5000
    imgsz = 784
    wt = np.random.rand(imgsz+1,10)/10
    ak = np.zeros((trainsz,10))
    yk = np.zeros((trainsz,10))

    #appending vector to data for bias
    b1 = np.ones((trainsz,1))
    b2 = np.ones((testsz,1))
    trainIn = np.append(trainIn,b1,axis=1)
    testIn = np.append(testIn,b2,axis=1)
    valIn = np.append(valIn,b2,axis=1)


    #converting labels to one hot matrix representaion
    lb1 = LabelBinarizer()
    yyy = trainOut
    lb1.fit(range(max(trainOut) + 1))
    trainOut = lb1.fit_transform(trainOut)
    #testOut = lb1.fit_transform(testOut)
    valOut = lb1.fit_transform(valOut)
    #initializing iteration amount and grad descent step size
    N = 200
    #gSteps = [0.0001, 0.0005, 0.0009, 0.001]
    gSteps = [0.5]
    er_best = 100
    for gStep in gSteps:
        for i in range(0,N):
            # applying softmax function
            ak = np.matmul(wt.T,trainIn.T)
            pval = np.exp(ak)
            pout = pval / pval.sum(axis=0)
            diff = pout.T - trainOut
            err_grad = (np.matmul(diff.T, trainIn))
            err_grad = (1 / 60000.) * err_grad
            wt = (wt - (gStep * err_grad.T))
            err = -np.sum((np.multiply(trainOut.T, np.log(pout))))
            err = err/60000
            #print(Nwrong)
            #print(err/60000)
        if err < er_best:
            gstep_best = gStep
            er_best = err

    #training using the optimal learning rate    for i in range(0, N):
    i = 0
    wt = np.random.rand(imgsz + 1, 10) / 10
    for i in range(0, N):
        # applying softmax function
        ak = np.matmul(wt.T, trainIn.T)
        pval = np.exp(ak)
        pout = pval / pval.sum(axis=0)
        diff = pout.T - trainOut
        err_grad = (np.matmul(diff.T, trainIn))
        err_grad = (1 / 60000.) * err_grad
        wt = (wt - (gstep_best * err_grad.T))
        err = -np.sum((np.multiply(trainOut.T, np.log(pout))))
        err = err / 60000

        regTrainLabels = np.argmax(pout, axis=0)
        regTrainOut = np.argmax(trainOut, axis=1)
        diffVal = regTrainOut - regTrainLabels
        Nwrong1 = np.count_nonzero(diffVal)

        #checking the values on validation data to avoid overfitting
        regValOut = np.matmul(wt.T, valIn.T)
        akval = np.exp(regValOut)
        outval = akval / akval.sum(axis=0)
        err_val = -np.sum((np.multiply(valOut.T, np.log(outval))))
        err_val = err_val / 5000
        regValLabels = np.argmax(outval, axis=0)
        regValOut = np.argmax(valOut, axis=1)
        diffVal = regValLabels - regValOut
        Nwrong = np.count_nonzero(diffVal)
        E = Nwrong / 5000
        if i == 0:
            err_val_prev = err_val
        else:
            if err_val > err_val_prev:
                i = N-1

    #final error rate for training data
    regTrainLabels = np.argmax(pout, axis=0)
    regTrainOut = np.argmax(trainOut, axis=1)
    diffVal = regTrainOut - regTrainLabels
    Nwrong = np.count_nonzero(diffVal)
    kkk = Nwrong/60000

    #test data
    test_log_model(wt,testIn,testOut,5000)
    return wt

def test_log_model(wt,testIn,testOut,size):
    #applying the weights on the test data
    regTestOut = np.matmul(wt.T,testIn.T)
    pval = np.exp(regTestOut)
    pout = pval / np.sum(pval, axis=0)
    regTestLabels = np.argmax(pout,axis=0)
    #regTestOut = np.argmax(testOut, axis=1)
    diffVal = testOut-regTestLabels
    Nwrong = np.count_nonzero(diffVal)
    E = Nwrong/size
    #correct_prediction = tf.equal(tf.argmax(testOut, 1), tf.argmax(pout.T, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(E)
