'''
Give a trained model, this will test the imputation performance..
'''

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_mpf import  *
from PIL import Image


class get_samples(object):

    def __init__(self, hidden_list = [], W = None, b = None):
        self.hidden_list = hidden_list
        self.num_rbm = len(hidden_list) - 1
        self.W = W
        self.b = b

    def propup(self, i, data, CD):

        if CD:
            pre_sigmoid_activation = np.dot(data, self.W) \
                             + self.b
        else:
            vis_units = self.hidden_list[i]
            pre_sigmoid_activation = np.dot(data, self.W[i][:vis_units,vis_units:]) \
                             + self.b[i][vis_units:]

        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]


    def get_mean_activation(self, input_data, CD = False):

        act = None
        for i in range(self.num_rbm):

            act = self.propup(i, data = input_data, CD = CD)
            input_data = act[1]

        return act[1]


def element_corruption(data, corrupt_row = 2):
    row = random.sample(range(10,18), 1)[0]
    x =  np.random.binomial(n=1, p= 0.5, size=(data.shape[0], corrupt_row * 28))
    data[:,row*28:(row+corrupt_row)*28]  += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def top_corruption(data, corrupt_row = 10, start = 4):
    x = np.random.binomial(n=1,p=0.5, size=(data.shape[0], 28*corrupt_row))
    data[:,28*start : start *28 + 28*corrupt_row] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def bottom_corruption(data, corrupt_row = 10, start = 0):
    x = np.random.binomial(n=1,p=0.5, size=(data.shape[0], 28*corrupt_row))
    data[:,28*(28-corrupt_row - start): 28*(28-start)] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def left_corruption(data, corrupt_row = 10):
    for i in range(28):
        x = np.random.binomial(n=1, p=0.5, size=(data.shape[0], corrupt_row))
        data[:,28*i : 28*i + corrupt_row] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)

def right_corruption(data, corrupt_row = 10):
    for i in range(28):
        x = np.random.binomial(n=1, p=0.5, size=(data.shape[0], corrupt_row))
        data[:,28*(i+1) - corrupt_row : 28*(i+1)] += x
    binarizer = preprocessing.Binarizer(threshold=0.99)
    data =  binarizer.transform(data)
    return np.array(data)


def reconstruct(activations, W, b, corrupt_row, CD = False):

    for idx in range(n_samples):
        persistent_vis_chain = np.random.binomial(n=1, p= activations, size=activations.shape)

        v_samples = persistent_vis_chain

        for i in range(num_rbm):

            vis_units = hidden_list[num_rbm-i - 1]

            if CD:
                W_sample = W
                b_down = b[:vis_units]
                b_up = b[vis_units:]

            else:
                W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
                b_down = b[num_rbm - i -1 ][:vis_units]
                b_up = b[num_rbm - i -1 ][vis_units:]

            for j in range(plot_every):
                downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
                down_sample1 = np.random.binomial(n=1, p= downact1)
                upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
                v_samples = np.random.binomial(n=1,p=upact1)
            v_samples = down_sample1
        print(' ... plotting sample ', idx)
    return downact1
    # image = Image.fromarray(image_data)
    # image.save(savepath1 + '/recover.eps')


########################import filters and bias parameters # ##################
# path_w = '../LLD/DBM_196_196_64/decay_1e-05/lr_0.001/weight_199.npy'
# path_b = '../LLD/DBM_196_196_64/decay_1e-05/lr_0.001/bias_199.npy'
# savepath1 = '../LLD/Samples/'
#
#
# W = np.load(path_w)
# b = np.load(path_b)
# W = [W[0]]
# b = [b[0]]


hidden_list = [784, 196]


CD = True
path_w = '../rbm_baseline/PCD_10/weights_180.npy'
path_bvis = '../rbm_baseline/PCD_10/bvis_180.npy'
path_bhid = '../rbm_baseline/PCD_10/bhid_180.npy'
savepath1 = '../LLD/Samples/'

W = np.load(path_w)
bvis = np.load(path_bvis)
bhid = np.load(path_bhid)

b = np.concatenate((bvis,bhid))


# hidden_list = [784, 196, 196, 64]

num_rbm = len(hidden_list) -1
n_chains = 8
n_samples = 1
plot_every = 1000
image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
)


def test_error():
    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
    f.close()
    #print(train_set[0][0])
    binarizer = preprocessing.Binarizer(threshold=0.5)
    data =  binarizer.transform(test_set[0])

    ####################select corruption level ##############################

    corrupt_row = 12
    #cor = top_corruption(img_data, corrupt_row=8)
    #cor = top_corruption(img_data, corrupt_row= 8)
    corruption_type = 'top'
    cor = top_corruption(data, corrupt_row= corrupt_row, start=0)

    # corruption_type = 'bottom'
    # cor = bottom_corruption(data, corrupt_row= corrupt_row, start=0)

    # corruption_type = 'left'
    # cor = left_corruption(data, corrupt_row= corrupt_row)

    # corruption_type = 'right'
    # cor = right_corruption(data,corrupt_row=corrupt_row)

    ############################### Draw the results #################
    feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=bhid)
    feed_data = feed_samplor.get_mean_activation(input_data=cor, CD=CD)
    downact1 = reconstruct(activations=feed_data, W=W, b=b, corrupt_row=corrupt_row, CD=CD)    # reconstruct the images give the corruptions

    #############################imputing the missing values  and show reuslts #################################
    if corruption_type is 'top':
        downact1[:,28*corrupt_row:] = test_set[0][:,28*corrupt_row:]

    if corruption_type is 'bottom':
        downact1[:,: 28* (28-corrupt_row)] = test_set[0][:,: 28*(28-corrupt_row)]

    if corruption_type is 'left':
        for i in range(28):
            downact1[:,28*i + corrupt_row: 28* (i+1)] = test_set[0][:,28*i + corrupt_row:28* (i+1)]

    if corruption_type is 'right':
        for i in range(28):
            downact1[:,28*i:28* (i+1)-corrupt_row] = test_set[0][:,28*i:28* (i+1) - corrupt_row]

    print(np.abs(downact1 - test_set[0])[0])
    print( np.sum(np.abs(downact1 - test_set[0])[0]))


    diff = np.sum(np.abs(downact1 - test_set[0])) / test_set[0].shape[0]
    print(diff)
    return diff


diff = 0
for i in range(3):
    diff += test_error()
print(diff/3)









