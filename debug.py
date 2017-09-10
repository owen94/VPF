

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_mpf import read
#from DBM import get_samples
from PIL import Image

# data, labels = read(digits = np.arange(10))
# data = data / 255
# print(np.max(data[0]))
# binarizer = preprocessing.Binarizer(threshold=0.5)
# data =  binarizer.transform(data)
# plt.imshow(data[0].reshape(28,28))
# plt.savefig('../LLD/fashion.png')
# print(data.shape)

# a = [221.70252617454528, 221.35000822029113, 221.73747456169127, 221.68824326629638, 222.4365290294647,
#      222.2857981754303, 221.0884706817627, 221.85699545707703, 221.9257755783081, 222.16784493026734]
#
# b = np.mean(np.array(a))
# c = np.std(np.array(a))
#
#
# d = [220.05687969779967, 220.80221944618225, 221.01207336921692, 221.01996198539734, 220.69540065422058,
#      220.72141383190154, 220.6366689147949, 221.33020700283052, 221.2608075279236, 220.8458219280243]
# b = np.mean(np.array(d))
# c = np.std(np.array(d))
# print(b)
# print(c)


#path_w = '../intra_mpf/DBM_196_196_64_hidden_not_symmetric/decay_0.0001/lr_0.001/temp_1/True/weight_99.npy'
path_w = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/weight_109.npy'
path_b = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/bias_'

w = np.load(path_w)

print(w[2][198, 198])
print(w[2][196, 199])
print(w[2][199, 196])

print(  np.sum(np.abs(w[0]-  w[0].T)))
