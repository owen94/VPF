

import numpy as np
import gzip, pickle, random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_mpf import read
#from DBM import get_samples
from PIL import Image

data, labels = read(digits = np.arange(10))
data = data / 255
print(np.max(data[0]))
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(data)
plt.imshow(data[0].reshape(28,28))
plt.savefig('../LLD/fashion.png')
print(data.shape)