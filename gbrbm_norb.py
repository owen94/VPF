'''
Try to convert NORB to binary dataset
'''

import matplotlib.pyplot as plt
from PIL import Image
from gbrbm import GBRBM
import numpy as np

# f = open('train_new.bin', "r")
#
# a = np.fromfile(f, dtype=np.uint8)
# print(a.shape)
# k = a[:96*96].reshape(96, 96)
# print(k)
# plt.imshow(k,cmap='Greys')
# plt.show()
# num_samples = int(a.shape[0]/(96*96))
#
# data = np.zeros((num_samples, 96*96))
# for i in range(num_samples):
#     data[i,:] = a[i *96*96:(i+1) * 96*96]

# data_path = 'train.npy'
# b = np.load(data_path)
# a = np.arange(24300) * 2
# new_data = b[a]
# np.save('half_train.npy', new_data)
# data = np.load('half_train.npy')
# # print(c.shape)
# #
# plt.imshow(data[0].reshape(96, 96),cmap='gray')
# plt.show()
# num_samples = data.shape[0]
# ori_size = 96
# cut = 8
# new_size = ori_size - 2*cut
# final_train = np.zeros((num_samples, new_size*new_size))
#
# for i in range(num_samples):
#     a = data[i,:].reshape(ori_size,ori_size)
#     b = a[cut:ori_size-cut, cut:ori_size-cut].ravel()
#     final_train[i] = b
#
# plt.imshow(final_train[50].reshape(new_size, new_size),cmap='gray')
# plt.show()
# np.save('final_train_80*80.npy', final_train)


data = np.load('final_train_80*80.npy')

bbrbm = GBRBM(n_visible=6400, n_hidden=4000, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(data, n_epoches=500, batch_size=20)

print(errs)

