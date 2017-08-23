'''
Try to convert NORB to binary dataset
'''

import matplotlib.pyplot as plt
from PIL import Image
from gbrbm import GBRBM
import numpy as np
from DBM_1 import  *
from sklearn import preprocessing


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
data = preprocessing.scale(data/255)

bbrbm = GBRBM(n_visible=6400, n_hidden=4000, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(data, n_epoches=500, batch_size=20)

#

# ##################  Generate training data with Gaussian Bernoulli RMB #####
#
# def feed_gaussian(data, w, bhid):
# 
#     activation = sigmoid(np.dot(data, w)+  bhid)
#     samples = np.random.binomial(n=1,p=activation,size=activation.shape)
#     return samples
#
# #################  Train the DBM with the input ############################
#
# def train_dbm(data, hidden_list, decay, lr, undirected = False,  batch_sz = 40, epoch = 200):
#
#
#     #########  load the Gaussian RBM data here    ############
#
#
#     num_rbm = len(hidden_list) -1
#     index = T.lscalar()    # index to a mini batch
#     x1 = T.matrix('x1')
#     x2 = T.matrix('x2')
#     x3 = T.matrix('x3')
#
#     if len(hidden_list) == 4:
#
#         if undirected:
#             path = '../DBM_results/Undirected_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         else:
#             path = '../DBM_results/Directed_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         dbm = DBM(hidden_list = hidden_list,
#               input1=x1,
#               input2=x2,
#               input3=x3,
#               batch_sz=batch_sz)
#
#     elif len(hidden_list) ==3:
#         if undirected:
#             path = '../DBM_results/Undirected_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         else:
#             path = '../DBM_results/Directed_DBM/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#
#         dbm = DBM(hidden_list = hidden_list,
#               input1=x1,
#               input2=x2,
#               batch_sz=batch_sz)
#
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     n_train_batches = data.shape[0]//batch_sz
#
#     new_data = []
#
#     for i in range(num_rbm):
#         num_units = hidden_list[i] + hidden_list[i+1]
#         new_data.append(theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
#                                   name = 'train',borrow = True) )
#
#     cost, updates = dbm.get_cost_update(decay_list=decay,learning_rate=lr)
#
#     if len(hidden_list) == 4:
#         train_func = theano.function([index], cost, updates= updates,
#                                  givens= {
#                                      x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
#                                      x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
#                                      x3: new_data[2][index * batch_sz: (index + 1) * batch_sz]
#                                  })
#     elif len(hidden_list) ==3:
#         train_func = theano.function([index], cost, updates= updates,
#                                  givens= {
#                                      x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
#                                      x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
#                                  })
#
#     mean_epoch_error = []
#     start_time = timeit.default_timer()
#
#     for n_epoch in range(epoch):
#
#         ## propup to get the trainning data
#
#         W = []
#         b = []
#         for i in range(num_rbm):
#             W.append(dbm.W[i].get_value(borrow = True))
#             b.append(dbm.b[i].get_value(borrow = True))
#
#         samplor = get_samples(hidden_list= hidden_list, W=W, b = b)
#         forward_act, forward_data = samplor.forward_pass(input_data= data)
#
#         if undirected:
#             undirected_data = samplor.undirected_pass(forward_act = forward_act)
#             for j in range(num_rbm):
#                 new_data[j].set_value(np.asarray(undirected_data[j], dtype=theano.config.floatX))
#         if not undirected:
#             for j in range(num_rbm):
#                 new_data[j].set_value(np.asarray(forward_data[j], dtype=theano.config.floatX))
#
#
#         ## Train the dbm
#         mean_cost = []
#         for batch_index in range(n_train_batches):
#             mean_cost += [train_func(batch_index)]
#         mean_epoch_error += [np.mean(mean_cost)]
#         print('The cost for mpf in epoch %d is %f'% (n_epoch,mean_epoch_error[-1]))
#
#         if int(n_epoch+1) % 50 ==0:
#             filename = path + '/dbm_' + str(n_epoch) + '.pkl'
#             save(filename,dbm)
#
#             W = []
#             b = []
#             for i in range(num_rbm):
#                 W.append(dbm.W[i].get_value(borrow = True))
#                 b.append(dbm.b[i].get_value(borrow = True))
#
#             w_name = path + '/weight_' + str(n_epoch) + '.npy'
#             b_name = path + '/bias_' + str(n_epoch) + '.npy'
#             np.save(w_name,W)
#             np.save(b_name,b)
#
#     loss_savename = path + '/train_loss.eps'
#     show_loss(savename= loss_savename, epoch_error= mean_epoch_error)
#
#     end_time = timeit.default_timer()
#
#     running_time = (end_time - start_time)
#
#     print ('Training took %f minutes' % (running_time / 60.))
#
#
# ################  Generate Samples from the DBM ############################
#
#
# def pre_gaussian_samples(data, hidden_list, W, b):
#
#     ############   Feed forward here #######################
#
#     feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
#     feed_data = feed_samplor.get_mean_activation(input_data= data)
#     feed_mean_activation = np.mean(feed_data, axis=0)
#     num_rbm = len(hidden_list) - 1
#
#     n_chains = 100
#     plot_every = 5
#
#     persistent_vis_chain = np.random.binomial(2, p = feed_mean_activation,size=(n_chains, hidden_list[-1]))
#
#     v_samples = persistent_vis_chain
#
#     for i in range(num_rbm):
#
#         vis_units = hidden_list[num_rbm-i - 1]
#         W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
#         b_down = b[num_rbm - i -1 ][:vis_units]
#         b_up = b[num_rbm - i -1 ][vis_units:]
#
#         for j in range(plot_every):
#             downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
#             down_sample1 = np.random.binomial(n=1, p= downact1)
#             upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
#             v_samples = np.random.binomial(n=1,p=upact1)
#         v_samples = down_sample1
#
#     return downact1, down_sample1
# ############### Generate Data from Gaussian RBM ############################
#
#
# def gaussian_samples(pre_samples, W, bvis, bhid, gibbs_steps):
#
#     for i in range(gibbs_steps):
#         bottom_output = sigmoid()
#
#
#
#     return gaussian_samples
#
#
#
#
# def plot_samples():
#
#
#     ##  run the previous functions all together, plot the final samples
#
#







