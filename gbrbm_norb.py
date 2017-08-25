'''
Try to convert NORB to binary dataset
'''

import matplotlib.pyplot as plt
from PIL import Image
from gbrbm import GBRBM
import numpy as np
from utils_mpf import *
from theano_optimizers import Adam
from theano.tensor.shared_randomstreams import RandomStreams

#from DBM import DBM
from get_samples import get_samples
from sklearn import preprocessing
import os, timeit

class DBM(object):

    def __init__(self, hidden_list = [] , batch_sz = 40, input1 = None, input2 = None, input3 = None):

        self.num_rbm = int(len(hidden_list) - 1 )
        self.hidden_list = hidden_list

        self.W = []
        self.b = []

        for i in range(self.num_rbm):
            initial_W = np.asarray(get_mpf_params(self.hidden_list[i],self.hidden_list[i+1]),
                dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

            self.W.append(W)

            num_neuron = hidden_list[i] + hidden_list[i+1]
            b = theano.shared(value=np.zeros(num_neuron,dtype=theano.config.floatX),
                name='bias',borrow=True)
            self.b.append(b)

        numpy_rng = np.random.RandomState(1233456)

        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.epsilon = 0.01

        self.batch_sz = batch_sz

        self.input1 = input1
        self.input2 = input2
        self.input3 = input3

        if len(hidden_list) == 4:
            self.input = [self.input1, self.input2, self.input3]
        elif len(hidden_list) == 3:
            self.input = [self.input1, self.input2]



    def get_cost_update(self,decay_list = [], learning_rate = 0.001):
        '''
        :param undirected_act: The input from the forward and backward pass
        :return: the cost function and the updates
        '''
        updates = []
        cost = 0
        for i in range(self.num_rbm):

            decay = decay_list[i]

            W = self.W[i]
            b = self.b[i]

            z = 1/2 - self.input[i]
            energy_difference = z * (T.dot(self.input[i], W)+ b.reshape([1,-1]))

            cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))
            cost_weight = 0.5 * decay * T.sum(W**2)
            cost += cost_weight

            h = z * T.exp(energy_difference)
            W_grad = (T.dot(h.T,self.input[i])+T.dot(self.input[i].T,h))*self.epsilon/self.batch_sz
            b_grad = T.mean(h,axis=0)*self.epsilon
            decay_grad = decay*W
            W_grad += decay_grad

            visible_units = self.hidden_list[i]
            hidden_units = self.hidden_list[i+1]
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.zeros((hidden_units,hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)
            zero_grad = theano.shared(value=np.asarray(zero_grad,dtype=theano.config.floatX),
                                           name='zero_grad',borrow = True)
            W_grad *= zero_grad
            grads = [W_grad,b_grad]

            params = [self.W[i],self.b[i]]

            update_rbm = Adam(grads=grads, params=params,lr=learning_rate)
            updates += update_rbm

        return cost, updates




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


data = np.load('../LLD/final_train_80*80.npy')
data = preprocessing.scale(data)
# print(data[0])
# plt.imshow(data[50].reshape(80, 80),cmap='gray')
# plt.savefig('../LLD/gaussian_dbm/original.eps')

gbrbm = GBRBM(n_visible=6400, n_hidden=4000, learning_rate=0.01, momentum=0.95, use_tqdm=True,
              sample_visible=True, sigma=1)

errs = gbrbm.fit(data, n_epoches=500, batch_size=20)


# #
#
# # ##################  Generate training data with Gaussian Bernoulli RMB #####
# #
# def feed_gaussian(data, w, bhid):
#
#     activation = sigmoid(np.dot(data, w)+  bhid)
#     samples = np.random.binomial(n=1,p=activation,size=activation.shape)
#     return samples
#
# #################  Train the DBM with the input ############################
#
# def train_gdbm(data, hidden_list, decay, lr, undirected = False,  batch_sz = 40, epoch = 200):
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
#             path = '../LLD/gaussian_dbm/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         else:
#             path = '../LLD/gaussian_dbm/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         dbm = DBM(hidden_list = hidden_list,
#               input1=x1,
#               input2=x2,
#               input3=x3,
#               batch_sz=batch_sz)
#
#     elif len(hidden_list) ==3:
#         if undirected:
#             path = '../LLD/gaussian_dbm/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
#                + '/decay_' + str(decay[1]) + '/lr_' + str(lr)
#         else:
#             path = '../LLD/gaussian_dbm/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
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
#         if int(n_epoch+1) % 10 ==0:
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
# def pre_gaussian_samples(data, hidden_list, W, b, plot_every = 5, num_samples = 100):
#
#     ############   Feed forward here #######################
#
#     feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
#     feed_data = feed_samplor.get_mean_activation(input_data= data)
#     feed_mean_activation = np.mean(feed_data, axis=0)
#     num_rbm = len(hidden_list) - 1
#
#     n_chains = num_samples
#     plot_every = plot_every
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
#         bottom_output = (np.dot(pre_samples, W.T) + bvis)
#         top_output = sigmoid(np.dot(bottom_output, W) + bhid)
#         pre_samples = np.random.binomial(n=1, p=top_output,size=top_output.shape)
#
#     return bottom_output
#
#
# def _run():
#     W = np.load('../LLD/gaussian/gaussian_w_499.npy')
#     bhid = np.load('../LLD/gaussian/gaussian_bhid_499.npy')
#     ori_data = np.load('../LLD/final_train_80*80.npy')
#
#
#     binary_data = feed_gaussian(data=ori_data,w=W,bhid=bhid)
#     print(binary_data.shape)
#
#     hidden_list = [4000, 2048, 1024, 512]
#     decay = [0.0001, 0.0001, 0.0001, 0.0001]
#     lr = 0.001
#
#     train_gdbm(data=binary_data,hidden_list=hidden_list,decay=decay,lr=lr, batch_sz=20, epoch=200)
#
# def _sample_dbm():
#     gW = np.load('../LLD/gaussian/gaussian_w_499.npy')
#     gbhid = np.load('../LLD/gaussian/gaussian_bhid_499.npy')
#     ori_data = np.load('../LLD/final_train_80*80.npy')
#     gb_vis = np.load('../LLD/gaussian/gaussian_bvis_499.npy')
#
#     dW = np.load('../LLD/gaussian_dbm/weight_39.npy')
#     db = np.load('../LLD/gaussian_dbm/bias_39.npy')
#
#     hidden_list = [4000, 2048, 1024, 512]
#
#     binary_data = feed_gaussian(data=ori_data, w=gW,bhid=gbhid)
#
#     print(binary_data.shape)
#
#     downact1, pre_samples = pre_gaussian_samples(binary_data,hidden_list,dW,db, plot_every=5, num_samples=10)
#
#     savepath_dbm = '../LLD/gaussian_dbm/dbm_outut.npy'
#     np.save(savepath_dbm, pre_samples)
#
#     return savepath_dbm
#
# def _samples_norb(savepath_dbm, gibbs_steps = 100):
#
#     pre_samples = np.load(savepath_dbm)
#     gW = np.load('../LLD/gaussian/gaussian_w_499.npy')
#     gbhid = np.load('../LLD/gaussian/gaussian_bhid_499.npy')
#     gb_vis = np.load('../LLD/gaussian/gaussian_bvis_499.npy')
#
#     ori_data = np.load('../LLD/final_train_80*80.npy')
#     pre_samples = feed_gaussian(data=ori_data, w=gW,bhid=gbhid)[:1,:]
#     print(pre_samples.shape)
#
#     g_samples = gaussian_samples(pre_samples,gW,gb_vis,gbhid,gibbs_steps=gibbs_steps)
#
#     savepath = '../LLD/gaussian_dbm/generated_samples.npy'
#     np.save(savepath, g_samples)
#     plt.imshow(g_samples[0].reshape(80, 80),cmap='gray')
#     plt.show()
#
#
# #_sample_dbm()
# #
# savepath_dbm = '../LLD/gaussian_dbm/dbm_outut.npy'
#
# _samples_norb(savepath_dbm, gibbs_steps=1000)
#
# savepath = '../LLD/gaussian_dbm/generated_samples.npy'
#
# g_samples = np.load(savepath)
# #g_samples = preprocessing.scale(g_samples)
# print(np.min(g_samples[0]))
# print(g_samples.shape)
# plt.imshow((g_samples[0].reshape(80, 80)),cmap='gray')
# plt.savefig('../LLD/gaussian_dbm/samples.eps')
    ##  run the previous functions all together, plot the final samples









