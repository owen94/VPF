'''
In this file, we train a deep model which have connections within layers.
'''
from dmpf_optimizer import *
from sklearn import preprocessing
import timeit, pickle, sys, math
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils_mpf import *
from intra_dmpf import asyc_gibbs, one_gibbs
from get_samples import get_samples
from comp_likelihood import  get_ll, gpu_parzen
class Deep_intra(object):

    def __init__(self, hidden_list = [] , temp = 1, batch_sz = 40, input1 = None, input2 = None, input3 = None):

        self.num_rbm = int(len(hidden_list) - 1 )
        self.hidden_list = hidden_list
        self.temp = temp

        self.W = []
        self.b = []

        for i in range(self.num_rbm):
            initial_W = np.asarray(get_intra_mpf_params(self.hidden_list[i],self.hidden_list[i+1]),
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

    def get_intra_update(self, decay_list = [], learning_rate = 0.001):
        updates = []
        cost = 0
        for i in range(self.num_rbm):

            decay = decay_list[i]

            W = self.W[i]
            b = self.b[i]

            z = (1/2 - self.input[i]) / self.temp
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
            a1 = np.zeros((visible_units,visible_units))
            c = np.ones((hidden_units,hidden_units)) - np.diagflat(np.ones(hidden_units))
            zero_grad_u = np.concatenate((a1,a),axis = 1)
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

def train_dbm(hidden_list, decay, lr, temp, n_round =1, feed_first = True,  batch_sz = 40, epoch = 500):

    data = load_mnist()

    # ori_data, labels = read(digits = np.arange(10))
    # data = ori_data/255
    # binarizer = preprocessing.Binarizer(threshold=0.5)
    # data =  binarizer.transform(data)

    num_rbm = len(hidden_list) -1
    index = T.lscalar()    # index to a mini batch
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    x3 = T.matrix('x3')

    if len(hidden_list) == 4:
        path = '../intra_mpf/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '_' + str(hidden_list[3]) + '/decay_' + str(decay[1]) + '/lr_' + \
               str(lr)+ '/temp_' + str(temp) + '/' + str(feed_first)
        deep_intra = Deep_intra(hidden_list = hidden_list,
                temp=temp,
              input1=x1,
              input2=x2,
              input3=x3,
              batch_sz=batch_sz)

    elif len(hidden_list) ==3:
        path = '../intra_mpf/DBM_' + str(hidden_list[1]) + '_' + str(hidden_list[2]) \
               + '/decay_' + str(decay[1]) + '/lr_' + str(lr) + '/temp_' + str(temp) + '/' + str(feed_first)

        deep_intra = Deep_intra(hidden_list = hidden_list,
                         temp=temp,
              input1=x1,
              input2=x2,
              batch_sz=batch_sz)

    if not os.path.exists(path):
        os.makedirs(path)

    n_train_batches = data.shape[0]//batch_sz

    new_data = []

    for i in range(num_rbm):
        num_units = hidden_list[i] + hidden_list[i+1]
        new_data.append(theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                  name = 'train',borrow = True) )

    cost, updates = deep_intra.get_intra_update(decay_list=decay,learning_rate=lr)

    if len(hidden_list) == 4:
        train_func = theano.function([index], cost, updates= updates,
                                 givens= {
                                     x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
                                     x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
                                     x3: new_data[2][index * batch_sz: (index + 1) * batch_sz]
                                 })
    elif len(hidden_list) ==3:
        train_func = theano.function([index], cost, updates= updates,
                                 givens= {
                                     x1: new_data[0][index * batch_sz: (index + 1) * batch_sz],
                                     x2: new_data[1][index * batch_sz: (index + 1) * batch_sz],
                                 })

    mean_epoch_error = []
    start_time = timeit.default_timer()

    train_lld = []
    train_std = []
    test_lld = []
    test_std = []

    feed_first_Data = []

    if not feed_first:
        for i in range(num_rbm):
            rand_h = np.random.binomial(n=1, p=0.5, size = (data.shape[0], hidden_list[i+1]))
            if i == 0:
                feed_first_Data += [np.concatenate((data, rand_h), axis=1)]
            else:
                feed_first_Data += [np.concatenate((feed_first_Data[i-1][:,hidden_list[i-1]:], rand_h), axis=1)]
                assert feed_first_Data[i].shape[1] == hidden_list[i] + hidden_list[i+1]


    for n_epoch in range(epoch):

        ## propup to get the trainning data
        W = []
        b = []

        for i in range(num_rbm):
            W.append(deep_intra.W[i].get_value(borrow = True))
            b.append(deep_intra.b[i].get_value(borrow = True))

        if feed_first:
            samplor = get_samples(hidden_list= hidden_list, W=W, b = b)
            forward_act, forward_data = samplor.forward_pass(input_data= data)
            for i in range(num_rbm):
                assert forward_data[i].shape[1] == hidden_list[i] + hidden_list[i+1]
                sample_data = asyc_gibbs(forward_data[i],W[i],b[i], n_round=n_round,temp=temp,vis_units=hidden_list[i],
                                         hid_units=hidden_list[i+1])
                new_data[i].set_value(np.asarray(sample_data, dtype=theano.config.floatX))

        elif not feed_first:

            for i in range(num_rbm):

                sample_data = asyc_gibbs(feed_first_Data[i], W[i], b[i], n_round=n_round,temp=temp,vis_units=hidden_list[i],
                                         hid_units=hidden_list[i+1])
                new_data[i].set_value(np.asarray(sample_data, dtype=theano.config.floatX))

                feed_first_Data[i] = sample_data

                if i != num_rbm -1 :
                    feed_first_Data[i+1][:,:hidden_list[i+1]] = feed_first_Data[i][:, hidden_list[i]:]

                assert feed_first is False

        ###### Perform gibbs sampling here #######################

        ## Train the dbm
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_func(batch_index)]
        mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (n_epoch,mean_epoch_error[-1]))


        if int(n_epoch+1) >= 200 and int(n_epoch+1) % 5 ==0:

            saveName = path + '/weights_' + str(n_epoch) + '.png'
            tile_shape = (10, hidden_list[1]//10)

            #displayNetwork(W1.T,saveName=saveName)

            filter = deep_intra.W[0].get_value(borrow = True)
            visible_units = hidden_list[0]

            image = Image.fromarray(
                tile_raster_images( X=(filter[:visible_units,visible_units:]).T,
                        img_shape=(28, 28),
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                    )
            image.save(saveName)

            w_name = path + '/weight_' + str(n_epoch) + '.npy'
            b_name = path + '/bias_' + str(n_epoch) + '.npy'
            np.save(w_name,W)
            np.save(b_name,b)

        # if int(n_epoch+1) % 20 ==0:
        #
        #     n_chains = 8
        #     n_samples = 8
        #     plot_every = 3
        #     image_data = np.zeros(
        #         (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
        #     )
        #
        #     feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
        #     feed_data = feed_samplor.get_mean_activation(input_data= data)
        #
        #     feed_mean_activation = np.mean(feed_data, axis=0)
        #
        #     for idx in range(n_samples):
        #         #persistent_vis_chain = np.random.randint(2,size=(n_chains, hidden_list[-1]))
        #         feed_initial = np.random.binomial(n=1, p= feed_mean_activation, size=(n_chains, hidden_list[-1]))
        #
        #         v_samples = feed_initial
        #
        #         for i in range(num_rbm):
        #
        #             vis_units = hidden_list[num_rbm-i - 1]
        #             W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
        #             b_down = b[num_rbm - i -1 ][:vis_units]
        #             b_up = b[num_rbm - i -1 ][vis_units:]
        #
        #             for j in range(plot_every):
        #                 downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
        #                 down_sample1 = np.random.binomial(n=1, p= downact1)
        #                 upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
        #                 v_samples = np.random.binomial(n=1,p=upact1)
        #
        #                 x = np.concatenate((down_sample1,v_samples),axis=1)
        #                 v_samples = mix_in(x=x,w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=1)[:,vis_units:]
        #
        #             v_samples = down_sample1
        #         print(' ... plotting sample ', idx)
        #
        #         image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        #             X= downact1,
        #             img_shape=(28, 28),
        #             tile_shape=(1, n_chains),
        #             tile_spacing=(1, 1)
        #         )
        #
        #     image = Image.fromarray(image_data)
        #     image.save(path + '/samples_' + str(n_epoch) + '.png')
        #
        #
        # if int(n_epoch+1) % 20 ==0:
        #
        #     n_chains = 8
        #     n_samples = 8
        #     plot_every = 3
        #     image_data_2 = np.zeros(
        #         (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
        #     )
        #
        #     feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
        #     feed_data = feed_samplor.get_mean_activation(input_data= data)
        #
        #     feed_mean_activation = np.mean(feed_data, axis=0)
        #
        #     for idx in range(n_samples):
        #         #persistent_vis_chain = np.random.randint(2,size=(n_chains, hidden_list[-1]))
        #         feed_initial = np.random.binomial(n=1, p= feed_mean_activation, size=(n_chains, hidden_list[-1]))
        #
        #         v_samples = feed_initial
        #
        #         for i in range(num_rbm):
        #
        #             vis_units = hidden_list[num_rbm-i - 1]
        #             W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
        #             b_down = b[num_rbm - i -1 ][:vis_units]
        #             b_up = b[num_rbm - i -1 ][vis_units:]
        #
        #             for j in range(plot_every):
        #                 downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
        #                 down_sample1 = np.random.binomial(n=1, p= downact1)
        #                 upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
        #                 v_samples = np.random.binomial(n=1,p=upact1)
        #             v_samples = down_sample1
        #         print(' ... plotting sample ', idx)
        #
        #         image_data_2[29 * idx:29 * idx + 28, :] = tile_raster_images(
        #             X= downact1,
        #             img_shape=(28, 28),
        #             tile_shape=(1, n_chains),
        #             tile_spacing=(1, 1)
        #         )
        #
        #     image = Image.fromarray(image_data_2)
        #     image.save(path + '/nomix_samples_' + str(n_epoch) + '.png')

        if int(n_epoch+1) >= 200 and int(n_epoch+1)% 5 == 0:
            W = []
            b = []
            for i in range(num_rbm):
                W.append(deep_intra.W[i].get_value(borrow = True))
                b.append(deep_intra.b[i].get_value(borrow = True))
            dataset = 'mnist.pkl.gz'
            f = gzip.open(dataset, 'rb')
            train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
            f.close()

            binarizer = preprocessing.Binarizer(threshold=0.5)
            training_data =  binarizer.transform(train_set[0])
            test_data = test_set[0]
            train_data = train_set[0]

            ##############################################################################
            n_sample = 10000
            plot_every = 3
            ################################################################################
            # for i in range(num_rbm):
            #     feed_vis_units = hidden_list[i]
            #     feed_w = W[i][:feed_vis_units,feed_vis_units:]
            #     feed_b = b[i][feed_vis_units:]
            #     feed_data = sigmoid(np.dot(feed_data, feed_w) + feed_b)
            error_bar_lld = []
            error_bar_std = []

            #for kk in range(1):

            feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
            feed_data = feed_samplor.get_mean_activation(input_data= training_data)

            feed_mean_activation = np.mean(feed_data, axis=0)
            feed_initial = np.random.binomial(n=1, p= feed_mean_activation, size=(n_sample, hidden_list[-1]))
            ###########################################################

            ######### generate the parzen sample to compute the model distribution ###########
            v_samples = feed_initial
            for i in range(num_rbm):
                vis_units = hidden_list[num_rbm-i - 1]
                W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
                b_down = b[num_rbm - i -1 ][:vis_units]
                b_up = b[num_rbm - i -1 ][vis_units:]

                for j in range(plot_every):
                    downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
                    down_sample1 = np.random.binomial(n=1, p= downact1)
                    upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
                    v_samples = np.random.binomial(n=1,p=upact1)

                    x = np.concatenate((down_sample1,v_samples),axis=1)
                    v_samples = mix_in(x=x,vis_units=vis_units,
                                       w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=1)[:,vis_units:]


                v_samples = down_sample1

            parzen_sample = downact1
            # compute the log-likelihood for the training data
            # epoch_train_lld = get_ll(x=train_data[:1],
            #                          gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=20)
            # train_mean_lld = np.mean(np.array(epoch_train_lld))
            # train_std_lld = np.std(np.array(epoch_train_lld))
            # train_lld += [train_mean_lld]
            # train_std += [train_std_lld]
            train_mean_lld = 0

            # comppute the log-likelihood for the test data
            epoch_test_lld = get_ll(x=test_data, gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=10)
            test_mean_lld = np.mean(np.array(epoch_test_lld))
            test_std_lld = np.std(np.array(epoch_test_lld))
            test_lld += [test_mean_lld]
            test_std += [test_std_lld]

            print('The loglikehood in epoch {} is: train {}, test {}'.format(n_epoch, train_mean_lld, test_mean_lld))

    path_1 = path + '/train_lld.npy'
    path_2 = path + '/train_std.npy'
    path_3 = path + '/test_lld.npy'
    path_4 = path + '/test_std.npy'


    np.save(path_1, train_lld)
    np.save(path_2, train_std)
    np.save(path_3, test_lld)
    np.save(path_4, test_std)

    print('...............................................')
    print(train_lld)
    print('...............................................')
    print(test_lld)

    loss_savename = path + '/train_loss.eps'
    show_loss(savename= loss_savename, epoch_error= mean_epoch_error)

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))

    ###  generate samples ##########################

if __name__ == '__main__':


    learning_rate_list = [0.001]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    hidden_units_list = [[784, 196, 196, 64]]
    n_samples_list = [1]
    beta_list = [0]
    sparsity_list = [0]
    batch_list = [40]
    temp_list = [2]
    decay_list = [[0.0001, 0.0001, 0.0001, 0.0001]]
    feed_list = [True]

    undirected_list = [False]
    for undirected in undirected_list:
        for learning_rate in learning_rate_list:
            for hidden_list in hidden_units_list:
                for decay in decay_list:
                    for temp in temp_list:
                        for feed in feed_list:
                            train_dbm(hidden_list=hidden_list,decay=decay,
                                      lr=learning_rate, feed_first=feed, temp=temp)
