from utils_mpf import *
from get_samples import get_samples
from PIL import Image
from intra_dmpf import asyc_gibbs
from comp_likelihood import get_ll, gpu_parzen


path_w = '../intra_mpf/DBM_196_196_64_hidden_not_symmetric/decay_0.0001/lr_0.001/temp_1/True/weight_499.npy'
path_b = '../intra_mpf/DBM_196_196_64_hidden_not_symmetric/decay_0.0001/lr_0.001/temp_1/True/bias_499.npy'
savepath1 = '../intra_mpf/Samples/'


W = np.load(path_w)
b = np.load(path_b)
hidden_list = [784, 196, 196, 64]

num_rbm = len(hidden_list) -1

n_chains = 8
n_samples = 8
plot_every = 3

temp = 1
n_round = 1

image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

image_data_2 = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')


feed_first_Data = []

dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

binarizer = preprocessing.Binarizer(threshold=0.5)
training_data =  binarizer.transform(train_set[0])
test_data = test_set[0]

for idx in range(1):

    #index = np.random.randint(training_data.shape[0], size=n_chains)

    data = training_data[:10000,:]

    for i in range(num_rbm):
        rand_h = np.random.binomial(n=1, p=0.5, size = (data.shape[0], hidden_list[i+1]))
        if i == 0:
            feed_first_Data += [np.concatenate((data, rand_h), axis=1)]
        else:
            feed_first_Data += [np.concatenate((feed_first_Data[i-1][:,hidden_list[i-1]:], rand_h), axis=1)]
            assert feed_first_Data[i].shape[1] == hidden_list[i] + hidden_list[i+1]

    for i in range(num_rbm):
        sample_data = asyc_gibbs(feed_first_Data[i], W[i], b[i], n_round=n_round,temp=temp,vis_units=hidden_list[i],
                                             hid_units=hidden_list[i+1])
        feed_first_Data[i] = sample_data
        if i != num_rbm -1 :
            feed_first_Data[i+1][:,:hidden_list[i+1]] = feed_first_Data[i][:, hidden_list[i]:]

    v_samples = feed_first_Data[-1][:,-hidden_list[-1]:]
# print(v_samples.shape)

    for i in range(num_rbm):

        vis_units = hidden_list[num_rbm-i - 1]
        W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
        b_down = b[num_rbm - i -1 ][:vis_units]
        b_up = b[num_rbm - i -1 ][vis_units:]

        # for j in range(plot_every):
        #     x = feed_first_Data[num_rbm - i - 1]
        #     v_samples = mix_in(x=x, vis_units= vis_units,
        #                        w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=1)
        #     feed_first_Data[num_rbm - i - 1] = v_samples
        #     print(v_samples.shape)
        for j in range(plot_every):
            downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
            down_sample1 = np.random.binomial(n=1, p= downact1)
            upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
            v_samples = np.random.binomial(n=1,p=upact1)
            # #
            x = np.concatenate((down_sample1,v_samples),axis=1)
            v_samples = mix_in(x=x, vis_units= vis_units,
                               w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=5)[:,vis_units:]


        v_samples = down_sample1


    parzen_sample = downact1
    # comppute the log-likelihood for the test data
    epoch_test_lld = get_ll(x=test_data, gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=10)
    test_mean_lld = np.mean(np.array(epoch_test_lld))
    #test_std_lld = np.std(np.array(epoch_test_lld))
    print('The loglikehood in  is: test {}'.format(test_mean_lld))