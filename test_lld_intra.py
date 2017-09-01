'''
this file will test the lld for the intra_layer MPF
'''

from utils_mpf import *
from get_samples import  get_samples
from comp_likelihood import get_ll, gpu_parzen

def test_lld_intra(path_w, path_b, plot_every, mix_steps, random_initial, temp = 1):

    W = np.load(path_w)
    b = np.load(path_b)
    hidden_list = [784, 196, 196, 64]

    num_rbm = len(hidden_list) -1


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
    plot_every = plot_every
    ################################################################################

    #for kk in range(2):
    if random_initial:
        feed_initial = np.random.binomial(n=1, p= 0.5, size=(n_sample, hidden_list[-1]))

    else:
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
            v_samples = mix_in(x=x,vis_units= vis_units,
                               w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=mix_steps)[:,vis_units:]

        v_samples = down_sample1

    parzen_sample = downact1


    # comppute the log-likelihood for the test data
    epoch_test_lld = get_ll(x=test_data, gpu_parzen=gpu_parzen(mu=parzen_sample,sigma=0.2),batch_size=10)
    test_mean_lld = np.mean(np.array(epoch_test_lld))
    #test_std_lld = np.std(np.array(epoch_test_lld))
    print('The loglikehood in  is: test {}'.format(test_mean_lld))
    return test_mean_lld



plot_list = [3]
mix_lis = [1]
random_list = [True, False]
path_list = [499]
path_w = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/weight_'
path_b = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/bias_'
save_path = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/'


test_lld = []
test_std = []
for n_iter in path_list:
    iter_lld = []
    for i in range(10):
        new_path_w = path_w + str(n_iter) + '.npy'
        new_path_b = path_b + str(n_iter) + '.npy'
        lld_1 =  test_lld_intra(path_w=new_path_w, path_b=new_path_b, plot_every=3, mix_steps=1, random_initial= True)

        iter_lld += [lld_1]
    test_lld += np.mean(iter_lld)
    test_std += np.std(iter_lld)

print(test_lld)
print(test_lld)
savename_1 = save_path + 'check_lld.npy'
savename_2 = save_path + 'check_std.npy'
np.save(savename_1, test_lld)
np.save(savename_2, test_std)
