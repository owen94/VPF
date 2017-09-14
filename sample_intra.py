'''
This file test the generating performance of deep intra MPF.

Sample the intra_dbm can contain several steps, for example,
'''

from utils_mpf import *
from get_samples import get_samples
from PIL import Image


def mix_in_hidden(x, vis_units,  w, b, temp, mix = 1):
    hid_units = x.shape[1] - vis_units
    h_data = x[:,vis_units:] # only the hidden part data
    input_w = w[vis_units:,vis_units:]
    #assert dddjust
    input_b = b[vis_units:]
    for j in range(mix):
        for i in range(hid_units):
            act = sigmoid(1/temp * (np.dot(h_data, input_w[:,i]) + input_b[i]))
            h_i = np.random.binomial(n=1, p= act, size = act.shape)
            h_data[:, i] = h_i

    x[:, vis_units:] =  h_data
    return x

# path_w = '../intra_mpf/DBM_196_196_64_hidden_not_symmetric/decay_0.0001/lr_0.001/temp_1/True/weight_499.npy'
# path_b = '../intra_mpf/DBM_196_196_64_hidden_not_symmetric/decay_0.0001/lr_0.001/temp_1/True/bias_499.npy'
# savepath1 = '../intra_mpf/Samples/'

path_w = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/weight_499.npy'
path_b = '../intra_mpf/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/bias_499.npy'
savepath1 = '../intra_mpf/Samples/'

# path_w = '../intra_mpf/Fashion/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/weight_999.npy'
# path_b = '../intra_mpf/Fashion/DBM_196_196_64/decay_0.0001/lr_0.001/temp_1/True/bias_999.npy'
# savepath1 = '../intra_mpf/Samples/'

W = np.load(path_w)
b = np.load(path_b)
hidden_list = [784, 196, 196, 64]

num_rbm = len(hidden_list) -1

n_chains = 8
n_samples = 8
plot_every = 3

temp = 1

image_data = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
)

image_data_2 = np.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
)


#
# ori_data, labels = read(digits = np.arange(10))
# #print(labels)
# data = ori_data/255
# binarizer = preprocessing.Binarizer(threshold=0.5)
# training_data =  binarizer.transform(data)


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

binarizer = preprocessing.Binarizer(threshold=0.5)
training_data =  binarizer.transform(train_set[0])
train_data = test_set[0]

feed_samplor = get_samples(hidden_list=hidden_list, W=W, b=b)
feed_data = feed_samplor.get_mean_activation(input_data= training_data)

feed_mean_activation = np.mean(feed_data, axis=0)



for idx in range(n_samples):

    persistent_vis_chain = np.random.binomial(n=1, p= feed_mean_activation, size=(n_chains, hidden_list[-1]))
    #persistent_vis_chain = np.random.binomial(n=1, p= 0.5, size=(n_chains, hidden_list[-1]))


    v_samples = persistent_vis_chain

    for i in range(num_rbm):

        vis_units = hidden_list[num_rbm-i - 1]
        W_sample = W[num_rbm - i -1 ][:vis_units,vis_units:]
        b_down = b[num_rbm - i -1 ][:vis_units]
        b_up = b[num_rbm - i -1 ][vis_units:]

        # if there is no feedforward, maybe we need to mixin the hidden states
        #  independent of visible states firstly
        # for several steps

        for j in range(plot_every):
            downact1 = sigmoid(np.dot(v_samples,W_sample.T) + b_down )
            down_sample1 = np.random.binomial(n=1, p= downact1)
            upact1 = sigmoid(np.dot(down_sample1,W_sample)+b_up)
            v_samples = np.random.binomial(n=1,p=upact1)
            #
            x = np.concatenate((down_sample1,v_samples),axis=1)
            v_samples = mix_in(x=x, vis_units= vis_units,
                               w=W[num_rbm - i -1 ],b=b[num_rbm - i -1 ], temp=temp, mix=1)[:,vis_units:]


        v_samples = down_sample1
    print(' ... plotting sample ', idx)

    image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        X= downact1,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

image = Image.fromarray(image_data)
image.save(savepath1 + '/samples.pdf')





