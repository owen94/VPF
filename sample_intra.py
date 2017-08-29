'''
This file test the generating performance of deep intra MPF.

Sample the intra_dbm can contain several steps, for example,
'''

from utils_mpf import *


def mix_in(x, w, b, temp, mix = 1):
    vis_units = 784
    hid_units = x.shape[1] - vis_units
    for j in range(mix):
        for i in range(hid_units):
            input_w = w[:,vis_units + i]
            input_b = b[vis_units + i]
            act = sigmoid( 1/temp * (np.dot(x,input_w) + input_b))
            h_i = np.random.binomial(n=1, p = act,size=act.shape)
            x[:,vis_units + i] = h_i
    return x

