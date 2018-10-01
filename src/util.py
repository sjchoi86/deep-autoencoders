import warnings,gym
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def kernel_se(_X1,_X2,_hyp={'gain':1,'len':1,'noise':1e-8}):
    hyp_gain = float(_hyp['gain'])**2
    hyp_len  = 1/float(_hyp['len'])
    pairwise_dists = cdist(_X2,_X2,'euclidean')
    K = hyp_gain*np.exp(-pairwise_dists ** 2 / (hyp_len**2))
    return K

def kdpp(_X,_k):
    # Select _n samples out of _X using K-DPP
    n,d = _X.shape[0],_X.shape[1]
    mid_dist = np.median(cdist(_X,_X,'euclidean'))
    out,idx = np.zeros(shape=(_k,d)),[]
    for i in range(_k):
        if i == 0:
            rand_idx = np.random.randint(n)
            idx.append(rand_idx) # append index
            out[i,:] = _X[rand_idx,:] # append  inputs
        else:
            det_vals = np.zeros(n)
            for j in range(n):
                if j in idx:
                    det_vals[j] = -np.inf
                else:
                    idx_temp = idx.copy()
                    idx_temp.append(j)
                    X_curr = _X[idx_temp,:]
                    K = kernel_se(X_curr,X_curr,{'gain':1,'len':mid_dist,'noise':1e-4})
                    det_vals[j] = np.linalg.det(K)
            max_idx = np.argmax(det_vals)
            idx.append(max_idx)
            out[i,:] = _X[max_idx,:] # append  inputs
    return out,idx

def remove_warnings():
    gym.logger.set_level(40)
    warnings.filterwarnings("ignore") 
    tf.logging.set_verbosity(tf.logging.ERROR)

def numpy_setting():
    np.set_printoptions(precision=3)
    
def get_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data', one_hot=True)
    return mnist

def gpu_sess():
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

def plot_imgs(_imgs,_imgSz=(28,28),_nR=1,_nC=10,_figsize=(15,2),
            _title=None,_titles=None,_tfs=15,
            _wspace=0.05,_hspace=0.05):
    nr,nc = _nR,_nC
    fig = plt.figure(figsize=_figsize)
    if _title is not None:
        fig.suptitle(_title, size=15)
    gs  = gridspec.GridSpec(nr,nc)
    gs.update(wspace=_wspace, hspace=_hspace)
    for i, img in enumerate(_imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if len(img.shape) == 1:
            img = np.reshape(img,newshape=_imgSz) 
        plt.imshow(img,cmap='Greys_r',interpolation='none')
        plt.clim(0.0, 1.0)
        if _titles is not None:
            plt.title(_titles[i],size=_tfs)
    plt.show() 