import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util import kdpp,get_mnist,remove_warnings,gpu_sess,plot_imgs

class gmm_sampler_class(object):
    def __init__(self,_name='gmm',_z_dim=16,_k=5,_var=0.05):
        self.name = _name
        self.z_dim = _z_dim
        self.k = _k
        # Determine k means within [-1~+1]
        self.mu,_ = kdpp(_X=2*np.random.rand(1000,self.z_dim)-1,_k=self.k)
        # Fix variance of each dim to be 0.1
        self.var = _var*np.ones(shape=(self.k,self.z_dim))
    def sample(self,_n):
        samples = np.zeros(shape=(_n,self.z_dim))
        for i in range(_n):
            k = np.random.randint(low=0,high=self.k) # mixture
            mu_k = self.mu[k,:]
            var_k = self.var[k,:]
            samples[i,:] = mu_k + np.sqrt(var_k)*np.random.randn(self.z_dim)
        return samples
    def plot(self,_n=1000,_title='Samples',_tfs=18):
        samples = self.sample(_n=_n)
        plt.figure(figsize=(6,6))
        plt.plot(samples[:,0],samples[:,1],'k.')
        plt.xlim(-2.0,2.0); plt.ylim(-2.0,2.0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(_title,fontsize=_tfs)
        plt.show()
        
tfd = tf.contrib.distributions
tfrni = tf.random_normal_initializer
tfci = tf.constant_initializer
tfrui = tf.random_uniform_initializer
tfscewl = tf.nn.sigmoid_cross_entropy_with_logits
class AAE_class(object):
    def __init__(self,_name='aae',_x_dim=784,_z_dim=16,
                 _h_dims_Q=[64,16],_h_dims_P=[64,16],_h_dims_D=[64,16],
                 _actv_Q=tf.nn.relu,_actv_P=tf.nn.relu,_actv_D=tf.nn.relu,
                 _l2_reg_coef=1e-4,
                 _opmz=tf.train.RMSPropOptimizer,_lr=1e-3,
                 _sess=None,_seed=0,
                 _VERBOSE=True):
        self.name = _name
        self.x_dim = _x_dim
        self.z_dim = _z_dim
        self.h_dims_Q = _h_dims_Q
        self.h_dims_P = _h_dims_P
        self.h_dims_D = _h_dims_D
        self.actv_Q = _actv_Q
        self.actv_P = _actv_P
        self.actv_D = _actv_D
        self.l2_reg_coef = _l2_reg_coef
        self.opmz = _opmz
        self.lr = _lr
        self.sess = _sess
        self.seed = _seed
        self.VERBOSE = _VERBOSE
        # Define sampler
        self.sampler = gmm_sampler_class(_z_dim=self.z_dim,_k=5,_var=0.02)
        if self.VERBOSE:
            self.sampler.plot()
        # Build graph
        self._build_graph()
        # Check parameters
        self._check_params()
        
    # Build graph
    def _build_graph(self):
        with tf.variable_scope(self.name,reuse=False) as scope:
            # Placeholders
            self.x_real = tf.placeholder(shape=[None,self.x_dim],dtype=tf.float32,name='x') # [n x x_dim]
            self.z_sample = tf.placeholder(shape=[None,self.z_dim],dtype=tf.float32,name='z') # [n x z_dim]
            self.kp = tf.placeholder(shape=[],dtype=tf.float32,name='kp') # [1]
            
            # Encoder netowrk Q(z|x): x_real => z_real
            with tf.variable_scope('Q',reuse=False):
                self.net = self.x_real
                for h_idx,hid in enumerate(self.h_dims_Q):
                    self.net = tf.layers.dense(self.net,hid,activation=self.actv_Q,
                                               kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                               name='hid_Q_%d'%(h_idx))
                    self.net = tf.layers.dropout(self.net, rate=self.kp)
                self.z_real = tf.layers.dense(self.net,self.z_dim,activation=None,
                                          kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                          name='z_real') # [n x z_dim]
                
            # Decoder network P(x|z): z_real => x_recon 
            with tf.variable_scope('P',reuse=False):
                self.net = self.z_real
                for h_idx,hid in enumerate(self.h_dims_P):
                    self.net = tf.layers.dense(self.net,hid,activation=self.actv_P,
                                               kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                               name='hid_P_%d'%(h_idx))
                    self.net = tf.layers.dropout(self.net, rate=self.kp)
                self.x_recon = tf.layers.dense(self.net,self.x_dim,activation=None,
                                          kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                          name='x_recon') # [n x x_dim]
                
            # Decoder network P(x|z): z_sample => x_sample 
            with tf.variable_scope('P',reuse=True):
                self.net = self.z_sample
                for h_idx,hid in enumerate(self.h_dims_P):
                    self.net = tf.layers.dense(self.net,hid,activation=self.actv_P,
                                               kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                               name='hid_P_%d'%(h_idx))
                    self.net = tf.layers.dropout(self.net, rate=self.kp)
                self.x_sample = tf.layers.dense(self.net,self.x_dim,activation=None,
                                          kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                          name='x_recon') # [n x x_dim]
                
            # Discriminator D(z): z_real => d_real
            with tf.variable_scope('D',reuse=False):
                self.net = self.z_real
                for h_idx,hid in enumerate(self.h_dims_D):
                    self.net = tf.layers.dense(self.net,hid,activation=self.actv_D,
                                               kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                               name='hid_D_%d'%(h_idx))
                    self.net = tf.layers.dropout(self.net, rate=self.kp)
                self.d_real_logits = tf.layers.dense(self.net,1,activation=None,
                                              kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                              name='d_logits') # [n x 1]
                self.d_real = tf.sigmoid(self.d_real_logits,name='d') # [n x 1]
            
            # Discriminator D(z): 
            with tf.variable_scope('D',reuse=True):
                self.net = self.z_sample
                for h_idx,hid in enumerate(self.h_dims_D):
                    self.net = tf.layers.dense(self.net,hid,activation=self.actv_D,
                                               kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                               name='hid_D_%d'%(h_idx))
                    self.net = tf.layers.dropout(self.net, rate=self.kp)
                self.d_fake_logits = tf.layers.dense(self.net,1,activation=None,
                                              kernel_initializer=tfrni(stddev=0.1),bias_initializer=tfci(0),
                                              name='d_logits') # [n x 1]
                self.d_fake = tf.sigmoid(self.d_real_logits,name='d') # [n x 1]
            
            # Loss functions
            self.d_loss_reals = tfscewl(logits=self.d_real_logits,labels=tf.zeros_like(self.d_real_logits)) # [n x 1]
            self.d_loss_fakes = tfscewl(logits=self.d_fake_logits,labels=tf.ones_like(self.d_fake_logits)) # [n x 1]
            self.d_losses = self.d_loss_reals + self.d_loss_fakes # [n x 1]
            self.g_losses = tfscewl(logits=self.d_real_logits,labels=tf.ones_like(self.d_real_logits)) # [n x 1]
            self.ae_losses = 0.5*tf.norm(self.x_recon-self.x_real,ord=1,axis=1) # [n x 1]
            self.d_loss = tf.reduce_mean(self.d_losses) # [1]
            self.g_loss = tf.reduce_mean(self.g_losses) # [1]
            self.ae_loss = tf.reduce_mean(self.ae_losses) # [1]
            self.t_vars = tf.trainable_variables()
            self.c_vars = [var for var in self.t_vars if '%s/'%(self.name) in var.name]
            self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in self.c_vars])) # [1]
            
        # Optimizer 
        self.ae_vars = [var for var in self.t_vars if '%s/Q'%(self.name) or '%s/P'%(self.name) in var.name]
        self.d_vars = [var for var in self.t_vars if '%s/D'%(self.name) in var.name]
        self.g_vars = [var for var in self.t_vars if '%s/Q'%(self.name) in var.name]
        self.optm_ae = self.opmz(self.lr).minimize(self.ae_loss+self.l2_reg,var_list=self.ae_vars)
        self.optm_d = self.opmz(self.lr/2.).minimize(self.d_loss+self.l2_reg,var_list=self.d_vars)
        self.optm_g = self.opmz(self.lr).minimize(self.g_loss+self.l2_reg,var_list=self.g_vars)
        
    # Check parameters
    def _check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print ("  [%02d/%d] Name:[%s] Shape:%s" % (i,len(self.g_vars),w_name,w_shape))

    # Train 
    def train(self,_X,_Y=None,_max_iter=1e4,_batch_size=256,
             _PRINT_EVERY=1e3,_PLOT_EVERY=1e3):
        tf.set_random_seed(self.seed); np.random.seed(self.seed) # fix seeds 
        self.sess.run(tf.global_variables_initializer()) # initialize variables
        n_x = _X.shape[0] # number of training data 
        for _iter in range((int)(_max_iter)):
            rand_idx = np.random.permutation(n_x)[:_batch_size] 
            x_batch = _X[rand_idx,:] 
            z_sample = self.sampler.sample(_batch_size)
            feeds = {self.x_real:x_batch,self.kp:0.8,self.z_sample:z_sample}
            _,ae_loss_val = self.sess.run([self.optm_ae,self.ae_loss],feed_dict=feeds)
            _,d_loss_val = self.sess.run([self.optm_d,self.d_loss],feed_dict=feeds)
            for _ in range(2): _,g_loss_val = self.sess.run([self.optm_g,self.g_loss],feed_dict=feeds)
                
            # Print-out
            if (((_iter+1)%_PRINT_EVERY)==0) & (_PRINT_EVERY>0):
                total_loss = ae_loss_val+d_loss_val+g_loss_val
                print ("[%04d/%d]Loss AE:%.3f D:%.3f G:%.3f total loss:%.3f"%
                       (_iter+1,_max_iter,ae_loss_val,d_loss_val,g_loss_val,total_loss))
                
            # Plot samples
            if ( (_iter==0) | (((_iter+1)%_PLOT_EVERY)==0) ) & (_PLOT_EVERY>0):
                # Sample images using z~GMM
                z_samples4img = self.sampler.sample(10) 
                feeds = {self.z_sample:z_samples4img, self.kp:1.0}
                sampled_images = self.sess.run(self.x_sample,feed_dict=feeds)
                plot_imgs(_imgs=sampled_images,_imgSz=(28,28),
                    _nR=1,_nC=10,_figsize=(15,2),_title='Sampled Images',_tfs=18)
                
                # Plot z space
                rand_idx = np.random.permutation(n_x)[:min(n_x,2000)] # upto 2,000 inputs
                x_batch = _X[rand_idx,:]
                z_real = self.sess.run(self.z_real,feed_dict={self.x_real:x_batch,self.kp:1.0})
                z_samples = self.sampler.sample(1000)
                plt.figure(figsize=(6,6))
                h_sample,=plt.plot(z_samples[:,0],z_samples[:,1],'kx')
                if _Y is None:
                    h_real,=plt.plot(z_real[:,0],z_real[:,1],'b.')
                    plt.legend([h_sample,h_real],['Prior','Encoded'],fontsize=15)
                else:
                    hs,strs = [h_sample],['Prior']
                    ys = np.argmax(_Y[rand_idx,:],axis=1)
                    ydim = np.shape(_Y)[1]
                    cmap = plt.get_cmap('gist_rainbow')
                    colors = [cmap(ii) for ii in np.linspace(0,1,ydim)]
                    for i in range(ydim):
                        yi_idx = np.argwhere(ys==i).squeeze()
                        hi,=plt.plot(z_real[yi_idx,0],z_real[yi_idx,1],'.',color=colors[i])
                        hs.append(hi)
                        strs.append('Encoded (%d)'%(i))
                    plt.legend(hs,strs,fontsize=12,bbox_to_anchor=(1.04,1), loc="upper left")
                plt.xlim(-2.0,2.0); plt.ylim(-2.0,2.0)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title('Z-space',fontsize=18)
                plt.show()
                
        print ("[train] Done.")
    
    