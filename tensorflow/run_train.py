import numpy as np
import tensorflow as tf
import os
import h5py
from calc_cl_func import CalcPowerSpectrum
from unet import UNet

class TrainUNet:
    def __init__(self,n_pixel,n_channel_in,n_channel_first,n_channel_out,
                 n_depth,n_convpatch,lrelu_slope,dp_rate,calccl):

        self.x = tf.placeholder(tf.float32,[None,n_pixel,n_pixel,n_channel_in])
        self.y = tf.placeholder(tf.float32,[None,n_pixel,n_pixel])
        self.mask = tf.placeholder(tf.float32,[None,n_pixel,n_pixel])
        self.is_train = tf.placeholder(tf.bool,[])
        self.lr = tf.placeholder(tf.float32,[])

        model = UNet(n_channel_first,n_channel_out,n_depth,n_convpatch,lrelu_slope,dp_rate,self.x,self.is_train)
        self.output = model.predict()
        self.output = self.output[:,:,:,0]*self.mask

        self.rl_lowl,self.rl_highl = calccl.calc_rl_lowl_highl(self.output,self.y)
        self.loss1 = tf.reduce_mean(1.-self.rl_lowl)
        self.loss2 = tf.reduce_mean(1.-self.rl_highl)
        self.loss = self.loss1+self.loss2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

if __name__ == "__main__":

    temp = input('enter n_epoch, n_batch, learning rate, and dropout rate:\n')
    n_epoch = np.int32(temp.split()[0])
    n_batch = np.int32(temp.split()[1])
    lr = temp.split()[2]
    dp = temp.split()[3]

    print('start reading valid data...')
    fin = h5py.File('/mnt/ceph/users/ctchiang/cmb_dust_21cm/maps/val_64x64.hdf5','r')
    mask_valid = fin['mask'][:]
    X_valid = np.moveaxis(fin['train_x'][:],1,-1)
    y_valid = fin['train_y'][:]*mask_valid/100.
    print('done')

    n_pixel = X_valid.shape[1]
    n_channel_in = X_valid.shape[-1]

    n_depth = 4
    n_channel_first = 64
    n_channel_out = 1
    n_convpatch = 3

    res_arcmin = 180./460*60.
    size_arcmin = res_arcmin*n_pixel
    size_rad = size_arcmin/60./180.*np.pi
    lF = 2.*np.pi/size_rad
    nl = 32

    lrelu_slope = 0.
    dp_rate = np.float64(dp)

    calccl = CalcPowerSpectrum(n_pixel,n_pixel,lF,nl)
    trainunet = TrainUNet(n_pixel,n_channel_in,n_channel_first,n_channel_out,
                          n_depth,n_convpatch,lrelu_slope,dp_rate,calccl)
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print('total number of parameters: %d'%total_parameters)

    fd = '/mnt/ceph/users/ctchiang/cmb_dust_21cm/loss_rl/ctmodel_dropout_tf/dp'+dp+'/'
    model_fname = fd+'unet-model'
    loss_fname = fd+'training_valid_loss.dat'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists(model_fname+'.data-00000-of-00001'):
            print('restoring a trained model...')
            saver.restore(sess,tf.train.latest_checkpoint(fd))
        else:
            print('starting a new model...')
            sess.run(tf.global_variables_initializer())

        if n_epoch>0:
            print('start reading training data...')
            fin = h5py.File('/mnt/ceph/users/ctchiang/cmb_dust_21cm/maps/train_l460_mask72_dust.hdf5','r')
            mask_train = np.float32(fin['mask'][:])
            X_train = np.moveaxis(np.float32(fin['train_x'][:]),1,-1)
            y_train = np.float32(fin['train_y'][:])*mask_train/100.
            print('done')

            n_train = len(X_train)
            n_step = np.int32(np.ceil(n_train/n_batch))
            loss_out = open(loss_fname,'a')

            for i_epoch in range(n_epoch):
                index = np.random.choice(n_train,n_train,replace=False)
                for i_step in range(n_step):
                    if i_step==n_step-1:
                        index_batch = np.sort(index[i_step*n_batch:])
                    else:
                        index_batch = np.sort(index[i_step*n_batch:(i_step+1)*n_batch])

                    X_batch = X_train[index_batch,:,:,:]
                    y_batch = y_train[index_batch,:,:]
                    mask_batch = mask_train[index_batch,:,:]

                    _, batch_loss1, batch_loss2, batch_loss = sess.run(
                                (trainunet.train_step,trainunet.loss1,trainunet.loss2,trainunet.loss),
                                feed_dict={trainunet.x:X_batch,
                                           trainunet.y:y_batch,
                                           trainunet.mask:mask_batch,
                                           trainunet.is_train:True,
                                           trainunet.lr:lr})

                    valid_loss1, valid_loss2, valid_loss = sess.run(
                                (trainunet.loss1,trainunet.loss2,trainunet.loss),
                                feed_dict={trainunet.x:X_valid,
                                           trainunet.y:y_valid,
                                           trainunet.mask:mask_valid,
                                           trainunet.is_train:False})

                    print('epoch %d, step %d: %.3e %.3e %.3e %.3e %.3e %.3e'%(i_epoch,i_step,batch_loss,batch_loss1,batch_loss2,valid_loss,valid_loss1,valid_loss2))
                    loss_out.write('epoch %d, step %d: %.3e %.3e %.3e %.3e %.3e %.3e\n'%(i_epoch,i_step,batch_loss,batch_loss1,batch_loss2,valid_loss,valid_loss1,valid_loss2))

#                    if i_step%100==0:
#                        saver.save(sess,model_fname,write_meta_graph=False)
                saver.save(sess,model_fname,write_meta_graph=False)
                if i_epoch%10==0 and i_epoch>0:
                    lr = lr/10.
        else:
            rl_lowl,rl_highl = sess.run((trainunet.rl_lowl,trainunet.rl_highl),
                                         feed_dict={trainunet.x:X_valid,
                                                    trainunet.y:y_valid,
                                                    trainunet.mask:mask_valid,
                                                    trainunet.is_train:False})

            fout = open(fd+'rl_valid_pred.dat','w')
            for i in range(len(X_valid)):
                fout.write('%.6e %.6e\n'%(rl_lowl[i],rl_highl[i]))
            fout.close()

            mask_valid = None
            X_valid = None
            y_valid = None

            print('start reading test data...')
            fin = h5py.File('/mnt/ceph/users/ctchiang/cmb_dust_21cm/maps/test_l460_mask72_dust.hdf5','r')
            mask_test = np.float32(fin['mask'][:])
            X_test = np.moveaxis(np.float32(fin['train_x'][:]),1,-1)
            y_test = np.float32(fin['train_y'][:])*mask_test/100.
            print('done')

            fout = open(fd+'rl_test_pred.dat','w')
            n_test = len(X_test)
            n_split = n_test//2
            for k in range(2):
                split1 = k*n_split
                split2 = (k+1)*n_split

                rl_lowl,rl_highl = sess.run((trainunet.rl_lowl,trainunet.rl_highl),
                                             feed_dict={trainunet.x:X_test[split1:split2,:,:,:],
                                                        trainunet.y:y_test[split1:split2,:,:],
                                                        trainunet.mask:mask_test[split1:split2,:,:],
                                                        trainunet.is_train:False})

                for i in range(n_split):
                    fout.write('%.6e %.6e\n'%(rl_lowl[i],rl_highl[i]))
            fout.close()

            mask_test = None
            X_test = None
            y_test = None
