import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import matplotlib.pyplot as pt
import h5py
from unet import UNet
import time
from calc_cl_func_newbin import calc_cond,calc_rl

use_cuda = True
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
print('The code is running with %s'%(device))

n_pix = 64
res_arcmin = 180./460*60.
size_arcmin = res_arcmin*n_pix
size_rad = size_arcmin/60./180.*np.pi
lF = 2.*np.pi/size_rad
nl = 32
cond_lowl,cond_highl = calc_cond(n_pix,n_pix,lF,nl)

cond_lowl = cond_lowl.to(device)
cond_highl = cond_highl.to(device)

def train(model,X_train,y_train,mask_train,X_valid,y_valid,mask_valid,model_file,loss_file,lr=1e-3,n_batch=40,n_epoch=10,gamma=60.):
    n_train = len(X_train)
    n_valid = X_valid.size()[0]

#    optimizer = Adam(model.parameters(),lr=lr,weight_decay=wd)
#    scheduler = ReduceLROnPlateau(optimizer,'min',verbose=True,patience=20,factor=0.5)

    loss_out = open(loss_file,'a')

    n_step = np.int32(np.ceil(n_train/n_batch))
    for i_epoch in range(n_epoch):
        optimizer = Adam(model.parameters(),lr=lr)
        index = np.random.choice(n_train,n_train,replace=False)
        for i_step in range(n_step):
            if i_step==n_step-1:
                index_batch = np.sort(index[i_step*n_batch:])
            else:
                index_batch = np.sort(index[i_step*n_batch:(i_step+1)*n_batch])
            mask_batch = torch.from_numpy(mask_train[index_batch,:,:]).to(device)
            X_batch = torch.from_numpy(X_train[index_batch,:,:,:]).to(device)
            y_batch = torch.from_numpy(y_train[index_batch,:,:]).to(device)
            model.train(True)
            y_batch_pred = model(X_batch)
            y_batch_pred = y_batch_pred[:,0,:,:]*mask_batch
            rl_batch_lowl,rl_batch_highl = calc_rl(y_batch,y_batch_pred,cond_lowl,cond_highl)
            rl_batch_lowl = rl_batch_lowl.to(device)
            rl_batch_highl = rl_batch_highl.to(device)
            batch_loss1 = torch.mean(torch.abs(y_batch_pred-y_batch))
            batch_loss2 = torch.mean(1.-rl_batch_lowl)
            batch_loss3 = torch.mean(1.-rl_batch_highl)
            batch_loss = batch_loss1+gamma*(batch_loss2+batch_loss3)
            with torch.no_grad():
                model.train(False)
                y_valid_pred = model(X_valid)
                y_valid_pred = y_valid_pred[:,0,:,:]*mask_valid
                rl_valid_lowl,rl_valid_highl = calc_rl(y_valid,y_valid_pred,cond_lowl,cond_highl)
                rl_valid_lowl = rl_valid_lowl.to(device)
                rl_valid_highl = rl_valid_highl.to(device)
                valid_loss1 = torch.mean(torch.abs(y_valid_pred-y_valid))
                valid_loss2 = torch.mean(1.-rl_valid_lowl)
                valid_loss3 = torch.mean(1.-rl_valid_highl)
                valid_loss = valid_loss1+gamma*(valid_loss2+valid_loss3)
#                scheduler.step(valid_loss)
                print('epoch %d, step %d: %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e'%(i_epoch,i_step,batch_loss.item(),batch_loss1.item(),batch_loss2.item(),batch_loss3.item(),valid_loss.item(),valid_loss1.item(),valid_loss2.item(),valid_loss3.item()))
                loss_out.write('epoch %d, step %d: %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e\n'%(i_epoch,i_step,batch_loss.item(),batch_loss1.item(),batch_loss2.item(),batch_loss3.item(),valid_loss.item(),valid_loss1.item(),valid_loss2.item(),valid_loss3.item()))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
#            if i_step%100==0:
#                torch.save(model.state_dict(),model_file)
        torch.save(model.state_dict(),model_file)
        if i_epoch%10==0 and i_epoch>0:
            lr = lr/10.

    loss_out.close()

if __name__ == "__main__":

    temp = input('enter n_epoch, n_batch, learning rate, dropout rate, and gamma:\n')
    n_epoch = np.int32(temp.split()[0])
    n_batch = np.int32(temp.split()[1])
    lr = temp.split()[2]
    dp = temp.split()[3]
    gamma = temp.split()[4]

    print('start reading valid data...')
    fin = h5py.File('cmb_dust_21cm/maps/val_64x64.hdf5','r')
    mask_valid = torch.from_numpy(fin['mask'][:]).to(device)
    X_valid = torch.from_numpy(fin['train_x'][:]).to(device)
    y_valid = torch.from_numpy(fin['train_y'][:]).to(device)*mask_valid
    print('done')

    n_pixel = X_valid.size()[2]
    n_channel_in = X_valid.size()[1]

    n_depth = 4
    n_channel_out = 1
    n_convpatch = 3
    n_channel_first = 64

    model = UNet(n_channel_in,n_channel_first,n_channel_out,n_depth,n_convpatch,0.,float(dp)).to(device)
    print(model)

    fd = 'cmb_dust_21cm/pytorch/dp'+dp+'_gamma'+gamma+'/'
    model_fname = fd+'unet-model'
    loss_fname = fd+'training_valid_loss.dat'

    if os.path.exists(model_fname):
        model.load_state_dict(torch.load(model_fname))

    if n_epoch>0:
        print('start reading training data...')
        fin = h5py.File('cmb_dust_21cm/maps/train_l460_mask72_dust.hdf5','r')
        mask_train = np.float32(fin['mask'][:])
        X_train = np.float32(fin['train_x'][:])
        y_train = np.float32(fin['train_y'][:])*mask_train
        print('done')

        train(model,X_train,y_train,mask_train,X_valid,y_valid,mask_valid,model_fname,loss_fname,float(lr),n_batch,n_epoch,float(gamma))
    else:
        with torch.no_grad():
            model.load_state_dict(torch.load(model_fname))

            model.train(False)
            y_valid_pred = model(X_valid)
            y_valid_pred = y_valid_pred[:,0,:,:]*mask_valid

            rl_lowl,rl_highl = calc_rl(y_valid_pred,y_valid,cond_lowl,cond_highl)

            fout = open(fd+'rl_valid_pred.dat','w')
            n_valid = y_valid.size()[0]
            for i in range(n_valid):
                fout.write('%.6e %.6e\n'%(rl_lowl[i],rl_highl[i]))
            fout.close()

            mask_valid = None
            X_valid = None
            y_valid = None
            y_valid_pred = None

            print('start reading test data...')
            fin = h5py.File('cmb_dust_21cm/maps/test_l460_mask72_dust.hdf5','r')
            mask_test = np.float32(fin['mask'][:])
            X_test = np.float32(fin['train_x'][:])
            y_test = np.float32(fin['train_y'][:])*mask_test
            print('done')

            fout = open(fd+'rl_test_pred.dat','w')

            n_test = y_test.shape[0]
            n_split = n_test//2
            for k in range(2):
                mask_split = torch.from_numpy(mask_test[k*n_split:(k+1)*n_split,:,:]).to(device)
                X_split = torch.from_numpy(X_test[k*n_split:(k+1)*n_split,:,:,:]).to(device)
                y_split = torch.from_numpy(y_test[k*n_split:(k+1)*n_split,:,:]).to(device)

                model.train(False)
                y_split_pred = model(X_split)
                y_split_pred = y_split_pred[:,0,:,:]*mask_split

                rl_lowl,rl_highl = calc_rl(y_split_pred,y_split,cond_lowl,cond_highl)

                for i in range(n_split):
                    fout.write('%.6e %.6e\n'%(rl_lowl[i],rl_highl[i]))

            fout.close()
