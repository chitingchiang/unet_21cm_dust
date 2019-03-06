import numpy as np
import torch

def calc_cond(n0,n1,lF,nl):
    cn1 = n1//2+1

    i0 = np.arange(n0)
    i0 = np.repeat(i0,cn1).reshape(n0,cn1)
    x0 = (n0-i0)%n0
    cond = i0>n0/2
    l0 = i0*lF
    l0[cond] = l0[cond]-n0*lF

    i1 = np.arange(cn1)
    i1 = np.repeat(i1,n0).reshape(cn1,n0).transpose()
    l1 = i1*lF

    l = np.sqrt(l0**2+l1**2)

    real_dof = np.ones((n0,cn1))
    cond1 = i1==0
    cond2 = ((n1%2)==0)*(i1==(n1/2))
    cond3 = i0>x0
    cond = (cond1+cond2)*cond3
    real_dof[cond] = 0
    real_dof = real_dof==1

    lbin_lowl = (l>0)*(l<100)
    lbin_highl = (l>=100)*(l<200)

    cond_lowl = lbin_lowl*real_dof
    cond_highl = lbin_highl*real_dof

    cond_lowl = torch.from_numpy(cond_lowl.astype(np.int)).type(torch.uint8)
    cond_highl = torch.from_numpy(cond_highl.astype(np.int)).type(torch.uint8)

    return cond_lowl,cond_highl

def calc_rl(image1,image2,cond_lowl,cond_highl):

    image1_fft = torch.rfft(image1,2)
    image2_fft = torch.rfft(image2,2)

    cl11 = image1_fft[:,:,:,0]**2+image1_fft[:,:,:,1]**2
    cl22 = image2_fft[:,:,:,0]**2+image2_fft[:,:,:,1]**2
    cl12 = image1_fft[:,:,:,0]*image2_fft[:,:,:,0]+image1_fft[:,:,:,1]*image2_fft[:,:,:,1]

    rl = cl12/torch.sqrt(cl11*cl22)

    rl_lowl = torch.mean(rl[:,cond_lowl],dim=1)
    rl_highl = torch.mean(rl[:,cond_highl],dim=1)

    return rl_lowl,rl_highl

