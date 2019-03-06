import numpy as np
import tensorflow as tf

class CalcPowerSpectrum:
    def __init__(self,n0,n1,lF,nl):
        self.n0 = n0
        self.n1 = n1
        self.lF = lF
        self.nl = nl

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

        self.real_dof = np.ones((n0,cn1))
        cond1 = i1==0
        cond2 = ((n1%2)==0)*(i1==(n1/2))
        cond3 = i0>x0
        cond = (cond1+cond2)*cond3
        self.real_dof[cond] = 0
        self.real_dof = self.real_dof==1

        self.lbin = np.int32(l/lF)
        self.nmodes = np.zeros(nl)
        for i in range(nl):
            cond = (self.lbin==i)*self.real_dof
            self.nmodes[i] = np.sum(cond)

        lbin_lowl = (l>0)*(l<100)
        lbin_highl = (l>=100)*(l<200)

        self.cond_lowl = lbin_lowl*self.real_dof
        self.cond_highl = lbin_highl*self.real_dof

    def calc_rl_lowl_highl(self,image1,image2):
        image1_fft = tf.spectral.rfft2d(image1)
        image2_fft = tf.spectral.rfft2d(image2)

        cl11 = tf.real(image1_fft)**2+tf.imag(image1_fft)**2
        cl22 = tf.real(image2_fft)**2+tf.imag(image2_fft)**2
        cl12 = tf.real(image1_fft)*tf.real(image2_fft)+tf.imag(image1_fft)*tf.imag(image2_fft)

        rl = cl12/tf.sqrt(cl11*cl22)
        rl_lowl = tf.reduce_mean(tf.boolean_mask(rl,self.cond_lowl,axis=1),axis=1)
        rl_highl = tf.reduce_mean(tf.boolean_mask(rl,self.cond_highl,axis=1),axis=1)

        return rl_lowl,rl_highl

    def calc_cl_rl(self,image1,image2):
        cl11_bin = []
        cl22_bin = []
        cl12_bin = []

        image1_fft = tf.spectral.rfft2d(image1)
        image2_fft = tf.spectral.rfft2d(image2)

        cl11 = tf.real(image1_fft)**2+tf.imag(image1_fft)**2
        cl22 = tf.real(image2_fft)**2+tf.imag(image2_fft)**2
        cl12 = tf.real(image1_fft)*tf.real(image2_fft)+tf.imag(image1_fft)*tf.imag(image2_fft)

        for i in range(nl):
            cond = (self.lbin==i)*self.real_dof
            cl11_bin.append(tf.reduce_sum(tf.boolean_mask(cl11,cond,axis=1),axis=1))
            cl12_bin.append(tf.reduce_sum(tf.boolean_mask(cl22,cond,axis=1),axis=1))
            cl22_bin.append(tf.reduce_sum(tf.boolean_mask(cl12,cond,axis=1),axis=1))

        cl11_bin = tf.transpose(tf.convert_to_tensor(cl11_bin))/self.nmodes
        cl12_bin = tf.transpose(tf.convert_to_tensor(cl12_bin))/self.nmodes
        cl22_bin = tf.transpose(tf.convert_to_tensor(cl22_bin))/self.nmodes
        rl_bin = cl12_bin/tf.sqrt(cl11_bin*cl22_bin)

        return cl11_bin,cl22_bin,cl12_bin,rl_bin

if __name__ == "__main__":
    """
    testing
    """

    n_pixel = 64
    res_arcmin = 180./460*60.
    size_arcmin = res_arcmin*n_pixel
    size_rad = size_arcmin/60./180.*np.pi
    lF = 2.*np.pi/size_rad

    nl = 32

    CalcCl = CalcPowerSpectrum(n_pixel,n_pixel,lF,nl)
    image1_pf = tf.placeholder(tf.float32,[None,n_pixel,n_pixel])
    image2_pf = tf.placeholder(tf.float32,[None,n_pixel,n_pixel])

    n_image = 20
    image1 = np.random.normal(size=(n_image,n_pixel,n_pixel))
    image2 = np.random.normal(size=(n_image,n_pixel,n_pixel))

#    sess = tf.Session()
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        rl_lowl,rl_highl = sess.run(CalcCl.calc_rl_lowl_highl(image1,image2))
#        cl11,cl22,cl12,rl = sess.run(CalcCl.calc_cl_rl(image1,image2))
#        rl_lowl,rl_highl = sess.run(CalcCl.calc_rl_lowl_highl(image1_pf,image2_pf),feed_dict={image1_pf:image1,image2_pf:image2})
#        cl11,cl22,cl12,rl = sess.run(CalcCl.calc_cl_rl(image1_pf,image2_pf),feed_dict={image1_pf:image1,image2_pf:image2})
#        sess.close()

    rl_lowl,rl_highl = CalcCl.calc_rl_lowl_highl(image1,image2)
    cl11,cl22,cl12,rl = CalcCl.calc_cl_rl(image1,image2)

    print(rl_lowl.shape)
    print(cl11.shape)
