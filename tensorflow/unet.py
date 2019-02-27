import numpy as np
import tensorflow as tf

class UNet:
    def __init__(self,n_channel_first,n_channel_out,n_depth,n_convpatch,lrelu_slope,dp_rate,x,is_train):

        self.n_convpatch = n_convpatch
        self.lrelu_slope = lrelu_slope
        self.dp_rate = dp_rate

#        print(x)

        # in conv
        x = self.doubleconv(x,n_channel_first,is_train)

        # down conv
        n_ch_out = n_channel_first*2
        x_down = []
        for i in range(n_depth-1):
            x_down.append(x)
            x = self.downconv(x,n_ch_out,is_train)
            n_ch_out = n_ch_out*2
        x_down.append(x)
        n_ch_out = n_ch_out//2
        x = self.downconv(x,n_ch_out,is_train)

        # up conv
        for i in range(n_depth-1):
            n_ch_out = n_ch_out//2
            x = self.upconv(x,x_down[-i-1],n_ch_out,is_train)
        x = self.upconv(x,x_down[0],n_ch_out,is_train)

        # final conv
        self.output = tf.layers.conv2d(x,filters=n_channel_out,kernel_size=1,padding='same')
#        print(self.output)

    def predict(self):
        return self.output

    def doubleconv(self,x,n_ch_out,is_train):
        for _ in range(2):
            x = tf.layers.conv2d(x,filters=n_ch_out,kernel_size=self.n_convpatch,padding='same',use_bias=False)
#            print(x)
            x = tf.layers.batch_normalization(x,axis=-1,momentum=0.9,training=is_train)
#            print(x)
            x = tf.nn.leaky_relu(x,alpha=self.lrelu_slope)
#            print(x)
            x = tf.layers.dropout(x,rate=self.dp_rate,training=is_train)
#            print(x)
        return x

    def downconv(self,x,n_ch_out,is_train):
        x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
#        print(x)
        x = self.doubleconv(x,n_ch_out,is_train)
        return x

    def upconv(self,x1,x2,n_ch_out,is_train):
        x = tf.layers.conv2d_transpose(x1,filters=x1.shape[-1],kernel_size=(2,2),strides=(2,2))
#        print(x)
        x = tf.concat([x,x2],axis=-1)
#        print(x)
        x = self.doubleconv(x,n_ch_out,is_train)
        return x

if __name__ == "__main__":
    """
    testing
    """
    n_batch = 32
    n_pixel = 64
    n_channel_in = 50
    n_channel_first = 64
    n_channel_out = 1
    n_depth = 4
    n_convpatch = 3
    lrelu_slope = 0.
    dp_rate = 0.5

    image_pf = tf.placeholder(tf.float32,[None,n_pixel,n_pixel,n_channel_in])
    is_train_pf = tf.placeholder(tf.bool,[])

    model = UNet(n_channel_first,n_channel_out,n_depth,n_convpatch,lrelu_slope,dp_rate,image_pf,is_train_pf)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print('total number of parameters: %d'%total_parameters)

    image_value = np.random.normal(size=(n_batch,n_pixel,n_pixel,n_channel_in))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run((model.predict()),feed_dict={image_pf:image_value,is_train_pf:True})

    print(y.shape)
