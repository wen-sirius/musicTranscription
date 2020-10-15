import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.nn as nn

class MusicNet():
    def __init__(self, training, name):
        self.is_training = training
        self.name = name

    def gaussian_noise(self, input, std):
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, name='noise')
        return input + noise

    def Temporal_conv(self, x, outchannel, kernel, dilation_rate, name='Temporal'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            r = tf.layers.conv1d(x, filters=outchannel, kernel_size=kernel, padding='same', dilation_rate=dilation_rate, activation=nn.relu)
            r = tf.layers.conv1d(r, filters=outchannel, kernel_size=kernel, padding='same', dilation_rate=dilation_rate)

            if x.shape[-1] == outchannel:
                shortcut = x
            else:
                shortcut = tf.layers.conv1d(x, filters=outchannel, kernel_size=kernel, padding='same')

            o = tf.nn.relu(tf.add_n([r, shortcut]), name='relu')
            return o

    def TCM(self, x, name='TCM'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            skip = x
            x_reduce = tf.reduce_mean(x, axis=2, keepdims=False)
            #assert x_reduce.shape.as_list() == [None, None, 32]
            x_reduce = self.Temporal_conv(x_reduce, outchannel=32, kernel=3, dilation_rate=1, name='block1')
            x_reduce = self.Temporal_conv(x_reduce, outchannel=32, kernel=3, dilation_rate=2, name='block2')
            x_reduce = self.Temporal_conv(x_reduce, outchannel=32, kernel=3, dilation_rate=3, name='block3')
            x_reduce = self.Temporal_conv(x_reduce, outchannel=32, kernel=3, dilation_rate=4, name='block4')

            x_reduce = tf.expand_dims(x_reduce, axis=2)
            x_m = tf.multiply(x, x_reduce)
            out = tf.add_n([x_m, skip])
            return out

    def SFE(self, x, name='SFE_module'):
        x = slim.conv2d(inputs=x, num_outputs=32, kernel_size=(1,3), activation_fn=nn.relu,normalizer_fn=slim.batch_norm,
                         normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='SFE_conv1')
        l, r = tf.split(value=x, num_or_size_splits=2, axis=0)

        lp = slim.conv2d(inputs=l, num_outputs=16, kernel_size=1, activation_fn=nn.softmax,
                        normalizer_fn=None, scope='SFE_conv2_l')
        rp = slim.conv2d(inputs=r, num_outputs=16, kernel_size=1, activation_fn=nn.softmax,
                        normalizer_fn=None, scope='SFE_conv2_r')

        lp = tf.reduce_mean(lp, axis=3, keepdims=True)
        rp = tf.reduce_mean(rp, axis=3, keepdims=True)

        lmrp = l * rp
        rmlp = r * lp

        l = tf.add_n([l, lmrp])
        l = slim.dropout(inputs=l, keep_prob=0.75, is_training=self.is_training, scope='dropout_l_1')
        r = tf.add_n([r, rmlp])
        r = slim.dropout(inputs=r, keep_prob=0.75, is_training=self.is_training, scope='dropout_r_1')
        outputs = tf.concat([l, r], axis=0)

        return outputs


    def net(self, spec_batch, name='net'):
        #assert spec_batch.shape.as_list() == [None, None, 229, 3]
        inputs = tf.concat(tf.split(value=spec_batch, num_or_size_splits=2, axis=-1), axis=0)
        assert inputs.shape.as_list() == [None, None, 229, 1]

        x = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=(3, 7), activation_fn=nn.relu, normalizer_fn=slim.batch_norm,
                         normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='conv_x_1'
                        )    # outputs.shape: (?, ?, 229, 32)
        x = slim.conv2d(inputs=x, num_outputs=32, kernel_size=3, activation_fn=nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='conv_x_2'
                        )
        x = slim.max_pool2d(inputs=x, kernel_size=[1, 2], stride=[1, 2], scope='maxpool_x_1')  # outputs.shape: (?, ?, 114, 32)
        x = slim.dropout(inputs=x, keep_prob=0.75, is_training=self.is_training, scope='dropout_x_1')

        #x = self.gaussian_noise(x, std=0.1)

        x_new = self.TCM(x, name='TCM1')

        x_new = slim.conv2d(inputs=x_new, num_outputs=32, kernel_size=3, activation_fn=nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='conv_xnew_1'
                        )

        x_new = slim.dropout(inputs=x_new, keep_prob=0.75, is_training=self.is_training, scope='dropout_xnew_1')

        outputs = self.SFE(x_new)

        outputs = slim.conv2d(inputs=outputs, num_outputs=64, kernel_size=3, normalizer_fn=slim.batch_norm,
                              normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='conv_2')

        outputs = slim.max_pool2d(inputs=outputs, kernel_size=[1, 2], stride=[1, 2], scope='maxpool_1')
        outputs = slim.dropout(inputs=outputs, keep_prob=0.75, is_training=self.is_training, scope='dropout_2')

        dims = tf.shape(outputs)
        outputs = tf.reshape(tensor=outputs, shape=[dims[0], dims[1], outputs.shape[2].value * outputs.shape[3].value],
                             name='flatten_3')  # outputs.shape: (?, ?, 57*64)

        outputs = slim.fully_connected(inputs=outputs, num_outputs=512, normalizer_fn=slim.batch_norm,
                                       normalizer_params=dict(decay=0.99, is_training=self.is_training), scope='fc_1')

        #assert outputs.shape.as_list() == [None, None, 512]

        outputs = slim.dropout(inputs=outputs, keep_prob=0.5, is_training=self.is_training, scope='dropout_3')

        outputs = slim.fully_connected(inputs=outputs, num_outputs=88, activation_fn=None,scope='output')

        return outputs

    def __call__(self, spec_batch):
        return self.net(spec_batch)
