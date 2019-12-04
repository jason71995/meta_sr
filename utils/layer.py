import tensorflow as tf
from tensorflow.keras.layers import Layer

class MetaUpSample(Layer):

    def __init__(self, filters, ksize, **kwargs):
        self.filters = filters
        self.ksize = ksize
        super(MetaUpSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MetaUpSample, self).build(input_shape)

    def call(self, inputs):
        x, meta_w = inputs

        w_shape = tf.shape(meta_w)
        x_shape = tf.shape(x)

        # ========== Get projection positions ==========
        indices = tf.meshgrid(tf.range(w_shape[0]),tf.range(w_shape[1]),tf.range(w_shape[2]))
        indices = tf.reshape(tf.transpose(indices, [2,1,3,0]), [-1,3])
        indices = tf.cast(indices, "float32")
        b_idx, h_idx, w_idx = tf.split(indices, num_or_size_splits=3, axis=-1)
        h_idx = tf.cast(x_shape[1], "float32") * (h_idx / tf.cast(w_shape[1], "float32"))
        w_idx = tf.cast(x_shape[2], "float32") * (w_idx / tf.cast(w_shape[2], "float32"))
        indices = tf.concat([b_idx, h_idx, w_idx], axis=-1)
        indices = tf.cast(indices, "int32")

        meta_w = tf.reshape(meta_w,[w_shape[0],w_shape[1],w_shape[2],w_shape[3]//self.filters,self.filters])

        y = tf.image.extract_patches(x, sizes=[1, self.ksize[0], self.ksize[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        y = tf.gather_nd(y, indices)
        y = tf.reshape(y,[w_shape[0],w_shape[1],w_shape[2],w_shape[3]//self.filters,1])
        y = tf.reduce_sum(y*meta_w,axis=-2)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)