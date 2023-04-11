

from keras import backend as K

from keras.engine.topology import Layer

# import numpy as np
# old = np.load
# np.load = lambda *a,**k: old(*a,**k,allow_pickle=True) 



 
class Self_Attention(Layer):
 
    def __init__(self, output_dim,return_attention=False, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
        self.return_attention = return_attention
 
    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
 
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
 
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
 
        print("WQ.shape",WQ.shape)
 
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
 
 
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
 
        QK = QK / K.sqrt(K.cast(self.output_dim, dtype=K.floatx()))
 
        QK = K.softmax(QK)
 
        print("QK.shape",QK.shape)
 
        V = K.batch_dot(QK,WV)
        if self.return_attention:
            return [V, QK]
 
        return V
 
    def compute_output_shape(self, input_shape):
 
        return (input_shape[0],input_shape[1],self.output_dim)
        
    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(Self_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))