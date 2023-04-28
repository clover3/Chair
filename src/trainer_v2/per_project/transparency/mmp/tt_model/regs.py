import tensorflow as tf



class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t



class FLOPS(tf.keras.layers.Layer):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super(FLOPS, self).__init__()
        self.alpha = tf.Variable(0.1, trainable=False, name="alpha")


    def call(self, batch_rep):
        losses = tf.reduce_mean(tf.abs(batch_rep), axis=0) ** 2
        return tf.reduce_sum(losses) * self.alpha

class L1:

    def __call__(self, batch_rep):
        losses = tf.reduce_sum(tf.abs(batch_rep), axis=-1)
        return tf.reduce_mean(losses)

#
# class L0:
#     """non-differentiable
#     """
#
#     def __call__(self, batch_rep):
#         return tf.count_nonzero(batch_rep, dim=-1).float().mean()
#
