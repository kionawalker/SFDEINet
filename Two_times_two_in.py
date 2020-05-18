# _*_ coding:utf-8 _*_

import chainer
from chainer import Chain, initializers, cuda
from chainer import functions as F
from chainer import links as L
import numpy as np

class two_times_two_in(Chain):

    def __init__(self):
        super(two_times_two_in, self).__init__()
        self.W = initializers.GlorotUniform()
        self.b = initializers.Constant(fill_value=0.0)
        with self.init_scope():

            self.conv1 = L.Convolution2D(3, 16, ksize=7, stride=1, initialW=self.W, initial_bias=self.b)
            self.conv2 = L.Convolution2D(16, 64, ksize=7, stride=1, initialW=self.W, initial_bias=self.b)
            self.conv3 = L.Convolution2D(64, 256, ksize=7, stride=1, initialW=self.W, initial_bias=self.b)
            self.fc4 = L.Linear(None, 52, initialW=self.W, initial_bias=self.b)
            self.BN1 = L.BatchNormalization(16)
            self.BN2 = L.BatchNormalization(64)
            self.BN3 = L.BatchNormalization(256)
            self.BN4 = L.BatchNormalization(52)

    def forward(self, input):


        v = F.relu(self.BN1(self.conv1(input)))

        v = F.max_pooling_2d(v, ksize=2, stride=2)

        v = F.relu(self.BN2(self.conv2(v)))
        # v.unchain()

        v = F.max_pooling_2d(v, ksize=2, stride=2)

        v = F.dropout(F.relu(self.BN3(self.conv3(v))), ratio=0.5)

        # v.unchain()
        return v

    '''
    def __call__(self, pos1, pos2, neg1, neg2):
        # 同じdtを1つのバッチに束ねる
        dt1 = F.vstack((pos1, neg1))
        dt2 = F.vstack((pos2, neg2))

        dt1 = self.forward(dt1)
        p1, n1 = F.split_axis(dt1, [int(dt1.shape[0] / 2)], axis=0)

        dt2 = self.forward(dt2)
        p2, n2 = F.split_axis(dt2, [int(dt1.shape[0] / 2)], axis=0)

        p = F.concat((p1, p2), axis=1)
        p = F.relu(self.BN4(self.fc4(p)))

        # n1 = self.forward(neg1)
        # n2 = self.forward(neg2)
        n = F.concat((n1, n2), axis=1)
        n = F.relu(self.BN4(self.fc4(n)))

        # p.unchain()
        # n.unchain()
        p_out = p
        n_out = n

        # print 'p_out:{}'.format(cuda.to_cpu(p_out.data)), 'n_out:{}'.format(cuda.to_cpu(n_out.data))

        return p_out, n_out
    '''

    def __call__(self, dt1, dt2):

        feature_dt1 = self.forward(dt1)
        feature_dt2 = self.forward(dt2)

        l = F.concat((feature_dt1, feature_dt2), axis=1)
        out_put = F.relu(self.BN4(self.fc4(l)))

        return out_put


if __name__=='__main__':
    # x = cp.arange([1, 1, 128, 88])
    y = np.random.randn(10, 1, 128, 88).astype(np.float32)
    z = np.random.randn(10, 1, 128, 88).astype(np.float32)

    model = two_in()
    output = model(y, z)
