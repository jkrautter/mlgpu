'''
Created on Feb 5, 2017

@author: jonas
'''

import numpy as np
import reikna.cluda
import reikna.core
import reikna.algorithms
import reikna.linalg

class Mahalanobis:
    def __init__(self, data):
        self.api = reikna.cluda.cuda_api()
        self.thr = self.api.Thread.create()
        self.data = data
        self.d_data = self.thr.to_device(data)
        self.num = data.shape[0]
        self.n = data.shape[1]
        self.computeInvCovMatrix()
        
    def computeInvCovMatrix(self):
        means = np.zeros(self.n, dtype=np.float32)
        d_means = self.thr.to_device(means)
        reducer = reikna.algorithms.Reduce(self.d_data, reikna.algorithms.predicate_sum(np.float32), axes=[0])
        reducerc = reducer.compile(self.thr)
        reducerc(d_means, self.d_data)
        means = d_means.get()
        for i in range(len(means)):
            means[i] /= self.num
        self.thr.to_device(means, dest=d_means)
        d_data_mod = self.thr.copy_array(self.d_data)
        data_t = reikna.core.Type(np.float32, shape=self.data.shape)
        means_t = reikna.core.Type(np.float32, shape=means.shape)
        sub_mean = reikna.algorithms.PureParallel(
            [reikna.core.Parameter("data_mod", reikna.core.Annotation(data_t, "io")),
             reikna.core.Parameter("mean", reikna.core.Annotation(means_t, "i"))],
                        """
                        VSIZE_T idx = ${idxs[0]};
                        VSIZE_T idy = ${idxs[1]};
                        ${data_mod.store_idx}(idx, idy, ${data_mod.load_idx}(idx, idy) - ${mean.load_idx}(idy));
                        """)
        sub_meanc = sub_mean.compile(self.thr)
        sub_meanc(d_data_mod, d_means)
        d_S = self.thr.array((self.n, self.n), dtype=np.float32)
        mul = reikna.linalg.MatrixMul(d_data_mod, d_data_mod, out_arr=d_S, transposed_a=True)
        mulc = mul.compile(self.thr)
        mulc(d_S, d_data_mod, d_data_mod)
        S = d_S.get()
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                S[i][j] = S[i][j] / (self.n - 1)
        S_inv = np.linalg.inv(S)
        self.d_S_inv = self.thr.to_device(S_inv)
    
    def computeDistance(self, x, y):
        mul = reikna.linalg.MatrixMul(self.d_S_inv, y, transposed_b=True)
        mulc = mul.compile(self.thr)
        d_temp = mulc(self.d_S_inv, y)
        mul = reikna.linalg.MatrixMul(x, d_temp)
        mulc = mul.compile(self.thr)
        d_res = mulc(x, d_temp)
        res = d_res.get()
        print(str(res))