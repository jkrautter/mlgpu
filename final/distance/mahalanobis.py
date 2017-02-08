'''
Created on Feb 5, 2017

@author: jonas
'''

import numpy as np
import reikna.cluda
import reikna.core
import reikna.algorithms
import reikna.linalg
import math

class Mahalanobis:
    def __init__(self, data, ids):
        self.api = reikna.cluda.cuda_api()
        self.thr = self.api.Thread.create()
        self.data = data
        self.ids = ids
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
        self.S = d_S.get()
        for i in range(self.S.shape[0]):
            for j in range(self.S.shape[1]):
                self.S[i][j] = self.S[i][j] / (self.num - 1)
        self.S_inv = np.linalg.inv(self.S)
        self.d_S_inv = self.thr.to_device(self.S_inv)
    
    def computeDistance(self, x, y):
        diff = np.array([np.subtract(x, y)])
        d_diff = self.thr.to_device(diff)
        d_temp = self.thr.array((y.shape[0], 1), dtype=np.float32)
        mul = reikna.linalg.MatrixMul(self.d_S_inv, d_diff, out_arr=d_temp, transposed_b=True)
        mulc = mul.compile(self.thr)
        mulc(d_temp, self.d_S_inv, d_diff)
        
        d_res = self.thr.array((1, 1), dtype=np.float32)
        mul = reikna.linalg.MatrixMul(d_diff, d_temp, out_arr=d_res)
        mulc = mul.compile(self.thr)
        mulc(d_res, d_diff, d_temp)
        res = d_res.get()
        
        return math.sqrt(res[0][0])
    
    def getKNearest(self, x, k):
        maxd = 0
        nearest = [0]
        dists = [self.computeDistance(x, self.data[0])]
        for i in range(1, self.data.shape[0]):
            if (i % 1000) == 0:
                print("Compared " + str(i) + " vectors...\n")
            dist = self.computeDistance(x, self.data[i])
            if dist < dists[maxd] or len(nearest) < k:
                if len(nearest) < k:
                    nearest.append(i)
                    dists.append(dist)
                    if dist > dists[maxd]:
                        maxd = len(nearest) - 1
                else:
                    nearest[maxd] = i
                    dists[maxd] = dist
                    for j in range(len(nearest)):
                        if dists[j] > dists[maxd]:
                            maxd = j
        return {"nearest": nearest, "dists": dists}
