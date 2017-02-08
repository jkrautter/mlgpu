'''
Created on Feb 5, 2017

@author: jonas
'''

import numpy as np
import reikna.cluda
import reikna.core
import reikna.algorithms

class Mahalanobis:
    def __init__(self, data):
        self.api = reikna.cluda.cuda_api()
        self.thr = self.api.Thread.create()
        self.data = data
        self.d_data = self.thr.to_device(data)
        self.n = data.shape[1]
        self.computeInvCovMatrix()
        
    def computeInvCovMatrix(self):
        self.d_means = self.thr.array((self.data.shape[1]), dtype=np.float32)
        reducer = reikna.algorithms.Reduce(self.d_data, reikna.algorithms.predicate_sum(np.float32), axes=[1])
        reducerc = reducer.compile(self.thr)
        reducerc(self.d_data, self.d_means)
        self.means = self.d_means.get()
        for i in range(len(self.means)):
            self.means[i] /= self.n
            print(str(self.means[i]))        
        self.thr.to_device(self.means, dest=self.d_means)
        self.d_data_mod = self.thr.copy_array(self.d_data)
        data_t = reikna.core.Type(np.float32, shape=self.data.shape)
        means_t = reikna.core.Type(np.float32, shape=self.d_means.shape)
        sub_mean = reikna.algorithms.PureParallel(
            [reikna.core.Parameter("data_mod", reikna.core.Annotation(data_t, "io")),
             reikna.core.Parameter("mean", reikna.core.Annotation(means_t, "i"))],
                        """
                        VSIZE_T idx = ${idxs[0]};
                        VSIZE_T idy = ${idxs[1]};
                        ${data_mod.store_idx}(idx, idy, ${data_mod.load_idx}(idx, idy) + ${mean.load_idx}(idy));
                        """)
        sub_meanc = sub_mean.compile(self.thr)
        sub_meanc(self.d_data_mod, self.d_means)
        
    
