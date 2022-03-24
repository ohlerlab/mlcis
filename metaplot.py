import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

#ob = [('candy', 10, 300), ('apple', 20, 100), ('horse', 30, 200)]
#print(sorted(ob, key= lambda x:x[1], reverse = False))
# sorted function takes the second object in each element of the list to sort from

#a = np.zeros((2,4))
#b = np.arange(8).reshape((2,4))
#c = np.add(a,b)


def one_hot_encode(df, col='utr', seq_len=50):
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    vectors=np.empty([len(df),seq_len,4])
    
    for i,seq in enumerate(df[col].str[:seq_len]): 
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def meta_untransposed(ex_seq):

        dim = ex_seq.shape[1:]
        meta = np.zeros(dim)
        for i in ex_seq:
            meta = np.add(meta,i)
        return meta


class Metaplot:

    '''
    Computes a metaplot across attribution maps for given examples sequences from input data.

    Input: numpy array with attribution maps to compute metaplots for.

    Params:
    - ex_seq: numpy array storing attribution values of sequences to relate
    - matrix: computed matrix for metaplot

    Output: numpy array of metaplot visualized as heatmap.
    '''

    def __init__(self, ex_seq, colorbar):
        
        self.visualize(self.compute_meta(ex_seq), colorbar)

    def compute_meta(self, ex_seq):

        counter = 0
        dim = ex_seq.shape[1:]
        meta = np.zeros(dim)

        for i in ex_seq:
            meta = np.add(meta,i)
            counter += 1

        meta = np.transpose((meta/counter))

        return meta

    def visualize(self, matrix, colorbar = False):

        xlabel_list =   [1, None, None, None, None, None, None, None, None, 10,
                    None, None, None, None, None, None, None, None, None, 20,
                    None, None, None, None, None, None, None, None, None, 30,
                    None, None, None, None, None, None, None, None, None, 40,
                    None, None, None, None, None, None, None, None, None, 50]

        ax = plt.subplots(figsize = (20,10))
        ax = sns.heatmap(data = matrix, linewidths=.1, cmap = 'coolwarm', center = 0, yticklabels = ['A', 'C', 'G', 'T'], xticklabels = xlabel_list, cbar = colorbar, cbar_kws = {'shrink': .13, 'pad':0.02}, square = True)
        plt.show(ax)
