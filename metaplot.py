import numpy as np
import pandas as pd

import IntegratedGradients as ig

import matplotlib.pyplot as plt
import seaborn as sns


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



class Metaplot:

    '''
    Computes a metaplot across attribution maps for given examples sequences from input data.

    Input: numpy array with attribution maps to compute metaplots for.

    Params:
    - ex_seq: numpy array storing attribution values of sequences to relate
    - matrix: computed matrix for metaplot

    Output: numpy array of metaplot visualized as heatmap.
    '''

    def __init__(self, ex_seq):
        
        self.visualize(self.compute_meta(ex_seq))

    def compute_meta(self, ex_seq):

        dim = ex_seq.shape[1:]
        meta = np.zeros(dim)
        for i in ex_seq:
            meta = np.add(meta,i)
        meta = np.transpose(meta)
        return meta

    def visualize(self, matrix):

        ax = plt.subplots(figsize = (10,5))
        ax = sns.heatmap(data = matrix, linewidths=.1, cmap = 'coolwarm', yticklabels = False, xticklabels = 10, cbar = False, square = True)
        plt.show(ax)





#data = pd.read_csv("/Users/frederickkorbel/Documents/projects/paper/data/MRL_pred.csv", nrows = 1000)

#seq = one_hot_encode(data)

#ind = [i[0] for i in data if data['rl'][i] > 7]

#att = np.array([ig.explain(seq[i] for i in ind)])
