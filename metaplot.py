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
    - dim: tuple supplying dimensions of input attribution maps
    - ex_seq: numpy array storing attribution values of sequences to relate

    Output: numpy array of metaplot visualized as heatmap.
    '''

    def __init__(self, dim: tuple, ex_seq):
        
        self.compute_meta(dim, ex_seq)
        self.visualize()


    def compute_meta(dim = (50,4), ex_seq):

        


    def visualize():






data = pd.read_csv("/Users/frederickkorbel/Documents/projects/paper/data/MRL_pred.csv", nrows = 1000)

seq = one_hot_encode(data)

#ind = [i[0] for i in data if data['rl'][i] > 7]

#att = np.array([ig.explain(seq[i] for i in ind)])
att = np.random.normal(loc=10, size =(4, 50))

meta = np.zeros((4,50))

for i in att[i]:
    meta = np.add(att, meta)

print(meta)