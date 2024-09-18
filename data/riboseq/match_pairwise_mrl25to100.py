import pandas as pd
import pickle as pkl
from Bio import Align

utr_data = pd.read_csv('/fast/AG_Ohler/frederick/projects/mlcis/data/riboseq/calviello_te_processed.csv')

mrl50 = pd.read_csv('/fast/AG_Ohler/frederick/projects/mlcis/data/mrl25to100/GSM3130443_designed_library.csv').drop('Unnamed: 0', axis=1)
mrl50 = mrl50[(mrl50['library'] == 'human_utrs') | (mrl50['library'] == 'snv')]
mrl50 = mrl50.sort_values('total', ascending = False).reset_index(drop = True).iloc[:25000]

mrl25to100 = pd.read_csv('/fast/AG_Ohler/frederick/projects/mlcis/data/mrl25to100/GSM4084997_varying_length_25to100.csv').drop('Unnamed: 0', axis=1)
mrl25to100 = mrl25to100.loc[mrl25to100.set == 'human']
mrl25to100 = mrl25to100.loc[mrl25to100.total_reads >= 10 ]  #select reporters with most reads similar to Sample et. al
mrl25to100 = mrl25to100.sort_values('total', ascending = False).reset_index(drop = True)






def match_pairwise(x, col_x = 'utr', threshold = 40):
    a = x[col_x].to_numpy()
    b = utr_data['five_utr'].to_numpy()
    score_dict = {}

    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = -5
    aligner.extend_gap_score = 0
    
    for i,seq1 in enumerate(a):
        for j,seq2 in enumerate(b):
            score = aligner.score(seq1, seq2[-100:])
            if score > threshold:
                #if i not in [key[0] for key in score_dict.keys()]:
                score_dict[(i,j)] = (seq1,seq2)

    return score_dict




score_mrl25to100_calviello = match_pairwise(mrl25to100)

with open('score_mrl25to100_calviello.pkl', 'wb') as f:
    pkl.dump(score_mrl25to100_calviello, f)
