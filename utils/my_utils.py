
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

######################################################################################################
#
#                                         UTILITY FUNCTIONS
#
######################################################################################################

minOrfLength = 3
# This function is written with code adapted from https://github.com/SchulzLab/SplitOrfs/blob/master/OrfFinder.py
def codons(seq, der_frame):
    '''
    :param seq: A nucleotide sequence as string
    :param countOrfs: The number of ORFs in that nucleotide sequence
    :return: The number of ORFs in that nucleotide sequence
    '''

    stops = ['TAA', 'TGA', 'TAG']
    lst1 = []
    lst2 = []
    start = 0
    counter = 0
    countOrfs = 0
    startPosition = []  # to collect start position of uorfs
    stopPosition = []  # to collect stop position of uorfs

    # initialize list for 3 optional orfs
    for i in range(3):
        lst1.append([])
        lst2.append([])

    # add positions of start and stop codons to lists
    while (seq and counter < 3):
        for i in range(start, len(seq), 3):
            codon = seq[i:i + 3]
            if (codon == 'ATG'):
                lst1[start].append(i)
            if (codon in stops):
                lst2[start].append(i)

        # increase starting position and counter by one
        start += 1
        counter += 1

    # for each frame extract ORFs
    frame = der_frame

    if len(lst1[frame]) > 0 and len(lst2[frame]) > 0:  # at least one start and one stop codon exist

        currentStart = 0
        currentStop = 0

        while currentStart < len(lst1[frame]) and currentStop < len(lst2[frame]):
            if lst1[frame][currentStart] < lst2[frame][currentStop]:  # this is the condition for a valid orf
                
                orfseq = seq[lst1[frame][currentStart]:lst2[frame][currentStop]]
                start_pos = lst1[frame][currentStart]
                stop_pos = lst2[frame][currentStop]
                
                if len(orfseq) >= minOrfLength:
                    countOrfs += 1

                    startPosition.append(start_pos)
                    stopPosition.append(stop_pos)

                while currentStart < len(lst1[frame]) and lst1[frame][currentStart] < lst2[frame][currentStop]:
                    currentStart = currentStart + 1
                currentStop = currentStop + 1

            elif lst1[frame][currentStart] > lst2[frame][currentStop]:
                currentStop = currentStop + 1

    result = [countOrfs, startPosition, stopPosition]

    return result


######################################################################################################

def orf_column(data, frame):
    '''
    param data: dataframe containing a column with sequences of RNA 5' untranslated regions
    param frame: the biological reading frame to count uORFs in (0,1 or 2)
    param column_name: name of the column to save uORF counts in
    output: dataframe with appended column denoting number of uORFs in each sequence
    '''
    assert frame < 3, 'There are only three biological reading frames.'
    column_name='orfs_outframe_'+str(frame) if frame in [0,1] else 'orfs_inframe'
    data[column_name]=[codons(row, frame) for row in data['utr']]
    return data

######################################################################################################

def count_augs(data, frame):
    '''
    counts frame-wise number of ATG motifs in each utr sequence and adds a saves them in a new column
    '''
    #split_seq = [data['utr'][i][frame::3] for i in data.index]
    #new=[seq.count('ATG') for seq in split_seq]

    new = []
    for row in data.index:
        sequence = data['utr'][row]
        split_seq = [sequence[i:i + 3] for i in range(frame, len(sequence), 3)]
        new.append(split_seq.count('ATG'))

    column_name='aug_outframe_'+str(frame) if frame in [0,1] else 'aug_inframe'
    data[column_name]=new

    return data

   
######################################################################################################

def extract_overlaps(data):
    '''
    This function creates a new column 'overlapping_orfs' in the input dataframe 'data'
    and populates it with information about whether the ORFs present in the UTR are overlapping or not.
    '''
    new_column = []
    for x in data.index:
        # empty positions of all orfs from list after each entry
        uorf_pos_list = []
        # aggregate positions of all orfs in utr
        for column in ['orfs_inframe', 'orfs_outframe_0', 'orfs_outframe_1']:
            for orf in range(0, data[column][x][0]):
                uorf_pos_list.append(range(data[column][x][1][orf],
                                           data[column][x][2][orf]))
        # in case the sequence has multiple uorfs
        if len(uorf_pos_list) >= 2:
            # aggregate positions of orfs as sets
            for r in range(len(uorf_pos_list)):
                uorf_pos_list[r] = set(uorf_pos_list[r])
            # return intersection of sets
            inter = list(set.intersection(*uorf_pos_list))
            # if orfs are non-overlapping, length is 0
            if len(inter) == 0:
                new_column.append(False)
            # if orfs are overlapping, length is longer than 0
            elif len(inter) > 0:
                new_column.append(True)
        elif len(uorf_pos_list) <= 1:
            new_column.append('none')
    data['overlapping_orfs'] = new_column
    return data


######################################################################################################

def extract_orf_length(data):
    '''
    This function creates a new column 'orf_length' in the input dataframe 'data'
    and populates it with the length of the ORF(s) present in the UTR.
    '''
    new_column = []

    for x in data.index:

        # empty positions of all orfs from list after each entry
        uorf_pos_list = []

        # aggregate positions of all orfs in utr
        for column in ['orfs_inframe', 'orfs_outframe_0', 'orfs_outframe_1']:
            for orf in range(0, data[column][x][0]):
                uorf_pos_list.append(data[column][x][2][orf] - data[column][x][1][orf])

        # In case the 5UTR has exactly one orf
        if len(uorf_pos_list) == 1:
            new_column.append(uorf_pos_list[0])

        else:
            new_column.append('none')

    data['orf_length'] = new_column

    return data


######################################################################################################

def extract_orf_number(data):
    '''
    quantifies the total number of uORFs in each utr sequence in the dataframe
    and saves them in a new column called 'orf_number'
    '''
    new_column = []

    for x in data.index:

        # empty positions of all orfs from list after each entry
        uorf_pos_list = []

        # aggregate positions of all orfs in utr
        for column in ['orfs_inframe', 'orfs_outframe_0', 'orfs_outframe_1']:
            for orf in range(0, data[column][x][0]):
                uorf_pos_list.append(range(data[column][x][1][orf],
                                           data[column][x][2][orf]))

        # add orf count to new column
        new_column.append(len(uorf_pos_list))

    data['orf_number'] = new_column

    return data
######################################################################################################

def extract_aug_number(df):
    '''
    quantifies the total number of ATG motifs in each utr sequence in the dataframe
    and saves them in a new column called 'aug_number'
    '''
    df['aug_number'] = df['utr'].str.count('ATG')
    return df

######################################################################################################


def analyze(df):
    '''
    input: dataframe with utr sequences and corresponding measures
    output: dataframe withh columns for orf measures

    '''
    #frame-wise number of uORFs/AUGs
    for frame in [2, 0, 1]:
        df = orf_column(df, frame)
        df = count_augs(df, frame)

    df = extract_overlaps(df)
    #length of each identified orf
    df = extract_orf_length(df)
    #total number of orfs per utr
    df = extract_orf_number(df)
    #total number of augs per utr
    df = extract_aug_number(df)

    return df


######################################################################################################

def barplot_mrl(data):
    '''
    input: dataframe containing utr sequences as well as corresponding measured
    ans predicted MRL values
    output: dataframe which can be plotted with sns.boxplot(..., hue='pred')
    '''

    dfx = pd.DataFrame()
    dfx['MRL'] = data['rl']
    dfx['utr'] = data['utr']
    dfx['feature'] = np.nan
    dfx['pred'] = [False for row in dfx.index]

    dfy = pd.DataFrame()
    dfy['MRL'] = data['pred']
    dfy['utr'] = data['utr']
    dfy['feature'] = np.nan
    dfy['pred'] = [True for row in dfy.index]

    for x in data.index:
        if (data['orf_number'][x] == 1) & (data['aug_number'][x] == 1) & (data['orfs_inframe'][x][0] == 1):
            dfx['feature'][x] = 'IF_uORF'
            dfy['feature'][x] = 'IF_uORF'
        elif (data['orf_number'][x] == 1) & (data['aug_number'][x] == 1) & (data['orfs_inframe'][x][0] == 0):
            dfx['feature'][x] = 'OOF_uORF'
            dfy['feature'][x] = 'OOF_uORF'
        elif (data['orf_number'][x] == 0) & (data['aug_number'][x] == 1) & (data['aug_inframe'][x] == 0):
            dfx['feature'][x] = 'OOF_OV_uORF'
            dfy['feature'][x] = 'OOF_OV_uORF'
        elif (data['orf_number'][x] == 0) & (data['aug_number'][x] == 1) & (data['aug_inframe'][x] == 1):
            dfx['feature'][x] = 'NTEx'
            dfy['feature'][x] = 'NTEx'
        elif (data['aug_number'][x] == 0) & (data['utr'][x].startswith('TG') == False):
            dfx['feature'][x] = 'None'
            dfy['feature'][x] = 'None'
        
    comp_data = pd.concat([dfx, dfy], axis = 0)

    return comp_data
######################################################################################################

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2


######################################################################################################

def corplot(df, obs_col='rl', pred_col='pred'):

    print(r2(df['rl'], df['pred']))

    g, ax = plt.subplots(figsize=(6,6))
    g = sns.regplot(data = df, x = obs_col, y = pred_col, scatter_kws={'alpha':0.5}, line_kws={"color": "black"})
    g.set(ylim=(0,9), xlim=(0,9), xticks = range(1,10,1), yticks = range(1,10,1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Measured MRL', fontsize=20)
    plt.ylabel('Predicted MRL', fontsize=20)
    sns.despine()