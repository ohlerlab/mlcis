######################################################################################################
#
#                                         PART #1: MODEL
#
######################################################################################################

import tensorflow as tf








######################################################################################################
#
#                                         PART #2: UTILITY
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

def orf_column(data,
               frame,
               column_name='ORFs'):
    new_column = []

    if frame < 3:
        for row in data.index:
            orfcount = codons(data['utr'][row], frame)
            new_column.append(orfcount)

        data[str(column_name)] = new_column
    elif frame >= 3:
        print('IMPOSSIBLE FRAME!')

    return data


######################################################################################################

def count_augs(data, frame):
    new = []

    for row in data.index:
        sequence = data['utr'][row]

        # This is one of the alternative frames (out-of-frame)
        if frame == 0:
            split_seq = [sequence[i:i + 3] for i in range(0, len(sequence), 3)]

        # This is the other alternative frame (out-of-frame)
        elif frame == 1:
            split_seq = [sequence[i:i + 3] for i in range(1, len(sequence), 3)]

            # For 50-nt UTRs from Sample et al., this means 'in-frame'
        else:
            split_seq = [sequence[i:i + 3] for i in range(2, len(sequence), 3)]

        # True, if ATG present, False if not
        counter = split_seq.count('ATG')

        # Append indicator for each row
        new.append(counter)

    if frame == 2:
        data['aug_inframe'] = new
    elif frame == 1:
        data['aug_outframe_1'] = new
    else:
        data['aug_outframe_0'] = new

    return data


######################################################################################################

def extract_overlaps(data):
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

def extract_aug_number(dataframe):
    new_column = []

    for i in dataframe['utr'].index:
        aug_count = dataframe['utr'][i].count('ATG')
        new_column.append(aug_count)

    dataframe['aug_number'] = new_column

    return dataframe

######################################################################################################


def analyze(df):
    df = orf_column(df, 2, 'orfs_inframe')
    df = orf_column(df, 0, 'orfs_outframe_0')
    df = orf_column(df, 1, 'orfs_outframe_1')

    df = count_augs(df, 2)
    df = count_augs(df, 0)
    df = count_augs(df, 1)

    df = extract_overlaps(df)

    df = extract_orf_length(df)

    df = extract_orf_number(df)

    df = extract_aug_number(df)

    return df