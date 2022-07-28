###################################################################################################
#
#                                   IMPORTS
#
###################################################################################################

import random

import numpy as np
import pandas as pd

np.random.seed(1337)

import tensorflow as tf
import keras


import scipy.io
import sklearn
import scipy.stats as stats
import sklearn.preprocessing as preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

###################################################################################################
#
#                                   PREPROCESSING FUNCTIONS
#
###################################################################################################

def get_utr_lengths(data, seq_column = '5UTR', name = 'utr_length'):
    new_column = []
    for x in data.index:
        new_column.append(len(data[seq_column][x]))

    data[name] = new_column

    return data

# zero-padding to length of longest sequence in df
def apply_zeropadding(data, max_length, column = '5UTR'):

    for i in data.index:
        number = max_length - len(data[column][i])

        data[column][i] = (number * 'N') + data[column][i]

    return data

# one-hot encoding
def apply_one_hot_encoding(df, col = '5UTR', seq_len = 50):

    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0], 'u': [0, 0, 0, 1],}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def apply_ohe(df, seq_len = 50):

    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0], 'u': [0, 0, 0, 1],}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df.str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

# build training and test datasets
def build_datasets(data,
                   seq_column = '5UTR',
                   te_column = 'log2(TE)',
                   min_utr_length = 40,
                   max_utr_length = 400,
                   split = 0.2,
                   use_sklearn = False,
                   
                    ):

    # filter sequences
    data = get_utr_lengths(data, seq_column)
    data = data[data['utr_length'] >= min_utr_length]
    data = data[data['utr_length'] <= max_utr_length]

    # apply zeropadding
    data = apply_zeropadding(data, max_utr_length, seq_column)

    if not use_sklearn:
        
        # split dataset into test & train dataset
        data = shuffle(data)
        data.reset_index(inplace = True, drop = True)
    
        # separate test and train sequences
        test = data.iloc[:(round(len(data.index) * split))]
        train = data.iloc[(round(len(data.index) * split)):]

        # one-hot encode
        test_seq = apply_one_hot_encoding(test, col = seq_column, seq_len = max_utr_length)
        train_seq = apply_one_hot_encoding(train, col = seq_column, seq_len = max_utr_length)

        # get only TE Column for Train & Test
        #test = test[te_column]
        #train = train[te_column]
    
    elif use_sklearn:
        
        train_seq, test_seq, train, test = train_test_split(
            data[seq_column], data[te_column], test_size = 0.2, random_state = 1337)
        
        test_seq = apply_ohe(test_seq, seq_len = max_utr_length)
        train_seq = apply_ohe(train_seq, seq_len = max_utr_length)
        

    return test_seq, test, train_seq, train


# cut all UTRs to 50 nt length and discard shorter ones.

def build_uniform_datasets(data,
                        seq_column = '5UTR',
                        length = 50,
                        split = 0.2):

    # filter sequences
    data = get_utr_lengths(data, seq_column)
    data = data[data['utr_length'] >= length]

    # take last 50 positions of the 5UTR
    for row in data.index:
        if data['utr_length'][row] > length:
            data[seq_column][row] = data[seq_column][row][-length:]


    # split dataset into test & train dataset
    data = shuffle(data)
    data.reset_index(inplace = True, drop = True)

    # separate test and train sequences
    test = data.iloc[:(round(len(data.index) * split))]
    train = data.iloc[(round(len(data.index) * split)):]

    # one-hot encode
    test_seq = apply_one_hot_encoding(test, seq_len = length)
    train_seq = apply_one_hot_encoding(train, seq_len = length)

    # get only TE Column for Train & Test
    #test = test[te_column]
    #train = train[te_column]


    return test_seq, test, train_seq, train


###################################################################################################
#
#                                   MODEL BUILDING
#
###################################################################################################

# Layer which slices input tensor into three tensors, one for each frame w.r.t. the canonical start
class FrameSliceLayer(Layer):

    # super() lets you avoid referring to the base class explicitly
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    # run if FrameSliceLayer is called on a set of tensors from a previous layer
    def call(self, x):
        shape = K.shape(x)
        x = K.reverse(x, axes=1)  # reverse, so that frameness is related to fixed point (start codon of CDS)
        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
        return [frame_1, frame_2, frame_3]      #returns 3 matrices, one for each frame.

    # computes the output shape for each of the output vectors
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return [(input_shape[0], None), (input_shape[0], None), (input_shape[0], None)]
        return [(input_shape[0], None, input_shape[2]), (input_shape[0], None, input_shape[2]),
                (input_shape[0], None, input_shape[2])]


# Layer which slices input tensor into three tensors with respect to frame and reverses each tensor.
class DreizackLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):

        shape = K.shape(x)

        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)


        return [frame_1, frame_2, frame_3]      #returns 3 matrices, one for each frame.

    # computes the output shape for each of the output vectors
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return [(input_shape[0], None), (input_shape[0], None), (input_shape[0], None)]
        return [(input_shape[0], None, input_shape[2]), (input_shape[0], None, input_shape[2]),
                (input_shape[0], None, input_shape[2])]


# creates a cool model with the keras functional API

def create_cool_model(
        only_max_pool = False,
        n_conv_layers = 3,
        n_conv_filters = 128,
        conv_filter_size = 8,
        dilation = 1,
        padding = 'same',
        n_dense_layers = 1,
        n_dense_neurons = 64,
        dense_dropout = 0.2,
        loss = 'mean_squared_error'

):

    # Inputs
    inputs = Input(shape = (None, 4))
    conv_features = inputs

    # Convolutions
    for i in range(n_conv_layers):
        conv_features = Conv1D(filters = n_conv_filters, kernel_size = conv_filter_size, dilation_rate = dilation, activation = 'relu',
                         padding = padding)(conv_features)


    # Frameslicing
    frame_masked_features = FrameSliceLayer()(conv_features)


    # Pooling
    pooled_features = []
    max_pooling = GlobalMaxPooling1D()
    avg_pooling = GlobalAveragePooling1D()
    pooled_features = pooled_features + \
                      [max_pooling(frame_masked_features[i]) for i in range(len(frame_masked_features))]

    if not only_max_pool:
        pooled_features = pooled_features + [avg_pooling(frame_masked_features[i]) for i in
                     range(len(frame_masked_features))]

    concat_features = Concatenate(axis=-1)(pooled_features)


    # Dense and Regression
    prediction = concat_features

    for i in range(n_dense_layers):
        prediction = Dense(n_dense_neurons, activation = 'relu')(prediction)
        prediction = Dropout(rate = dense_dropout)(prediction)

    prediction = Dense(1, activation = 'linear')(prediction)


    # Compile the Model
    model = Model(inputs = inputs, outputs = prediction)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss = loss, optimizer = adam)


    return model


def create_cnn_pool(border_mode='same',
               inp_len=100,
               dense_layers = 1,
               nodes=40,
               layers=3,
               filter_len=[8,8,8],
               nbr_filters=[128,128,128],
               dropout3=0.2,
               pool_size = 5,
               stride = 2,
               learnr = 0.001,
               dil = 1
               ):
    
    model = Sequential()
    if layers >= 1:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters[0],kernel_size=filter_len[0]))
        model.add(MaxPool1D(pool_size = pool_size, strides = stride, padding = border_mode))
        
    if layers >= 2:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters[1],kernel_size=filter_len[1]))
        model.add(MaxPool1D(pool_size = pool_size, strides = stride, padding = border_mode))
        
    if layers >= 3:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters[2],kernel_size=filter_len[2]))
        model.add(MaxPool1D(pool_size = pool_size, strides = stride, padding = border_mode))        
    
    
    model.add(Flatten())

    if dense_layers >= 1:
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        model.add(Dropout(dropout3))
        
    if dense_layers >= 2:
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        model.add(Dropout(dropout3))

    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = keras.optimizers.Adam(lr=learnr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #epsilon=1e-08
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model
    

def create_cnn(border_mode='same',
               inp_len=100,
               dense_layers = 1,
               nodes=40,
               layers=3,
               filter_len=[8,8,8],
               nbr_filters=[128,128,128],
               dropout3=0.2,
               pool_bool = False,
               pool_size = 5,
               stride = 2,
               learnr = 0.001,
               ):

    model = Sequential()
    if layers >= 1:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters[0],
                         kernel_size=filter_len[0]))
    if layers >= 2:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters[1],
                         kernel_size=filter_len[1]))
    if layers >= 3:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters[2],
                         kernel_size=filter_len[2]))
        
        
    if pool_bool:
        model.add(MaxPool1D(pool_size = pool_size, strides = stride, padding = border_mode))
 
        
    
    
    model.add(Flatten())

    if dense_layers >= 1:
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        model.add(Dropout(dropout3))
        
    if dense_layers >= 2:
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        model.add(Dropout(dropout3))

    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = keras.optimizers.Adam(lr=learnr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #epsilon=1e-08
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model





def create_dreizack(
        only_max_pool = False,
        n_conv_layers = 3,
        n_conv_filters = 128,
        conv_filter_size = 8,
        dilation = 1,
        padding = 'same',
        n_dense_layers = 1,
        n_dense_neurons = 64,
        dense_dropout = 0.2,
        loss = 'mean_squared_error'
):

    # Inputs
    inputs = Input(shape=(None, 4))
    inputs = K.reverse(inputs, axes = 1)
    frame_sliced_features = inputs

    # Separate Frames and Pool by Codon
    pooled_frame_1 = []
    pooled_frame_2 = []
    pooled_frame_3 = []

    max_pooling = MaxPooling1D(pool_size = 7, stride = 3, padding = 'same')
    avg_pooling = AveragePooling1D(pool_size = 7, stride = 3, padding = 'same')

    pooled_frame_1 = pooled_frame_1 + \
                     [max_pooling(frame_sliced_features[i]) for i in range(len(frame_sliced_features))]
    pooled_frame_2 = pooled_frame_2 + \
                     [max_pooling(frame_sliced_features[i]) for i in range(1,len(frame_sliced_features))]
    pooled_frame_3 = pooled_frame_3 + \
                     [max_pooling(frame_sliced_features[i]) for i in range(2,len(frame_sliced_features))]

    if not only_max_pool:
        pooled_frame_1 = pooled_frame_1 + \
                         [avg_pooling(frame_sliced_features[i]) for i in range(len(frame_sliced_features))]
        pooled_frame_2 = pooled_frame_2 + \
                         [avg_pooling(frame_sliced_features[i]) for i in range(1, len(frame_sliced_features))]
        pooled_frame_3 = pooled_frame_3 + \
                         [avg_pooling(frame_sliced_features[i]) for i in range(2, len(frame_sliced_features))]

    # concatenate pooled features frame-wise
    concat_frame_1 = Concatenate(axis = -1)(pooled_frame_1)
    concat_frame_2 = Concatenate(axis = -1)(pooled_frame_2)
    concat_frame_3 = Concatenate(axis = -1)(pooled_frame_3)

    conv_frame_1 = concat_frame_1
    conv_frame_2 = concat_frame_2
    conv_frame_3 = concat_frame_3

    # Convolute each frame separately
    for i in range(n_conv_layers):
        conv_frame_1 = Conv1D(filters=n_conv_filters, kernel_size=conv_filter_size, dilation_rate=dilation,
                               activation='relu',
                               padding=padding)(conv_frame_1)
        conv_frame_2 = Conv1D(filters=n_conv_filters, kernel_size=conv_filter_size, dilation_rate=dilation,
                              activation='relu',
                              padding=padding)(conv_frame_2)
        conv_frame_3 = Conv1D(filters=n_conv_filters, kernel_size=conv_filter_size, dilation_rate=dilation,
                              activation='relu',
                              padding=padding)(conv_frame_3)

    # Concatenate convolutional outputs
    concat_all = Concatenate(axis = -1)(conv_frame_1, conv_frame_2, conv_frame_3)

    # Dense
    prediction = concat_all

    for i in range(n_dense_layers):
        prediction = Dense(n_dense_neurons, activation='relu')(prediction)
        prediction = Dropout(rate=dense_dropout)(prediction)

    prediction = Dense(1, activation='linear')(prediction)

    # Compile the Model
    model = Model(inputs=inputs, outputs=prediction)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=loss, optimizer=adam)

    return model




###################################################################################################
#
#                                   MODEL TRAINING & MONITORING
#
###################################################################################################

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []


history = LossHistory()

def train_model(model, x, y, test_seq, test, nb_epochs=3, batch = 128):
    model.fit(x, y, batch_size = batch, epochs=nb_epochs, verbose = 1, validation_data=(test_seq, test), callbacks = [history])
    return model


###################################################################################################
#
#                                   TEST MODEL & COMPUTE TEST METRICS
#
###################################################################################################

def make_prediction( model, scaler, data, test_seq, output_column='pred'):


    predictions = model.predict(test_seq).reshape(-1,1)
    data.loc[:, output_column] = scaler.inverse_transform(predictions)

    return data

def make_prediction_no_scale( model, data, test_seq, output_column='pred'):


    predictions = model.predict(test_seq).reshape(-1)
    data.loc[:, output_column] = predictions

    return data

# function to compute r2, plot correlation between measured/predicted and plot model training history (loss)
def test_model(data,
               model,
               te_column = 'log2(TE)',
               pred_column = 'pred',
               title = 'Model Training History'):

    # calculate metrics
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[te_column],data[pred_column])
    r_sq = r_value**2

    # plot training history
    model_loss = pd.DataFrame(model.history.history)

    fig = plt.figure()
    a = model_loss['loss']
    b = model_loss['val_loss']
    c = np.array(range(1,len(model_loss['loss'])+1,1))
    axes = fig.add_axes([0.0, 0.0, 0.8, 0.8])
    axes.plot(c, a, label='training loss')
    axes.plot(c, b, label='validation loss')
    axes.set_xlim(0, len(model_loss['loss']) + 1)
    axes.set_ylim(0, 1)
    axes.legend(loc=3)
    axes.set(xlabel = 'epoch', ylabel='loss',
             title = title)

    return r_sq, fig





###################################################################################################
#
#                                   SEQUENCE FEATURE ATTRIBUTION
#
###################################################################################################

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