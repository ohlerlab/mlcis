import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf






def lin_interpolate(input_matrix, baseline=False, steps=50):
    '''
    input: numpy array of the sample to explain

    params:
    - baseline: reference to use for linear interpolation; zero by default
    - steps: number of steps to take on the linear path from baseline to sample matrix

    output: (#steps) number of numpy arrays to be calculated local gradients for
    '''

    if baseline is False: baseline = np.zeros(input_matrix.shape)

    assert input_matrix.shape == baseline.shape

    linear_path = np.zeros(tuple([steps] + [t for t in input_matrix.shape]))
    for i in range(float(steps)):
        linear_path[i] = baseline + (input_matrix - baseline) * (i*float(1.0)/float(steps)) # baseline + delta * alpha

    return linear_path



def compute_gradients(examples, model):
    '''
    input: linear path of interpolated images from original sample to explain; 
            the models' prediction function

    params:

    output: gradients between output predictions with respect to input features
    '''

    with tf.GradienTape() as tape:
        tape.watch(examples)
        values = model(examples) #the models' prediction function
        dF_dx = tape.gradients(values, examples)

    return dF_dx


def integral_approximation(gradients):
    '''
    input: gradients to approximate integral for with Riemann's trapezoidal sum

    params:

    output: one tensor of the same dimensions as the input example with integrated gradients for each pixel position 
    '''

    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)

    integrated_grads = tf.math.reduce_mean(grads, axis=0)

    return integrated_grads



def scale_int_gradients(int_gradients, image, base):

    integrated_gradients = (image - base) * int_gradients

    return integrated_gradients





def integrated_gradients(model, input_matrix, baseline=False, steps=50):

    #linear interpolation
    if baseline is False: baseline = np.zeros(input_matrix.shape)
    assert input_matrix.shape == baseline.shape

    linear_path = np.zeros(tuple([steps] + [t for t in input_matrix.shape]))
    for i in range(steps):
        linear_path[i] = baseline + (input_matrix - baseline) * (i*1.0/steps) # baseline + delta * alpha
    linear_path = linear_path.squeeze()
    linear_path = tf.convert_to_tensor(linear_path, dtype = tf.float32)


    #compute gradients
    with tf.GradientTape() as tape:
        tape.watch(linear_path)
        values = model(linear_path) #call the model to obtain predictions for each position
        dF_dx = tape.gradient(values, linear_path)

    #integral approximation
    grads = (dF_dx[:-1] + dF_dx[1:]) / tf.constant(2.0)
    integrated_grads = tf.math.reduce_mean(grads, axis=0)
    assert input_matrix.squeeze().shape == integrated_grads.shape

    #scale integrated gradients with respect to input
    integrated_gradients = (input_matrix - baseline) * integrated_grads

    return integrated_gradients