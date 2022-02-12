import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class IntegratedGradients:

    def __init__(self, model, inp, base, alphas, steps = 50):

        self.model = model

        self.steps = steps
      

        self.integral_approximation(self.compute_gradients(self.interpolate_images(inp, base, alphas)))

  

    def compute_gradients(self, inputs):

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            preds = 
        
        gradients = tape.gradient(probs, images)

        return gradients
    
    def integral_approximation(self, gradients):
    
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_grads = tf.math.reduce_mean(grads, axis = 0)

        return integrated_grads

    def scale_gradients(self, integrated_grads):

        ig = (inp - base) * integrated_grads

        return ig






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
    for i in range(steps):
        linear_path[i] = baseline + (input_matrix - baseline) * (i*1.0/steps) # baseline + delta * alpha

    return linear_path



def compute_gradients(examples, model)
    '''
    input: linear path of interpolated images from original sample to explain

    params:
    - model: the models' prediction function

    output: gradients between output predictions with respect to input features
    '''

    with tf.GradienTape() as tape:
        tape.watch(examples)
        values = model(examples) #the models' prediction function applied to example
        dF_dx = tape.gradients(values, examples)

    return dF_dx