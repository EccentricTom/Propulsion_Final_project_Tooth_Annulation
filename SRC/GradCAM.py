# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:48:37 2020

@author: tom
"""

#!/usr/bin/python -tt

##-------------------------Utility Function------------------------##
from functools import wraps 

# function to add class methods dynamically
def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
##--------------------Libraries-------------------------------------##
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
##-------------------------GradCAM class----------------------------##
class GradCAM:
    pass
##-------------------------Constructor creation-----------------------##
@add_method(GradCAM)
def __init__(self, model, classIdx, layerName=None):

    # store the model, the class index used to measure the class
    # activation map, and the layer to be used when visualizing
    # the class activation map
    self.model = model
    self.classIdx = classIdx
    self.layerName = layerName

    # if the layer name is None, attempt to automatically find
    # the target output layer
    if self.layerName is None:
        self.layerName = self.find_target_layer()
##-------------------------Find Target Layer-----------------------------##
@add_method(GradCAM)
def find_target_layer(self):

    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(self.model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
            
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
##-----------------------Create Heatmap---------------------------------##
@add_method(GradCAM)
def compute_heatmap(self, image, eps=1e-8, verbose=False):

    # Step 1:
    # construct our gradient model by supplying 
    # (1) the inputs to our pre-trained model
    # (2) the output of the (presumably) final 4D layer in the network
    # (3) the output of the softmax activations from the model
    gradModel = Model(
        inputs=[self.model.inputs],
        outputs=[self.model.get_layer(self.layerName).output,
            self.model.output])
    if verbose:
        print('\nStep 1:')
        print('Model:', self.model.name)
        print('Conv layerName:', self.layerName)
        print('Conv Layer Shape:', gradModel.outputs[0].shape)
    

    # Step 2:
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:

        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(image, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, self.classIdx]
    
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    if verbose:
        print('\nStep 2:')
        print('loss:', loss)
        print('convOutputs shape;', convOutputs.shape)
        print('gradients shape:', grads.shape)
  

    # Step 3:  
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    if verbose:
        print('\nStep 3')
        print('guidedGrads shape', guidedGrads.shape)

    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    if verbose:
        print('guidedGrads shape batch removal', guidedGrads.shape)
        print('convOutputs shape batch removal', convOutputs.shape)


    # Step 4:
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    if verbose:
        print('\nStep 4:')
        print('weights shape:', weights.shape)
        print('cam shape:', cam.shape)


    # Step 5:
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    if verbose:
        print('\nStep 5:')
        print('heatmap shape:', heatmap.shape)

    # return the resulting heatmap to the calling function
    return heatmap
##---------------------------Overlay Heatmap--------------------------##
@add_method(GradCAM)
def overlay_heatmap(self, heatmap, image, alpha=0.5, 
                    colormap=cv2.COLORMAP_VIRIDIS):
    
    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    
    # return a 2-tuple of the color mapped heatmap and the output,
    # overlaid image
    return (heatmap, output)
##----------------------------Image Pre-processing Function---------------##
def image_processing_function(im_path):
  # load the original image from gdrive (in OpenCV format)
  # resize the image to its target dimensions

  orig = cv2.imread(im_path)
  orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  

  # load the input image from gdrive (in Keras/TensorFlow format)
  # basic image pre-processing

  image = load_img(im_path, target_size=(224, 224))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  return image, orig


# helper function to get predicted classes
def get_class_predictions(preds, class_rank):
  rank = class_rank
  classes_ranked = np.argsort(preds[0])[::-1]
  i = classes_ranked[rank]

  decoded = imagenet_utils.decode_predictions(preds, 10)
  (imagenetID, label, prob) = decoded[0][0]

  label = "{}: {:.2f}%".format(label, prob * 100)
  print('Class with highest probability:')
  print("{}".format(label))

  return i, decoded