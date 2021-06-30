# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:34:18 2020

@author: tom
"""
import tensorflow as tf
from PIL import ImageEnhance

def up_contrast(img):
  x = tf.keras.preprocessing.image.array_to_img(img)
  img_contr_obj=ImageEnhance.Contrast(x)
  factor=2
  e_img=img_contr_obj.enhance(factor)
  final = tf.keras.preprocessing.image.img_to_array(e_img)
  return final
