import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class prm:
    def __init__(self):
        self.mode = 'soft'


def iou(pred,mask,prm):
    pred_vector = tf.layers.flatten(pred)
    mask_vector = tf.layers.flatten(mask)

    intersection = tf.reduce_sum(pred_vector*mask_vector)
    epsilon = 1e-7
    #union = tf.reduce_sum(pred_vector)+tf.reduce_sum(mask_vector)+epsilon
    union = tf.reduce_sum(pred_vector)
    union = tf.Print(union,[union,tf.shape(pred_vector), tf.shape(mask_vector)])

    return(intersection/union)

pred=np.zeros((640,480,1))
mask=np.zeros((640,480,1))

pred[200:400,200:400]=255
mask[200:300,200:300]=255
# plt.figure()
# plt.imshow(pred,cmap='gray')
# plt.figure()
# plt.imshow(mask,cmap='gray')

X =  tf.placeholder(shape=(640,480,1),dtype=tf.int8)
y =  tf.placeholder(shape=(640,480,1),dtype=tf.int8)

norm_pred = tf.image.per_image_standardization(X)
norm_mask = tf.image.per_image_standardization(y)
iou = iou(X,y,prm)


with tf.Session() as sess:
    iou = sess.run(iou, feed_dict={ X:pred,y:mask})

