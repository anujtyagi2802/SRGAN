from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
from scipy.misc import imresize
import argparse
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
import argparse

image_shape = (96,96,3)
    
downscale = 4

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        images.append(imresize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale], interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr

def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr ,filenames ,dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
    
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir + filenames[index] + '_sr.png')
    
        #plt.show()

loss = VGG_LOSS(image_shape)  
model = load_model('./model/gen_model3000.h5' , custom_objects={'vgg_loss': loss.vgg_loss})

def load_data(d, ext):
    files = []
    filenames = []
    for f in os.listdir(d): 
        if f.endswith(ext):
            image = data.imread(os.path.join(d,f))
            image = imresize(image, [384,384], interp='bicubic', mode=None)
            if len(image.shape) > 2:
                files.append(image)
                filenames.append(f)
    return files,filenames     

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_hig_res', action='store', dest='test_path', default='./data/' , help='Path for test input images Hig resolution')
values = parser.parse_args()
path = values.test_path
files,filenames = load_data(path, ".jpg")

x_test_hr = array(files)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(files, 4)
x_test_lr = normalize(x_test_lr)

plot_test_generated_images_for_model('./output/',model, x_test_hr, x_test_lr,filenames)
    



