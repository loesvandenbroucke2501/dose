import os
import tensorflow
import keras
from keras.layers import Conv3D, Activation, Concatenate, MaxPooling3D, Conv3DTranspose, UpSampling3D, BatchNormalization
from keras.initializers import HeNormal

import nibabel as nib

def HDunet3d_conv(input_, nb_filters, strides = (1,1,1), kernel_size = (3,3,3), initializer = HeNormal(seed = 10), padding = 'same', n = 2):

    x = input_
    # dropout?

    for i in range(n):
        y = Conv3D(filters = nb_filters, strides = strides, kernel_size = kernel_size, kernel_initializer = initializer, padding = padding)(x)
        y = Activation('relu')(y)
        y = BatchNormalization()(y)
    
        x = Concatenate()([x,y])

    return x

def HDunet3d_maxpool(input_, nb_filters, strides = (2,2,2), kernel_size = (2,2,2), initializer = HeNormal(seed = 10)):

    x = input_
    # dropout?

    y = MaxPooling3D(pool_size = kernel_size, strides = strides)(x)

    z = Conv3D(filters = nb_filters, kernel_size = (3,3,3), strides = (2,2,2), kernel_initializer = initializer, padding = 'same')(x)
    z = Activation('relu')(z)
    z = BatchNormalization()(z)

    x = Concatenate()([y,z])

    return x

def HDunet3d_upsample(input_, nb_filters, size = (2,2,2), strides = (1,1,1), kernel_size = (3,3,3), initializer = HeNormal(seed = 10), padding = 'same'):

    x = input_
    # dropout?

    # x = Conv3DTranspose(filters = nb_filters, strides = strides, kernel_size = kernel_size, kernel_initializer = initializer, padding = padding)(x)
    x = UpSampling3D(size = (2,2,2), data_format = 'channels_last')(x)
    x = Conv3D(filters = nb_filters, strides = strides, kernel_size = kernel_size, kernel_initializer = initializer, padding = padding)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    return x

def HDunet3d_output(input_, nb_filters, strides = (2,2,2), kernel_size = (3,3,3), initializer = HeNormal(seed = 10), padding = 'same'):

    x = input_
    # dropout?

    x = Conv3D(filters = nb_filters, strides = strides, kernel_size = kernel_size, kernel_initializer = initializer, padding = padding)(x)
    x = Activation('linear')(x)
    x = BatchNormalization()(x)

    return x

'''
input = (nib.load('data_test/train/0057_DC/FL_maps_volume.nii.gz')).get_fdata()
input = tensorflow.expand_dims(input, axis = 0) # add dimension for batch size
input = tensorflow.expand_dims(input, axis = -1) # add dimension for channels

output = HDunet3d_conv(input, nb_filters = 16)
print('output_shape = ', output.shape)
'''