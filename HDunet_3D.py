import os
import keras
from code.HDunet3d_blocks import HDunet3d_conv, HDunet3d_maxpool, HDunet3d_upsample, HDunet3d_output
from keras.layers import Input, Concatenate
from keras import Model
from keras import metrics
from keras import losses
import numpy as np
from code.datagenerator import fit_DataGenerator, predict_DataGenerator

class HDUnet_3D(object):

    """
    HDUnet_3D class to create a 3D HD U-Net model.

    Attributes:
        input_size (list): the size of the input data
        nb_inputs (int): the number of input channels
        nb_outputs (int): the number of output channels
        nb_filters (int): the number of filters
        nb_times_down_up (int): the number of times the model goes down and up/the number of hierarchical levels in the downward/upward path
        nb_conv_blocks (int): the number of convolutional blocks in each hierarchical level
        kernel_size_conv (tuple): the kernel size for the convolutional layers
        strides_conv (tuple): the stride for the convolutional layers
        kernel_size_maxpool (tuple): the kernel size for the maxpooling layers
        strides_maxpool (tuple): the stride for the maxpooling layers
        strides_final_layer (tuple): the stride for the final layer
        kernel_size_final_layer (tuple): the kernel size for the final layer
    """



    def __init__(self, 
                 input_size, 
                 nb_inputs,
                 nb_outputs, 
                 nb_filters = 16,
                 nb_times_down_up = 3,
                 nb_conv_blocks = 2,
                 kernel_size_conv = (3,3,3),
                 strides_conv = (1,1,1),
                 kernel_size_maxpool = (2,2,2),
                 strides_maxpool = (2,2,2),
                 strides_final_layer = (1,1,1),
                 kernel_size_final_layer = (1,1,1)
                 ):

        """
        Initializes the HDUnet_3D model with the given parameters.
        """


        self.input_size = input_size
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_filters = nb_filters
        self.nb_times_down_up = nb_times_down_up
        self.nb_conv_blocks = nb_conv_blocks
        self.kernel_size_conv = kernel_size_conv
        self.strides_conv = strides_conv
        self.kernel_size_maxpool = kernel_size_maxpool
        self.strides_maxpool = strides_maxpool
        self.strides_final_layer = strides_final_layer
        self.kernel_size_final_layer = kernel_size_final_layer

    def create_model(self):

        """
        Creates the 3D HD U-Net model.

        Returns:
            keras.Model: the 3D HD U-Net model  
        """

        inputs = []
        input_ = x = Input(shape = list(self.input_size) + [self.nb_inputs])
        inputs.append(input_)
        skip_connections = []

        # downward path / encoder

        for i_down in range(self.nb_times_down_up): 

            # print('i_down: ', i_down)

            x = HDunet3d_conv(x, self.nb_filters, strides = self.strides_conv, kernel_size = self.kernel_size_conv, n = self.nb_conv_blocks)
            skip_connections.append(x)
            x = HDunet3d_maxpool(x, self.nb_filters, strides = self.strides_maxpool, kernel_size = self.kernel_size_maxpool)
        
        # upward path / decoder
        for i_up in list(reversed(range(self.nb_times_down_up))):

            x = HDunet3d_conv(x, self.nb_filters, strides = self.strides_conv, kernel_size = self.kernel_size_conv, n = self.nb_conv_blocks)
            x = HDunet3d_upsample(x, self.nb_filters, strides = self.strides_conv, kernel_size = self.kernel_size_conv)
            x = Concatenate()([skip_connections[i_up], x])

        
        x = HDunet3d_conv(x, self.nb_filters, strides = self.strides_conv, kernel_size = self.kernel_size_conv, n = self.nb_conv_blocks)

        # output layer
        x = HDunet3d_output(x, self.nb_outputs, strides = self.strides_final_layer, kernel_size = self.kernel_size_final_layer, padding = 'same')

        # print('output_shape - ', x.shape)

        model = keras.Model(inputs = inputs, outputs = [x], name = 'HDUnet_3D')

        return model
    

input_size = [64,64,32]
n_inputs = 1
n_outputs = 1

HDunet_model = (HDUnet_3D(input_size = input_size, nb_inputs = n_inputs, nb_outputs = n_outputs)).create_model()
# HDunet_model.summary()

from keras import backend

def custom_mse(y_true, y_pred):
    return backend.mean(backend.square(y_true - y_pred))

HDunet_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5, clipvalue = 1.0), loss = 'mean_squared_error', metrics = ['mse', 'accuracy'])

# data_path = '/DATASERVER/MIC/GENERAL/STAFF/lvdnbrz/dose_prediction/data_test'
data_path = '/uz/data/radiotherapie/Shared/Data/LoesVdb/eclipse_data/fluence_volume_data'
train = 'train'
validatie = 'validatie'
# data_path_train = os.path.join(data_path, train)
data_path_train = data_path
# data_path_validatie = os.path.join(data_path, validatie)

patients_train = ['0057_DC', '0069_DC', '0072_DC', '0074_DC', '0078_DC', 'Lung_015', 'Lung_016', 'Lung_023', 'Lung_025', 'Lung_028']
patients_validatie = ['0004_DC', '0010_DC', '0017_DC', '0019_DC', '0022_DC']
remove_patients = ['0046_DC', '0038_DC', 'Lung_009', 'Lung_010', 'Lung_013', 'Lung_021', 'Lung_024', 'Lung_031', 'Lung_039', 'Lung_041', 'Lung_048', 'Lung_063', 'Lung_074', 'Lung_075', 'Lung_077', 'Lung_080', 'Lung_091', 'Lung_153', 'Lung_012', 'Lung_027', 'Lung_042', 'Lung_002', 'Lung_006', 'Lung_047', 'Lung_079', 'Lung_160', 'Lung_137', 'Lung_130']


inputs = ['FL_maps_volume_reshaped']
outputs = ['RTDOSE_reshaped']

def CreateListOfPaths(paths, listofnames, patients, removepatients):
    if not isinstance(paths, list):
        paths = [paths for i in listofnames]
    listofpaths = []
    for patient in patients:
        if patient in removepatients:
            continue
          
        input_paths = []  # voor elke patient willen we een andere lijst
        for inp_i, (inp, path) in enumerate(zip(listofnames, paths)):
            input_paths.append(os.path.join(path, patient, inp+'.nii.gz'))
    
        listofpaths.append(input_paths)
    return listofpaths

input_file_paths_train = CreateListOfPaths(data_path_train, inputs, patients_train, remove_patients)
output_file_paths_train = CreateListOfPaths(data_path_train, outputs, patients_train, remove_patients)

# input_file_paths_validatie = CreateListOfPaths(data_path_validatie, inputs, patients_validatie, remove_patients)
# output_file_paths_validatie = CreateListOfPaths(data_path_validatie, outputs, patients_validatie, remove_patients) 

batch_size = 5
input_size = [160,112,80]
output_size = [160,112,80]

train_generator = fit_DataGenerator(input_file_paths = input_file_paths_train, output_file_paths = output_file_paths_train, batch_size = batch_size, input_size = input_size, output_size = output_size)
validatie_generator = fit_DataGenerator(input_file_paths = input_file_paths_validatie, output_file_paths = output_file_paths_validatie, batch_size = batch_size, input_size = input_size, output_size = output_size)

history = HDunet_model.fit(train_generator, epochs = 100)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

test_generator = predict_DataGenerator(input_file_paths = input_file_paths_validatie[0:1], batch_size = 1, input_size = input_size)
Y_test = HDunet_model.predict(test_generator)
print(Y_test.shape)

Y_test = np.squeeze(Y_test)
plt.imshow(Y_test[:,:,10])
plt.show()