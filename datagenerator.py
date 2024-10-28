from keras.utils import Sequence
import nibabel as nib
import numpy as np

# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

class fit_DataGenerator(Sequence):

    '''
    Keras DataGenerator for training.
    '''

    def __init__(self, input_file_paths, output_file_paths, batch_size, input_size, output_size):
        
        self.input_file_paths = input_file_paths
        self.output_file_paths = output_file_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self): # should be implemented because of Sequence class
        'Calculate the number of batches per epoch'
        return len(self.input_file_paths) // self.batch_size
    
    def __getitem__(self, index): # should be implemented because of Sequence class
        'Generate one batch of data'
        batch_input_paths = self.input_file_paths[index * self.batch_size : (index+1)*self.batch_size]
        batch_output_paths = self.output_file_paths[index * self.batch_size : (index+1)*self.batch_size]

        X, y = self.__load_batch(batch_input_paths, batch_output_paths)

        return X, y
    
    def __load_batch(self, batch_input_paths, batch_output_paths):
        X = np.empty((self.batch_size, *self.input_size))
        y = np.empty((self.batch_size, *self.output_size))

        for i, (input_path, output_path) in enumerate(zip(batch_input_paths, batch_output_paths)):

            input_data = (nib.load(input_path[0])).get_fdata()
            output_data = (nib.load(output_path[0])).get_fdata()

            if np.isnan(input_data).any() or np.isnan(output_data).any():
                print(f'NaN detected in {input_path[0]} or {output_path[0]}')
            if np.isinf(input_data).any() or np.isinf(output_data).any():
                print(f'Inf detected in {input_path[0]} or {output_path[0]}')

            # normalize to have zero mean and standard deviation of 1 (z-score normalization)
            input_data_normalized = (input_data - np.mean(input_data)) / np.std(input_data)
            output_data_normalized = (output_data - np.mean(output_data)) / np.std(output_data)

            X[i,] = input_data_normalized
            y[i,] = output_data_normalized

        return X, y

class predict_DataGenerator(Sequence):

    '''
    Keras DataGenerator for predicting/testing.
    '''

    def __init__(self, input_file_paths, batch_size, input_size):

        self.input_file_paths = input_file_paths
        self.batch_size = batch_size
        self.input_size = input_size

    def __len__(self): # should be implemented because of Sequence class
        'Calculates the number of batches per epoch'

        return len(self.input_file_paths) // self.batch_size
    
    def __getitem__(self, index): # should be implemented because of Sequence class
        'Generate one batch of data'
        batch_input_paths = self.input_file_paths[index * self.batch_size : (index+1)*self.batch_size]

        X = self.__load_batch(batch_input_paths)

        return X
    
    def __load_batch(self, batch_input_paths):
        X = np.empty((self.batch_size, *self.input_size))

        for i, input_path in enumerate(batch_input_paths):

            input_data = (nib.load(input_path[0])).get_fdata()

            if np.isnan(input_data).any():
                print(f'NaN detected in {input_path[0]}')
            if np.isinf(input_data).any():
                print(f'Inf detected in {input_path[0]}')

            # normalize to have zero mean and standard deviation of 1 (z-score normalization)
            input_data_normalized = (input_data - np.mean(input_data)) / np.std(input_data)

            X[i,] = input_data_normalized

        return X