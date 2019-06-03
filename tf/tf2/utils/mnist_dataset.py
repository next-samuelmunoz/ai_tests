
import gzip
import struct

import numpy as np


class MNIST_Dataset():
    """Load MNIST as numpy arrays
    """
    def __init__(self, ds_path):
        """ds_path is the folder where the 4 .gz files are located
        """
        self.FILES = {
            'train': ds_path+"/train-images-idx3-ubyte.gz",
            'train_labels': ds_path+"/train-labels-idx1-ubyte.gz",
            'test': ds_path+"/t10k-images-idx3-ubyte.gz",
            'test_labels': ds_path+"/t10k-labels-idx1-ubyte.gz"
        }
        
    
    @staticmethod
    def _read_idx(filename):
        """Read a tensor from a file in idx format.
        """
        with gzip.open(filename, 'rb') as fd:
            _, data_type, dims = struct.unpack('>HBB', fd.read(4))
            shape = tuple(
                struct.unpack('>I', fd.read(4))[0]
                for d
                in range(dims)
            )
            return np.frombuffer(fd.read(), dtype=np.uint8).reshape(shape)
    
    def get_train(self):
        return(
            self._read_idx(self.FILES['train']),
            self._read_idx(self.FILES['train_labels'])
        )
        
    def get_test(self):
        return(
            self._read_idx(self.FILES['test']),
            self._read_idx(self.FILES['test_labels'])
        ) 
