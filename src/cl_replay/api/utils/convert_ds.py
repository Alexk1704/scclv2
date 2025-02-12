import sys
import gzip
import struct
import numpy as np



def read_img(file):
    with gzip.open(file, 'rb') as f:
        # first 16 bytes of the file for ds info
        magic_no, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # big endian, 4-byte uint
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 28, 28, 1)
    return images


def read_labels(file):
    with gzip.open(file, 'rb') as f:
        magic_no, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


if __name__ == '__main__':
    pass
    # NOTE: comment in below for converting EMNIST:
    """    
    tr_x = f'{sys.argv[1]}-train-images-idx3-ubyte.gz'
    tr_y = f'{sys.argv[1]}-train-labels-idx1-ubyte.gz'
    tst_x = f'{sys.argv[1]}-test-images-idx3-ubyte.gz'
    tst_y = f'{sys.argv[1]}-test-labels-idx1-ubyte.gz'
    
    tr_x = read_img(tr_x)
    tr_y = read_labels(tr_y)
    
    tst_x = read_img(tst_x)
    tst_y = read_labels(tst_y)
    
    np.savez_compressed('/home/a4k7/custom_datasets/emnist_balanced.npz', a=tr_x, b=tst_x, c=tr_y, d=tst_y)
    """