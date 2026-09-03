import struct
import numpy as np

def load_idx(path):
    with open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        if magic >> 16 != 0:
            raise ValueError(f'{path}: bad magic {magic} - wrong byte order?')
        ndim = magic & 0xFF
        