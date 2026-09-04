import struct
import numpy as np

def load_idx(path):
    with open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        if magic >> 16 != 0:
            raise ValueError(f'{path}: bad magic {magic} - wrong byte order?')
        ndim = magic & 0xFF
        dims = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size != np.prod(dims):
            raise ValueError(f'{path}: expected {np.prod(dims)} bytes, got {date.size}')
        return data.reshape(dims)

def load_mnist(directory, val_size=5000, seed=42):
    d = directory.rstrip('/')
    Xtr = load_idx(f'{d}/train-images.idx3-ubyte').reshape(-1, 784).astype(np.float32) / 255.0
    ytr = load_idx(f'{d}/train-labels.idx1-ubyte').astype(np.int64)
    Xte = load_idx(f'{d}/t10k-images.idx3-ubyte').reshape(-1, 784).astype(np.float32) / 255.0
    yte = load_idx(f'{d}/t10k-labels.idx1-ubyte').astype(np.int64)
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)
    assert ytr.min() == 0 and ytr.max() == 9

    perm = np.random.default_rng(seed).permutation(len(Xtr))
    va, tr = perm[:val_size], perm[val_size:]
    return (Xtr[tr], ytr[tr]), (Xtr[va], ytr[va]), (Xte, yte)

def ascii_art(img28, labels=' .:+*#%@'):
    img = np.asarray(img28, dtype=float).reshape(28, 28)
    hi = img.max() or 1.0
    scale = len(labels) - 1
    rows = (''.join(labels[int(p / hi * scale)] for p in row) for row in img)
    return '\n'.join(rows)

if __name__ == '__main__':
    (X, y), _, _ = load_mnist('/Users/stepan/Academy/sherlock-model/dataset')
    print(X.shape, y.shape, X.min(), X.max())
    print(f'label = {y[0]}'); print(ascii_art(X[0]))
