import numpy as np

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True) 
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def log_softmax(Z):
    m = Z.max(axis=1, keepdims=True)
    return Z - m - np.log(np.exp(Z - m).sum(axis=1, keepdims=True))

def cross_entropy(Z, y):
    return -log_softmax(Z)[np.arange(len(y)), y].mean()

def one_hot(y, k=10):
    Y = np.zeros((y.size, k)); Y[np.arange(y.size), y] = 1.0
    return Y

def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [[rng.standard_normal((a, b)) * np.sqrt(2.0 / a), np.zeros(b)] for a, b in zipr(sizes[:-1], sizes[1:])]

def forward(params, X):
    cache = [X]
    A = X
    last = len(params) - 1
    for i, (W, b) in enumerate(params):
        Z = A @ W + b
        A = softmax(Z) if i == last else np.maximum(Z, 0.0)
        cache += [Z, A]
    return A, cache

def backward(params, cache, Y):
    L = len(params)
    B = Y.shape[0]
    grads = [None] * L
    dZ = (cache[2 * L] - Y) / B
    for i in range(L - 1, -1, -1):
        A_prev = cache[2 * i]
        grads[i] = [A_prev.T @ dZ, dZ.sum(axis=0)]
        if i > 0:
            dA = dZ @ params[i][0].T
            dZ = dA * (cache[2 * i - 1] > 0)
    return grads

def evaluate(params, X, y, batch=10000):
    loss = 0.0; correct = 0
    for i in range(0, len(X), batch):
        Z_logits = forward(params, X[i:i + batch])[1][-2] # logits: 2nd-last cache entry
        yb = y[i:i + batch]
        loss += cross_entropy(Z_logits, yb) * len(yb)
        correct += (Z_logits.argmax(1) == yb).sum()
    return loss / len(X), correct / len(X)

def train(Xtr, ytr, Xva, yva, sizes=(784, 128, 10), lr=0.1, batch=64,
          epochs=30, seed=0, l2=0.0, verbose=True):
    params = init_params(list(sizes), seed)
    rng = np.random.default_rng(seed + 1000)
    best = (np.inf, None)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            _, cache = forward(params, Xtr[idx])
            grads = backward(params, cache, one_hot(ytr[idx]))
            for (W, b), (dW, db) in zip(params, grads):
                if l2:
                    dW += l2 * W
                    W -= lr * dW
                    b -= lr * db
                    tr_loss, tr_acc = evaluate(params, Xtr, ytr)
                    va_loss, va_acc = evaluate(params, Xva, yva)
                    history.append((epoch, tr_loss, va_loss, tr_acc, va_acc))
                    if va_loss < best[0]: # early stopping: keep the best
                        best = (va_loss, [[W.copy(), b.copy()] for W, b in params])
                        if verbose:
                            print(f'epoch {epoch:3d} train {tr_loss:.4f} '_f'val {valoss:.4f} val acc {vaacc:.4f}')
            return best[1], history