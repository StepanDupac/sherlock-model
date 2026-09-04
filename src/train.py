import numpy as np

import gradcheck
import mnist
import model

DATA = '/Users/stepan/Academy/sherlock-model/dataset'

(Xtr, ytr), (Xva, yva), (Xte, yte) = mnist.load_mnist(DATA)
print(f'{len(Xtr)} train / {len(Xva)} val / {len(Xte)} test')

check = gradcheck.gradient_check(model.init_params([784, 12, 10], seed=1),
                                 Xtr[:8].astype(np.float64), ytr[:8])
assert max(check.values()) < 1e-7, f'backward pass is wrong {check}'
print('gradient check passed:', {k: f'{v:.1e}' for k, v in check.items()})

params, history = model.train(Xtr, ytr, Xva, yva, sizes=(784, 128, 10), lr=0.1, batch=64, epochs=30, seed=0)

test_loss, test_acc = model.evaluate(params, Xte, yte)
ci = 1.96 * np.sqrt(test_acc * (1 - test_acc) / len(yte)) # 95% interval, section 2.6
print(f'TEST accuracy {test_acc:.4f} (+/- {ci:.4f})')

best_val = max(h[4] for h in history)
model.save('weights.npz', params, {
    'preprocessing': 'mnist-v1: 28x28, ink=high, x/255',
    'sizes': [784, 128, 10],
    'val_acc': best_val,
    'test_acc': test_acc,
})
print('saved weights.npz')
