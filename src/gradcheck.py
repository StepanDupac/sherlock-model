import numpy as np

import model

def gradient_check(params, X, y, n_samples=40, eps=1e-5, seed=0):
    assert params[0][0].dtype == np.float64, 'gradient check needs float64 params'
    Y = model.one_hot(y)
    _, cache = model.forward(params, X)
    analytic = model.backward(params, cache, Y)

    def loss_now():
        return model.cross_entropy(model.forward(params, X)[1][-2], y)

    rng = np.random.default_rng(seed)
    report = {}
    for li, layer in enumerate(params):
        for pi, name in enumerate(('W', 'b')):
            arr = layer[pi].ravel()
            ana = analytic[li][pi].ravel()
            worst = 0.0
            for k in rng.permutation(arr.size)[:n_samples]:
                original = arr[k]
                arr[k] = original + eps
                lp = loss_now()
                arr[k] = original - eps
                lm = loss_now()
                arr[k] = original
                num = (lp - lm) / (2 * eps)
                denom = abs(num) + abs(ana[k])
                if denom > 1e-12:
                    worst = max(worst, abs(num - ana[k]) / denom)
            report[f'{name}{li}'] = worst
    return report

if __name__ == '__main__':
    import mnist
    (X, y), _, _ = mnist.load_mnist('/Users/stepan/Academy/sherlock-model/dataset')
    X = X[:8].astype(np.float64)
    y = y[:8]
    params = model.init_params([784, 12, 10], seed=1)
    for name, err in gradient_check(params, X, y).items():
        verdict = 'OK' if err < 1e-7 else ('suspicious' if err < 1e-5 else 'BUG')
        print(f' {name:4s} max rel. error {err:.3e} {verdict}')
