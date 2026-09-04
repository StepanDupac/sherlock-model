import numpy as np
from PIL import Image

TARGET = 28
INNER = 20
INK_THRESHOLD = 0.55

def preprocess(img, ink_is_high=None):
    a = np.asarray(img.convert('L'), dtype=np.float64)
    if ink_is_high is None:
        ink_is_high = np.median(a) < 128
    if not ink_is_high:
        a = 255.0 - a

    a = np.where(a > a.max() * INK_THRESHOLD, a, 0.0)
    if a.max() <= 0:
        raise ValueError('no ink found in this image')

    rows, cols = np.nonzero(a)
    a = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = a.shape
    scale = INNER / max(h, w)

    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    small = Image.fromarray(a.astype(np.uint8)).resize(new, Image.LANCZOS)

    field = Image.new('L', (TARGET, TARGET), 0)
    field.paste(small, ((TARGET - new[0]) // 2, (TARGET - new[1]) // 2))

    b = np.asarray(field, dtype=np.float64)
    total = b.sum()
    if total <= 0:
        raise ValueError('no ink found in this image')
    ys, xs = np.mgrid[0:TARGET, 0:TARGET]
    cy, cx = (b * ys).sum() / total, (b * xs).sum() / total
    dy, dx = (TARGET - 1) / 2 - cy, (TARGET - 1) / 2 - cx

    field = field.transform((TARGET, TARGET), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                            resample=Image.BILINEAR, fillcolor=0)

    return np.asarray(field, dtype=np.float64).ravel() / 255.0

def to_preview(x784, size=140):
    a = (np.asarray(x784).reshape(TARGET, TARGET) * 255).astype(np.uint8)
    return Image.fromarray(a).resize((size, size), Image.NEAREST)
