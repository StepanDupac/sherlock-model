import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import model
import preprocess

CANVAS = 280
BRUSH = 20
PREVIEW = 140
REJECT_ENTROPY = 1.0

class App:
    def __init__(self, root, weights='weights.npz'):
        self.root = root
        root.title('Numero — digit recogniser')
        self.params, self.meta = model.load(weights)

        left = tk.Frame(root, padx=10, pady=10)
        left.grid(row=0, column=0, sticky='n')
        mid = tk.Frame(root, padx=10, pady=10)
        mid.grid(row=0, column=1, sticky='n')
        right = tk.Frame(root, padx=10, pady=10)
        right.grid(row=0, column=2, sticky='n')

        tk.Label(left, text='DRAW HERE, OR LOAD A FILE', font=('Helvetica', 9, 'bold'), fg='#898781').pack(anchor='w')
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg='black',
                                highlightthickness=1, highlightbackground='#c3c2b7')
        self.canvas.pack(pady=6)
        self.canvas.bind('<Button-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', lambda e: self.predict())

        row = tk.Frame(left)
        row.pack(fill='x')
        tk.Button(row, text='Load image...', command=self.on_load).pack(side='left')
        tk.Button(row, text='Clear', command=self.clear).pack(side='left', padx=6)

        tk.Label(mid, text='WHAT THE MODEL SEES', font=('Helvetica', 9, 'bold'), fg='#898781').pack(anchor='w')
        self.blank = ImageTk.PhotoImage(Image.new('L', (PREVIEW, PREVIEW), 0))
        self.preview = tk.Label(mid, bg='black', image=self.blank)
        self.preview.pack(pady=6)
        tk.Label(mid, justify='left', font=('Helvetica', 9), fg='#52514e',
                 text='28 × 28 after preprocessing.\nIf this looks wrong,\nthe model has no chance.').pack(anchor='w')

        tk.Label(right, text='PREDICTION', font=('Helvetica', 9, 'bold'), fg='#898781').pack(anchor='w')
        self.digit = tk.Label(right, text='–', font=('Helvetica', 52, 'bold'))
        self.digit.pack(anchor='w')
        self.conf = tk.Label(right, text='draw a digit, or load an image', font=('Helvetica', 11), fg='#52514e')
        self.conf.pack(anchor='w')
        self.bars = tk.Canvas(right, width=250, height=110, highlightthickness=0, bg='white')
        self.bars.pack(pady=8)

        acc = float(self.meta.get('val_acc', 0)) * 100
        sizes = '-'.join(map(str, self.meta['sizes']))
        tk.Label(root, text=f'model: {sizes} · validation accuracy {acc:.1f}%',
                 font=('Helvetica', 9), fg='#898781').grid(row=1, column=0, columnspan=3,
                                                           sticky='w', padx=12, pady=(0, 8))

        self.clear()

    def on_press(self, event):
        self.last = (event.x, event.y)

    def on_drag(self, event):
        x, y = event.x, event.y
        self.canvas.create_line(*self.last, x, y, width=BRUSH, fill='white',
                                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
        self.draw.line([self.last, (x, y)], fill=255, width=BRUSH, joint='curve')
        self.draw.ellipse([x - BRUSH // 2, y - BRUSH // 2, x + BRUSH // 2, y + BRUSH // 2], fill=255)
        self.last = (x, y)

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.drawn = True
        self.last = (0, 0)
        self.digit.config(text='–', fg='#0b0b0b')
        self.conf.config(text='draw a digit, or load an image')
        self.bars.delete('all')
        self.preview.config(image=self.blank)
        self.preview._keep = self.blank

    def on_load(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff')])
        if not path:
            return
        self.clear()
        self.image = Image.open(path).convert('L')
        self.draw = ImageDraw.Draw(self.image)
        self.drawn = False
        self.predict()

    def predict(self):
        try:
            x = preprocess.preprocess(self.image, ink_is_high=True if self.drawn else None)
        except ValueError:
            self.conf.config(text='no ink found in this image')
            return

        photo = ImageTk.PhotoImage(preprocess.to_preview(x, PREVIEW))
        self.preview.config(image=photo)
        self.preview._keep = photo

        digit, p = model.predict(self.params, x)
        entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        if entropy > REJECT_ENTROPY:
            self.digit.config(text='?', fg='#898781')
            self.conf.config(text=f'not confident enough (entropy {entropy:.2f})')
        else:
            self.digit.config(text=str(digit), fg='#0b0b0b')
            self.conf.config(text=f'{p[digit] * 100:.1f}% confident · entropy {entropy:.2f}')
        self.draw_bars(p, digit)

    def draw_bars(self, p, winner):
        self.bars.delete('all')
        w, h, base = 20, 80, 92
        for k in range(10):
            x = 8 + k * 24
            self.bars.create_rectangle(x, base - max(1, p[k] * h), x + w, base,
                                       fill='#eb6834' if k == winner else '#2a78d6', width=0)
            self.bars.create_text(x + w / 2, base + 10, text=str(k), font=('Helvetica', 8), fill='#52514e')

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
