import keras.backend as K
import numpy as np

def keras_pearsonr(x, y):

    xm = x - K.mean(x)
    ym = y - K.mean(y)

    r_num = K.sum(xm * ym)

    xs = K.sum(K.pow(xm, 2))
    ys = K.sum(K.pow(ym, 2))

    r_den = K.sqrt(xs) * K.sqrt(ys)
    r = r_num / r_den

    return r

def pearsonr(x, y):

    x = np.asarray(x)
    y = np.asarray(y)
    # n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den
    return r
