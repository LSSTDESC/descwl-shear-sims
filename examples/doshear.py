import numpy as np
import fitsio
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('flist', nargs='+')
    parser.add_argument('--nsigma', type=int, default=3)
    return parser.parse_args()


def get_sums(data, stype):
    w, = np.where(
        (data['flags'] == 0) &
        (data['shear_type'] == stype)
    )
    g_sum = data['wmom_g'][w].sum(axis=0)
    g2_sum = (data['wmom_g'][w]**2).sum(axis=0)
    return g_sum, g2_sum, w.size


def main():
    args = get_args()

    cols = [
        'flags',
        'shear_type',
        'wmom_g',
    ]

    nf = len(args.flist)

    g_sum = np.zeros(2)
    g2_sum = g_sum.copy()

    g_sum_1p = g_sum.copy()
    g_sum_1m = g_sum.copy()

    n, n_1p, n_1m = 0, 0, 0

    for i, f in enumerate(args.flist):
        print('%d/%d %s' % (i+1, nf, f))
        data = fitsio.read(f, columns=cols)

        tg_sum, tg2_sum, tn = get_sums(data, 'noshear')
        tg_sum_1p, _, tn_1p = get_sums(data, '1p')
        tg_sum_1m, _, tn_1m = get_sums(data, '1m')

        g_sum += tg_sum
        g2_sum += tg2_sum
        g_sum_1p += tg_sum_1p
        g_sum_1m += tg_sum_1m

        n += tn
        n_1p += tn_1p
        n_1m += tn_1m

    g = g_sum/n
    g2 = g2_sum/(n-1)

    g_1p = g_sum_1p/n
    g_1m = g_sum_1m/n

    g_std = np.sqrt(g2 - g**2)
    g_err = g_std/np.sqrt(n)

    R11 = (g_1p[0] - g_1m[0])/0.02
    print('R11:', R11)

    shear1 = g[0]/R11
    shear1_err = g_err[0]/R11

    m = shear1/0.02-1
    m_err = shear1_err/0.02

    tup = (m, m_err*args.nsigma, args.nsigma)
    print('m: %g +/- %g (%d sigma)' % tup)


main()
