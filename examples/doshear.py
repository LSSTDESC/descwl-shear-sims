import numpy as np
import esutil as eu
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('flist', nargs='+')
    parser.add_argument('--nsigma', type=int, default=3)
    return parser.parse_args()


def get_means(data, stype):
    w, = np.where(data['shear_type'] == stype)
    g_mean = data['wmom_g'][w].mean(axis=0)
    g_std = data['wmom_g'][w].std(axis=0)
    return g_mean, g_std, w.size


def main():
    args = get_args()

    # data = fitsio.read(args.fname)
    cols = [
        'shear_type',
        'wmom_g',
    ]

    print(args.flist)
    data = eu.io.read(args.flist, columns=cols)

    g_mean, g_std, num = get_means(data, 'noshear')
    g_mean_1p, _, _ = get_means(data, '1p')
    g_mean_1m, _, _ = get_means(data, '1m')

    R11 = (g_mean_1p[0] - g_mean_1m[0])/0.02

    shear1 = g_mean[0]/R11
    shear1_err = g_std[0]/np.sqrt(num)/R11

    m = shear1/0.02-1
    m_err = shear1_err/0.02

    tup = (m, m_err*args.nsigma, args.nsigma)
    print('m: %g +/- %g (%d sigma)' % tup)


main()
