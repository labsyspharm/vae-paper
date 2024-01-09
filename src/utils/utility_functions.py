import os
import re
import shutil

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm
from math import ceil
from decimal import Decimal
from itertools import combinations

import seaborn as sns
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from skimage.color import gray2rgb

from scipy.fft import fft2
from scipy.stats import binned_statistic
from skimage.metrics import mean_squared_error
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim

import zarr
import tifffile
import dask.array as da

from scipy.stats import tstd
from scipy.stats import kstest
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import tukey_hsd
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cityblock


def transposeZarr(z):
    """rearrange Zarr dimensions to fit shape of expected VAE input
    (i.e. cells, x, y, channels). Returns a Dask array."""
    z = da.from_zarr(z).transpose([1, 2, 3, 0])

    return z


def log_transform(img_batch):
    """Log-transform uint16 image patch pixel values."""
    
    img_batch = np.log10(img_batch.astype('float32') + 1)
    
    return img_batch


def clip_outlier_pixels(img_batch, percentile_cutoffs):
    """Clip lower and upper percentile outliers to 0 or 1, respectively
    based on pixel intensities of whole image."""
    pc = np.array(list(percentile_cutoffs.values()), np.float32)
    pc_min = pc[:, 0]
    pc_max = pc[:, 1]
    img_batch = np.clip((img_batch - pc_min) / (pc_max - pc_min), 0, 1)

    return img_batch


def compute_vignette_mask(img_batch, kernel_size=40):
    """Compute a 2D Gaussian-distributed vignette mask to apply to image patches."""
    kernel_y = cv2.getGaussianKernel(img_batch.shape[1], kernel_size).astype('float32') 
    kernel_x = cv2.getGaussianKernel(img_batch.shape[2], kernel_size).astype('float32') 
    kernel = kernel_y * kernel_x.T

    # normalize kernel to be between 0 and 1
    mask = cv2.normalize(kernel, None, 0, 1, cv2.NORM_MINMAX)
    mask = mask[np.newaxis, :, :, np.newaxis]  # adding a third dimension to mask (i.e. X, X, 1)

    return mask


def reverse_processing(percentile_cutoffs, channel_slice, channel_name, contrast_limits):
    """Reverses percentile normalization and log10-transformation,
       pixel outliers remained clipped)."""

    lower_cutoff_log, upper_cutoff_log = percentile_cutoffs[channel_name]

    # reverse percentile normalization
    channel_slice = (
        (((upper_cutoff_log - lower_cutoff_log) * (channel_slice - 0)) /
         (1 - 0)) + lower_cutoff_log
    )

    # reverse log10-transform
    channel_slice = np.rint(10 ** channel_slice)

    # Normalize pixel values between lower and upper percentile bounds
    # lower = np.rint(10**lower_cutoff_log)
    # upper = np.rint(10**upper_cutoff_log)
    # channel_slice = (channel_slice-lower) / (upper-lower)

    # Normalize pixel values between lower and upper contrast settings
    lower = contrast_limits[channel_name][0]
    upper = contrast_limits[channel_name][1]
    channel_slice = (channel_slice - lower) / (upper - lower)

    return channel_slice


def kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def u_stats(df, metric, cluster_select, expressed_channels, markers, combo_name):
    """Compute Mann-Whitney U-test between similarity
       metrics of two clusters."""

    stats = pd.DataFrame()
    for ch in expressed_channels.keys():

        channel_data = df[df['marker'] == cluster_select[markers[ch]] + ', ' + markers[ch]]

        data1 = channel_data[metric][channel_data['cluster'] != combo_name].astype('float')
        data2 = channel_data[metric][channel_data['cluster'] == combo_name].astype('float')

        # grab marker index for plotting
        plot_ch = expressed_channels[ch][1]

        # compute Mann-Whitney U-test
        uval, pval = mannwhitneyu(
            x=data1, y=data2, use_continuity=True, alternative='two-sided',
            axis=0, method='auto', nan_policy='propagate'
        )

        # append current stat to dataframe
        u_row = pd.DataFrame(
            index=[markers[ch]], data=[[uval, pval, plot_ch]],
            columns=['u-stat', 'pval', 'plot_ch'])
        stats = pd.concat([stats, u_row], ignore_index=False)
    print(f'{metric} U-stats:')
    print(stats)
    print()

    return stats


def hsd(df, metric):
    """Compute one-way ANOVA and Tukey HSD stats between
       three groups of cluster similarity metrics."""

    hsds = pd.DataFrame()
    for ch in expressed_channels.keys():
        clus1_data = df[metric][
            (df['cluster'] == clus_pair[0]) & (df['marker'] == markers[ch])].astype('float')
        clus2_data = df[metric][
            (df['cluster'] == clus_pair[1]) & (df['marker'] == markers[ch])].astype('float')
        combo_data = df[metric][
            (df['cluster'] == combo_name) & (df['marker'] == markers[ch])].astype('float')

        # compute one-way ANOVA
        fval, pval = f_oneway(clus1_data, clus2_data, combo_data)
        print(
            f"{metric} F-test pval is {'%.2E' % Decimal(pval)} for {markers[ch]}")

        if pval <= 0.05:
            # perform multiple pairwise comparisons (Tukey's HSD)
            # and store in df
            res = tukey_hsd(list(clus1_data), list(clus2_data), list(combo_data))
            res_df = pd.DataFrame(
                index=[clus_pair[0], clus_pair[1], combo_name],
                columns=[clus_pair[0], clus_pair[1], combo_name],
                data=res.pvalue)
            hsd = (
                res_df
                .mask(np.tril(np.ones(res_df.shape)).astype(bool))
                .stack()
                .reset_index()
                .rename(columns={0: 'pval'})
            )
            hsd['comparison'] = hsd['level_0'].astype(str) + '/' + hsd['level_1'].astype(str)
            hsd.drop(columns=['level_0', 'level_1'], inplace=True)
            hsd['marker'] = expressed_channels[ch][0]
            hsd['plot_ch'] = expressed_channels[ch][1]
            hsds = pd.concat([hsds, hsd], ignore_index=False)
    hsds.reset_index(drop=True, inplace=True)
    print()
    print(f'{metric} HSD-stats:')
    print(hsds)

    return hsds


def compare_clusters(clus1_name, clus1, clus1_ids, clus2_name, clus2, clus2_ids, metric, expressed_channels, markers, X_combo, combo_name, save_dir):
    """Compute similarity metrics for two clusters of image patches."""

    print()
    cluster_select = {}
    df = pd.DataFrame(columns=['cluster', 'marker', metric])
    for ch in expressed_channels.keys():
        channel_df = pd.DataFrame(columns=['cluster', 'marker', metric])
        for clus, clus_ids in zip([clus1_name, clus2_name], [clus1, clus2]):
            print(
                f'Computing {metric} for Cluster: {clus}, Channel: {markers[ch]}'
            )
            cluster_df = pd.DataFrame(columns=['cluster', 'marker', metric])
            measurements_within = []
            for combo in tqdm(
                    combinations(clus_ids, 2), total=len(list(combinations(clus_ids, 2)))):
                img1 = X_combo[ch, combo[0]]
                img2 = X_combo[ch, combo[1]]
                img1 = (
                    (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)).astype('float32')
                )
                img2 = (
                    (img2 - np.min(img2)) / (np.max(img2) - np.min(img2)).astype('float32')
                )

                if metric == 'SSIM':
                    sm = SSIM(img1, img2)
                    measurements_within.append(sm)

                elif metric == 'LBP':
                    hist1, hist2 = LBP(img1, img2)
                    cb_dist = cityblock(u=hist1, v=hist2)
                    measurements_within.append(cb_dist)

                    # kld_score = kl_divergence(hist1, hist2)
                    # measurements_within.append(kld_score)

                    # ks_score, kspval = kstest(
                    #     rvs=hist1, cdf=hist2, alternative='two-sided'
                    #     )
                    # measurements_within.append(ks_score)

                elif metric == 'ORB':
                    ob = ORB(img1, img2)
                    measurements_within.append(ob)

                elif metric == 'FFT':
                    min_mse = FFT(img1, img2)
                    measurements_within.append(min_mse)

                    # cb_dist = cityblock(u=hists[0], v=hists[1])
                    # measurements_within.append(cb_dist)

                    # corr = pearsonr(x=Abs[0], y=Abs[1])[0]
                    # measurements_within.append(corr)

                elif metric == 'STD':
                    score = STD(img1, img2)
                    measurements_within.append(score)

                elif metric == 'MSE':
                    mse = MSE(img1, img2)
                    measurements_within.append(mse)

            cluster_df[metric] = measurements_within
            cluster_df['cluster'] = clus
            cluster_df['marker'] = markers[ch]
            cluster_df['channel'] = ch
            channel_df = pd.concat([channel_df, cluster_df], axis=0)

        # select cluster with smaller median
        clus1_data = channel_df[metric][
            (channel_df['cluster'] == clus1_name) &
            (channel_df['marker'] == markers[ch])
        ].astype('float')
        clus2_data = channel_df[metric][
            (channel_df['cluster'] == clus2_name) &
            (channel_df['marker'] == markers[ch])
        ].astype('float')

        mean1 = np.median(clus1_data)
        mean2 = np.median(clus2_data)

        if mean1 <= mean2:
            channel_df = channel_df[channel_df['cluster'] == clus1_name]
            channel_df['marker'] = str(clus1_name) + ', ' + markers[ch]
            cluster_select[markers[ch]] = str(clus1_name)
        else:
            channel_df = channel_df[channel_df['cluster'] == clus2_name]
            channel_df['marker'] = str(clus2_name) + ', ' + markers[ch]
            cluster_select[markers[ch]] = str(clus2_name)

        # select cluster with smaller SD
        # lower = np.percentile(clus1_data, 5.0)
        # upper = np.percentile(clus1_data, 95.0)
        # std1 = tstd(clus1_data, limits=(lower, upper))

        # lower = np.percentile(clus2_data, 5.0)
        # upper = np.percentile(clus2_data, 95.0)
        # std2 = tstd(clus2_data, limits=(lower, upper))

        # if std1 <= std2:
        #     channel_df = channel_df[channel_df['cluster'] == clus1_name]
        #     channel_df['marker'] = str(clus1_name) + ', ' + markers[ch]
        #     cluster_select[markers[ch]] = str(clus1_name)
        # else:
        #     channel_df = channel_df[channel_df['cluster'] == clus2_name]
        #     channel_df['marker'] = str(clus2_name) + ', ' + markers[ch]
        #     cluster_select[markers[ch]] = str(clus2_name)

        df = pd.concat([df, channel_df], axis=0)
        print()

    sq_err = {}
    for ch in expressed_channels.keys():
        print(
            f'Computing {metric} across clusters {combo_name} for '
            f'channel: {markers[ch]}')

        z1 = zarr.open(
            os.path.join(save_dir, 'error1.zarr'), mode='w',
            shape=(30, 30, len(clus1) * len(clus2)),
            chunks=(30, 30, 200), dtype='float32'
        )
        z2 = zarr.open(
            os.path.join(save_dir, 'error2.zarr'), mode='w',
            shape=(30, 30, len(clus1) * len(clus2)),
            chunks=(30, 30, 200), dtype='float32'
        )
        temp_df = pd.DataFrame(columns=['cluster', 'marker', metric])
        measurements_across = []
        err_counter = 0
        for i in tqdm(clus1):
            for j in clus2:
                img1 = X_combo[ch, i]
                img2 = X_combo[ch, j]

                img1 = (
                    (img1 - np.min(img1)) / (np.max(img1) - np.min(img1)).astype('float32')
                )
                img2 = (
                    (img2 - np.min(img2)) / (np.max(img2) - np.min(img2)).astype('float32')
                )

                # compute squared error image array every 100 iterations
                if err_counter % 100 == 0:

                    err1 = (img1 - img2)
                    err1[err1 < 0] = 0
                    err1 = err1**2
                    z1[:, :, err_counter] = err1

                    err2 = (img2 - img1)
                    err2[err2 < 0] = 0
                    err2 = err2**2
                    z2[:, :, err_counter] = err2

                err_counter += 1

                if metric == 'SSIM':
                    # compute structural similarity index measures
                    # for images within current cluster
                    sm = SSIM(img1, img2)
                    measurements_across.append(sm)

                elif metric == 'LBP':
                    hist1, hist2 = LBP(img1, img2)
                    cb_dist = cityblock(u=hist1, v=hist2)
                    measurements_across.append(cb_dist)

                    # kld_score = kl_divergence(hist1, hist2)
                    # measurements_across.append(kld_score)

                    # ks_score, kspval = kstest(
                    #     rvs=hist1, cdf=hist2, alternative='two-sided'
                    #     )
                    # measurements_across.append(ks_score)

                elif metric == 'ORB':
                    ob = ORB(img1, img2)
                    measurements_across.append(ob)

                elif metric == 'FFT':
                    img1t, img2t, hists, Abs = FFT(img1, img2)
                    measurements_across.append(np.sum((abs(img1t - img2t))))

                    # cb_dist = cityblock(u=hists[0], v=hists[1])
                    # measurements_across.append(cb_dist)
                    #
                    # corr = pearsonr(x=Abs[0], y=Abs[1])[0]
                    # measurements_across.append(corr)

                elif metric == 'STD':
                    score = STD(img1, img2)
                    measurements_across.append(score)

                elif metric == 'MSE':
                    mse = MSE(img1, img2)
                    measurements_across.append(mse)

        temp_df[metric] = measurements_across
        temp_df['cluster'] = combo_name
        temp_df['marker'] = cluster_select[markers[ch]] + ', ' + markers[ch]
        temp_df['channel'] = ch
        df = pd.concat([df, temp_df], axis=0)

        sq_err[markers[ch]] = (np.mean(z1, axis=2), np.mean(z2, axis=2))

        shutil.rmtree(os.path.join(save_dir, 'error1.zarr'))
        shutil.rmtree(os.path.join(save_dir, 'error2.zarr'))
    print()

    return df, cluster_select, sq_err


def plot(df, metric, stats, color, sq_err, combo_name, clus_pair, save_dir):
    """Plot similarity metric data."""

    df['cluster'] = ['ref.' if i != combo_name else i for i in df['cluster']]
    df['cluster'] = df['cluster'].astype('str')

    g = sns.catplot(
        data=df, x='marker', y=metric, hue='cluster', kind='boxen',
        palette=['grey', color], aspect=1.5, legend=True
    )
    g.set(ylim=(None, None))
    g.set_xticklabels(fontsize=15, weight='normal', rotation=45)
    g.set_yticklabels(fontsize=12, weight='normal')
    g.set(xlabel=None)

    for ax in g.axes.flat:
        labels = [i.get_text() for i in ax.get_xticklabels()]
        labels = ['Ref.' + i.split(', ')[0] + ', ' + i.split(', ')[1] for i in labels]
        g.set_xticklabels(labels, fontsize=15, weight='normal', rotation=45)
        if metric == 'LBP':
            ax.set_ylabel('L1 distance', fontsize=20, weight='normal')
        elif metric == 'FFT':
            ax.set_ylabel('\u03A3|img1t-img2t|', fontsize=20, weight='normal')
        else:
            ax.set_ylabel(ax.get_ylabel(), fontsize=20, weight='normal')
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', borderaxespad=0)

    # add t-stats to figure
    for row in stats.iterrows():
        if row[1]['pval'] < 0.05:
            pval = '%.2E' % Decimal(f"{row[1]['pval']}")
            g.ax.text(
                x=row[1]['plot_ch'], y=g.ax.get_ylim()[1] * 0.95,
                s=f'U-test p={pval}',
                horizontalalignment='center', verticalalignment='center',
                size=7, color='black', weight='normal')

    # add HSD stats to figure
    # for ch, group in hsds.groupby('plot_ch'):
    #     spacer = 0
    #     for i in group.iterrows():
    #         if i[1]['pval'] < 0.05:
    #             pval = '%.2E' % Decimal(f"{i[1]['pval']}")
    #             g.ax.text(
    #                 x=ch, y=(0.9 - spacer) * g.ax.get_ylim()[1],
    #                 s=f"{i[1]['comparison']} HSD p={pval}",
    #                 horizontalalignment='center', verticalalignment='center',
    #                 size=7, color='black', weight='normal')
    #             spacer += 0.05

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric}_boxenplots.pdf'))
    plt.show()
    plt.close('all')

    # hists
    g = sns.displot(
        data=df, x=metric, hue='cluster', col='marker', kind='kde',
        aspect=1.5, palette=['grey', color], lw=5
    )
    for ax in g.axes.flat:
        if metric == 'LBP':
            ax.set_xlabel('L1 distance', fontsize=15, weight='normal')
        elif metric == 'FFT':
            ax.set_ylabel('\u03A3|img1t-img2t|', fontsize=20, weight='normal')
        else:
            ax.set_xlabel(ax.get_xlabel(), fontsize=15, weight='normal')
        ax.set_ylabel(ax.get_ylabel(), fontsize=15, weight='normal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric}_kdeplots.pdf'))
    plt.show()
    plt.close('all')

    # squared error
    fig = plt.figure(figsize=(10, 5))
    rows = 1
    cols = len(sq_err.keys())
    intensity_multiplier = 1.25
    outer = gridspec.GridSpec(1, 21, wspace=0.1, hspace=0.0)
    for e, (marker, img) in enumerate(sq_err.items()):

        ax = plt.Subplot(fig, outer[e])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        overlay = np.zeros((img[0].shape[0], img[0].shape[1]))
        overlay = gray2rgb(overlay)

        img1 = (img[0] - np.min(img[0])) / (np.max(img[0]) - np.min(img[0]))
        img1 = gray2rgb(img1)
        img1 = img1 * intensity_multiplier

        img2 = (img[1] - np.min(img[1])) / (np.max(img[1]) - np.min(img[1]))
        img2 = gray2rgb(img2)
        img2 = img2 * intensity_multiplier

        legend_elements = []
        for name, im, color in zip(
                [clus_pair[0], clus_pair[1]], [img1, img2], ['cornflowerblue', 'yellow']):
            overlay += im * to_rgb(color)
            legend_elements.append(Line2D([0], [0], color=color, lw=1, label=name))

        ax.imshow(overlay)

        ax.set_title(marker, fontsize=2, pad=1)
        fig.add_subplot(ax)

    # add legend to last plot
    leg = ax.legend(
        handles=legend_elements, prop={'size': 1}, bbox_to_anchor=(1.6, 1.0))
    leg.get_frame().set_linewidth(0.0)
    
    plt.savefig(
        os.path.join(save_dir, f'{metric}_sq_error.png'), dpi=800, bbox_inches='tight'
    )
    plt.show()
    plt.close('all')


def FFT(img1, img2):
    """Compute fast Fourier transform
       for images within current cluster"""

    img1t = fft2(img1)
    img1t = img1t.astype('float')

    results = []
    for k in [0, 1, 2, 3]:
        img3 = np.rot90(img2, k=k, axes=(0, 1))

        img3t = fft2(img3)
        img3t = img3t.astype('float')

        # l1 = np.sum((abs(img1t-img3t)))
        mse = mean_squared_error(img1t, img3t)
        results.append(mse)
    min_mse = min(results)

    return min_mse

    # hists = []
    # Abs = []
    # for im in [img1, img2]:
    #     npix = im.shape[0]
    #
    #     fourier_image = np.fft.fftn(im)
    #     fourier_amplitudes = np.abs(fourier_image)**2
    #
    #     kfreq = np.fft.fftfreq(npix) * npix
    #     kfreq2D = np.meshgrid(kfreq, kfreq)
    #     knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    #
    #     knrm = knrm.flatten()
    #     fourier_amplitudes = fourier_amplitudes.flatten()
    #
    #     kbins = np.arange(0.5, npix//2+1, 1.)
    #     kvals = 0.5 * (kbins[1:] + kbins[:-1])
    #     Abins, _, _ = binned_statistic(
    #         knrm, fourier_amplitudes, statistic='mean',
    #         bins=kbins
    #         )
    #     Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    #
    #     # plt.loglog(kvals, Abins)
    #     # plt.xlabel("$k$")
    #     # plt.ylabel("$P(k)$")
    #
    #     hist, _ = np.histogram(
    #         fourier_amplitudes, density=True, bins=kbins,
    #         range=(0, kbins)
    #         )
    #
    #     hists.append(hist)
    #     Abs.append(Abins)


def LBP(img1, img2):
    """Compute local binary patterns
       for images within current cluster"""

    METHOD = 'uniform'
    radius = 4
    n_points = 8 * radius
    lbp1 = local_binary_pattern(img1, n_points, radius, METHOD)
    n_bins = int(lbp1.max() + 1)
    hist1, _ = np.histogram(lbp1, density=True, bins=n_bins, range=(0, n_bins))
    lbp2 = local_binary_pattern(img2, n_points, radius, METHOD)
    hist2, _ = np.histogram(lbp2, density=True, bins=n_bins, range=(0, n_bins))
    return hist1, hist2


def MSE(img1, img2):
    results = []
    for k in [0, 1, 2, 3]:
        img3 = np.rot90(img2, k=k, axes=(0, 1))
        mse = mean_squared_error(img1, img3)
        results.append(mse)
    min_mse = min(results)

    return min_mse


def ORB(img1, img2):
    """Compute ORB similarity
       for images within current cluster"""

    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matching
    matches = bf.match(desc_a, desc_b)

    if len(matches) == 0:
        ob = 0
    else:
        # look for similar regions with distance < cutoff
        # (range: 0 to 100)
        cutoff = 50
        similar_regions = [i for i in matches if i.distance < cutoff]
        ob = len(similar_regions) / len(matches)

    return ob


def SSIM(img1, img2):
    """compute structural similarity index measures for images
       within current cluster"""

    sm = ssim(img1, img2)

    return sm


def STD(img1, img2):
    bim1 = np.where(img1 > 0.5, 1.0, (np.where(img1 < 0.2, 0.0, 0.5)))
    bim2 = np.where(img2 > 0.5, 1.0, (np.where(img2 < 0.2, 0.0, 0.5)))

    results = []
    for k in [0, 1, 2, 3]:
        bim3 = np.rot90(bim2, k=k, axes=(0, 1))
        res = bim1 + bim3
        res[res == 0] = 2
        res[res == 1] = -2
        results.append(np.sum(res))
    score = max(results)

    return score


def single_channel_pyramid(tiff_path, channel):
    tiff = tifffile.TiffFile(tiff_path)

    if 'Faas' not in tiff.pages[0].software:

        if len(tiff.series[0].levels) > 1:

            pyramid = [
                zarr.open(s[channel].aszarr()) for s in tiff.series[0].levels
            ]

            pyramid = [da.from_zarr(z) for z in pyramid]

            min_val = pyramid[0].min()
            max_val = pyramid[0].max()
            vmin, vmax = da.compute(min_val, max_val)

        else:

            img = tiff.pages[channel].asarray()

            pyramid = [img[::4**i, ::4**i] for i in range(4)]

            pyramid = [da.from_array(z) for z in pyramid]

            min_val = pyramid[0].min()
            max_val = pyramid[0].max()
            vmin, vmax = da.compute(min_val, max_val)

        return pyramid, vmin, vmax

    else:  # support legacy OME-TIFF format

        if len(tiff.series) > 1:

            pyramid = [zarr.open(s[channel].aszarr()) for s in tiff.series]

            pyramid = [da.from_zarr(z) for z in pyramid]

            min_val = pyramid[0].min()
            max_val = pyramid[0].max()
            vmin, vmax = da.compute(min_val, max_val)

        else:
            img = tiff.pages[channel].asarray()

            pyramid = [img[::4**i, ::4**i] for i in range(4)]

            pyramid = [da.from_array(z) for z in pyramid]

            min_val = pyramid[0].min()
            max_val = pyramid[0].max()
            vmin, vmax = da.compute(min_val, max_val)

        return pyramid, vmin, vmax


def read_markers(markers_filepath, markers_to_exclude, data):
    markers = pd.read_csv(markers_filepath, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#')
    if data is None:
        markers_to_include = [
            i for i in markers['marker_name']
            if i not in markers_to_exclude
        ]
    else:
        markers_to_include = [
            i for i in markers['marker_name']
            if i not in markers_to_exclude if i in data.columns
        ]

    markers = markers[markers['marker_name'].isin(markers_to_include)]

    dna1 = markers['marker_name'][markers['channel_number'] == 1][0]
    dna_moniker = str(re.search(r'[^\W\d]+', dna1).group())

    # abx channels
    abx_channels = [
        i for i in markers['marker_name'] if dna_moniker not in i
    ]

    return markers, dna1, dna_moniker, abx_channels


def categorical_cmap(numUniqueSamples, numCatagories, cmap='tab10', continuous=False):
    numSubcatagories = ceil(numUniqueSamples / numCatagories)

    if numCatagories > plt.get_cmap(cmap).N:
        raise ValueError('Too many categories for colormap.')
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, numCatagories))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(numCatagories, dtype=int))
        # rearrange hue order to taste
        cd = {
            'B': 0, 'O': 1, 'G': 2, 'R': 3, 'Pu': 4,
            'Br': 5, 'Pi': 6, 'Gr': 7, 'Y': 8, 'Cy': 9,
        }
        myorder = [
            cd['B'], cd['O'], cd['G'], cd['Pu'], cd['Y'],
            cd['R'], cd['Cy'], cd['Br'], cd['Gr'], cd['Pi']
        ]
        ccolors = [ccolors[i] for i in myorder]

        # use Okabe and Ito color-safe palette for first 6 colors
        # ccolors[0] = np.array([0.91, 0.29, 0.235]) #E84A3C
        # ccolors[1] = np.array([0.18, 0.16, 0.15]) #2E2926
        ccolors[0] = np.array([0.0, 0.447, 0.698, 1.0])  # blue
        ccolors[1] = np.array([0.902, 0.624, 0.0, 1.0])  # orange
        ccolors[2] = np.array([0.0, 0.620, 0.451, 1.0])  # bluish green
        ccolors[3] = np.array([0.8, 0.475, 0.655, 1.0])  # reddish purple
        ccolors[4] = np.array([0.941, 0.894, 0.259, 1.0])  # yellow
        ccolors[5] = np.array([0.835, 0.369, 0.0, 1.0])  # vermillion

    cols = np.zeros((numCatagories * numSubcatagories, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, numSubcatagories).reshape(numSubcatagories, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, numSubcatagories)
        arhsv[:, 2] = np.linspace(chsv[2], 1, numSubcatagories)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * numSubcatagories:(i + 1) * numSubcatagories, :] = rgb
    cmap = colors.ListedColormap(cols)

    # trim colors if necessary
    if len(cmap.colors) > numUniqueSamples:
        trim = len(cmap.colors) - numUniqueSamples
        cmap_colors = cmap.colors[:-trim]
        cmap = colors.ListedColormap(cmap_colors, name='from_list', N=None)

    return cmap


def makeColors(vals):
    colors = np.zeros((len(vals), 3))
    norm = Normalize(vmin=vals.min(), vmax=vals.max())

    # put any colormap you like here
    colors = [
        cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals
    ]

    return colors


def PlotReconstructedImages(orig_input_dims, percentile_cutoffs, contrast_limits, decoder, X, X_seg, X_encoded, y, numColumns, channel_color_dict, intensity_multiplier, thumbnail_font_size, filename, save_dir, mask, undo_mask):

    numSamples = len(X)
    numRows = ceil(numSamples / numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0)

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)
        
        if panel == 0:
            ax = plt.Subplot(fig, outer[panel])
            ax.set_title('Input Images', fontsize=7)
        elif panel == 1:
            ax = plt.Subplot(fig, outer[panel])
            ax.set_title('Learned Representations', fontsize=7)
        
        ax.axis('off')
        fig.add_subplot(ax)
        
        for e, (trans, encode, label, seg) in enumerate(zip(X, X_encoded, y.items(), X_seg)):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # select segmentation outlines slice
            seg_slice = seg[:, :, 0]

            # ensure segmentation outlines are normalized 0-1
            seg_slice = (seg_slice - np.min(seg_slice)) / np.ptp(seg_slice)

            # convert segmentation thumbnail to RGB
            # and add to blank image
            seg_slice = gray2rgb(seg_slice) * 0.25  # decrease alpha

            if panel == 0:
                
                if undo_mask:  # undo vignette mask
                    # mask function is designed to be applied to all cells in a batch during model
                    # training. Slicing first dimension in this case to apply to a single patch.
                    trans /= mask[0, :, :, :]
                
                overlay = np.zeros((trans.shape[0], trans.shape[1]))

                # add centroid point at the center of the image
                overlay[
                    int(trans.shape[0] / 2):int(trans.shape[0] / 2) + 1,
                    int(trans.shape[1] / 2):int(trans.shape[1] / 2) + 1
                ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():
    
                    channel_slice = trans[:, :, ch]

                    channel_slice = reverse_processing(
                        percentile_cutoffs, channel_slice, name, contrast_limits
                    )

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice.compute() * to_rgb(color)

                overlay += seg_slice.compute()

            elif panel == 1:

                z_sample = np.array([encode])

                x_decoded = decoder.predict(z_sample, verbose=0)

                reconstructed_img = x_decoded.reshape(
                    orig_input_dims[0], orig_input_dims[1], orig_input_dims[2])

                overlay = np.zeros((reconstructed_img.shape[0], reconstructed_img.shape[1]))

                # add centroid point at the center of the image
                overlay[
                    int(reconstructed_img.shape[0] / 2):int(
                        reconstructed_img.shape[0] / 2) + 1,
                    int(reconstructed_img.shape[1] / 2):int(
                        reconstructed_img.shape[1] / 2) + 1
                ] = 1

                overlay = gray2rgb(overlay)

                for name, (ch, color) in channel_color_dict.items():

                    channel_slice = reconstructed_img[:, :, ch]

                    channel_slice = reverse_processing(
                        percentile_cutoffs, channel_slice, name, contrast_limits
                    )

                    channel_slice = gray2rgb(channel_slice)

                    channel_slice = channel_slice * intensity_multiplier

                    overlay += channel_slice * to_rgb(color)

                overlay += seg_slice.compute()
            
            overlay = np.clip(overlay, 0, 1)  # avoiding matplotlib warning about clipping
            ax.imshow(overlay, cmap=plt.cm.binary)

            fig.add_subplot(ax)

    fig.suptitle(filename, x=0.0, y=1.03, ha='left', va='center', size=9)
    fig.subplots_adjust(bottom=0.01, top=0.94, left=0.01, right=0.85, wspace=0.2, hspace=0.1)
    
    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))

    fig.legend(
        handles=legend_elements, prop={'size': 5}, bbox_to_anchor=(0.98, 0.93))

    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800, bbox_inches='tight')
    plt.show()
    plt.close('all')
