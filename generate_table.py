import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
import astropy.constants as c

import random
import os
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import time
import sys
import gc
from joblib import Parallel, delayed


def main():
    
    print('Running... This may take a second. Double check N-tracks and grid-spacing!')
    
    def kauf_line():
        x=np.linspace(-2.5, 0, 100)
        y = (0.61/(x - 0.05) + 1.3)
        return x, y

    def kewl_line():
        x=np.linspace(-2.5, 0.4, 100)
        y = (0.61/(x - 0.47) + 1.19)
        return x, y

    def law_line():
        x=np.linspace(-0.24, 0.5)
        y = 0.95*x+0.56
        return x, y
    
    # use these to generate points to cover a larger grid space
    def kauf_line_alt(x):
        #x=np.linspace(-2.5, -0.75, 100)
        y = (0.61/(x - (0.05-0.75)) + (1.3-0.75))
        return y
    def kewl_line_alt(x):
        #x=np.linspace(-2.5, 0.65, 100)
        y = (0.61/(x - (0.47+1.0)) + (1.19+1.0))
        return y

    
    def get_data_file_path(filename):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.realpath(filename))

        # Construct the path to the data file relative to the script location
        data_file_path = os.path.join(script_dir, 'data', filename)

        return data_file_path

    def gen_bounds(real_points, func, num_random_points=300):
        x_real = [point[0] for point in real_points]
        y_real = [point[1] for point in real_points]

        # Calculate deltas and cumulative distances
        d = []
        for i in range(1, len(real_points)):
            delta = (x_real[i] - x_real[i-1])**2 + (y_real[i] - y_real[i-1])**2
            d.append(delta)

        # Calculate cumulative distances
        cumulative_d = np.zeros(len(d))
        cumulative_d[0] = d[0]
        for i in range(1, len(d)):
            cumulative_d[i] = cumulative_d[i-1] + d[i]

        total_length = cumulative_d[-1]

        random_points = []

        for _ in range(num_random_points):
            # Generate a random distance
            r = np.random.uniform(0, total_length)

            # Find which segment this distance falls into
            for i in range(len(cumulative_d)):
                if r <= cumulative_d[i]:
                    # Interpolate between the two points (i-1 and i)
                    if i == 0:
                        segment_start = 0
                    else:
                        segment_start = cumulative_d[i-1]

                    segment_length = cumulative_d[i] - segment_start
                    fraction_of_segment = (r - segment_start) / segment_length

                    # Calculate the random point (xr, yr)
                    xr = x_real[i-1] + fraction_of_segment * (x_real[i] - x_real[i-1])
                    yr = y_real[i-1] + fraction_of_segment * (y_real[i] - y_real[i-1])

                    # Instead, we could also use the line function for y
                    #yr = kewl_line_alt(xr)
                    yr = func(xr)

                    random_points.append((xr, yr))
                    break

        return random_points


    
    # read in BPT data
    eboss = pd.read_csv(get_data_file_path('eboss_bpt.csv'))
    dfsf = eboss.loc[eboss['gal'] == 'sf']
    dfagn = eboss.loc[(eboss['gal'] == 'agn') | (eboss['gal'] == 'lmagn')]
    dfr = eboss.loc[(eboss['gal'] != 'sf') | (eboss['gal'] != 'agn') | (eboss['gal'] != 'lmagn')]
    
    #plt.scatter(dfr['x'], dfr['y'], c='coral', s=1)
    plt.scatter(dfsf['x'], dfsf['y'], c='b', s=0.1)
    plt.scatter(dfagn['x'], dfagn['y'], c='r', s=0.1)
    plt.plot(kewl_line()[0], kewl_line()[1], c='k')
    plt.plot(kauf_line()[0], kauf_line()[1], c='g')
    plt.plot(law_line()[0], law_line()[1], c='k')
    plt.xlim(-2.5, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel(r'log [NII]/H$\alpha$')
    plt.ylabel(r'log [OIII]/H$\beta$')
    plt.savefig('plots/bpt_reg.pdf')
    #plt.show()

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

    ax[0].hist2d(dfsf['x'], dfsf['y'], bins=100, cmap='Blues')
    ax[0].plot(kewl_line()[0], kewl_line()[1], c='k')
    ax[0].plot(kauf_line()[0], kauf_line()[1], c='k')
    ax[0].plot(law_line()[0], law_line()[1], c='k')
    ax[0].set_xlabel(r'log [NII]/H$\alpha$')
    ax[0].set_ylabel(r'log [OIII]/H$\beta$')
    ax[0].set_title('SFG Sequence')

    ax[1].hist2d(dfagn['x'], dfagn['y'], bins=100, cmap='Reds')
    ax[1].plot(kewl_line()[0], kewl_line()[1], c='k')
    ax[1].plot(kauf_line()[0], kauf_line()[1], c='k')
    ax[1].plot(law_line()[0], law_line()[1], c='k')
    ax[1].set_xlim(-2, 0.65)
    ax[1].set_ylim(-1.25, 1.25)
    ax[1].set_xlabel(r'log [NII]/H$\alpha$')
    ax[1].set_title('AGN Sequence')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('plots/bpt_2dhist.pdf')
    #plt.show()

    # these are points in extreme grid space to add
    N_add = 300

    x_kauf = np.linspace(-2.5, -1, N_add)
    real_kauf = [(x, kauf_line_alt(x)) for x in x_kauf]
    random_kauf = gen_bounds(real_kauf, kauf_line_alt, num_random_points=N_add)
    sfg_add_x, sfg_add_y = zip(*sorted(random_kauf, key=lambda x: x[0]))
    
    x_kewl = np.linspace(-0.25, 1.15, N_add)
    real_kewl = [(x, kewl_line_alt(x)) for x in x_kewl]
    random_kewl = gen_bounds(real_kewl, kewl_line_alt, num_random_points=N_add)
    agn_add_x, agn_add_y = zip(*sorted(random_kewl, key=lambda x: x[0]))

    #N_add = 30
    #sfg_add_x = np.linspace(-2.5, -1, N_add)
    #sfg_add_y = kauf_line_alt(sfg_add_x)

    #agn_add_x = np.linspace(-0.5, 0.85, N_add)
    #agn_add_y = kewl_line_alt(agn_add_x)


    # Example logic for computing total_points
    def compute_total_points(hist, min_points=1, max_points=50):
        """
        Compute the total points required to satisfy the condition that
        every bin has at least `min_points` and the densest bin has `max_points`.

        hist: 2D histogram array
        min_points: minimum points per bin
        max_points: maximum points for the densest bin
        """
        # Find non-empty bins
        non_empty_bins = hist > 0
        num_non_empty_bins = np.sum(non_empty_bins)

        if num_non_empty_bins == 0:
            return 0  # no points required if there are no non-empty bins

        # Find min and max densities of non-empty bins
        min_density = hist[non_empty_bins].min()
        max_density = hist[non_empty_bins].max()

        # Scale densities from min_points to max_points
        scaled_density = np.zeros_like(hist, dtype=float)
        if max_density > min_density:
            scaled_density[non_empty_bins] = (
                (hist[non_empty_bins] - min_density) / (max_density - min_density) * (max_points - min_points) + min_points
            )
        else:
            # If all bins have the same density, assign max_points to all
            scaled_density[non_empty_bins] = max_points

        # Compute total points
        total_points = int(np.ceil(np.sum(scaled_density[non_empty_bins])))
        return total_points
    
    # generate points with scatter based on a 2D histogram
    def gen_points_2d(x, y, bins, total_points, scatter_range=0.01):
        """
        Function to generate points (with scatter) based on 2d histogram.
        x: x-data
        y: y-data
        bins: number of bins in the 2d histogram
        total_points: number of generated points
        scatter_range: how much scatter we want (in dex)
        """
        # compute 2d histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # find non-empty bins
        non_empty_bins = hist > 0
        bin_indices = np.argwhere(non_empty_bins)

        # normalize densities to range from 1 to N for non-empty bins
        # least populated bins will have 1 generated data point
        # most populated bin will have N generated data points
        N = 50
        scaled_density = np.zeros_like(hist, dtype=float)
        if non_empty_bins.any():
            min_density = hist[non_empty_bins].min()
            max_density = hist[non_empty_bins].max()
            if max_density > min_density:  # avoid divide by zero
                scaled_density[non_empty_bins] = (
                    (hist[non_empty_bins] - min_density) / (max_density - min_density) * (N - 1) + 1
                )

        # scale the total number of points to match `total_points`
        density_factor = total_points / np.sum(scaled_density[non_empty_bins])
        scaled_density[non_empty_bins] *= density_factor

        # generate points based on density
        generated_x = []
        generated_y = []
        for bin_idx in bin_indices:
            i, j = bin_idx
            count = int(np.floor(scaled_density[i, j]))
            if count > 0:  # only generate points for non-empty bins!!!
                x_min, x_max = xedges[i], xedges[i + 1]
                y_min, y_max = yedges[j], yedges[j + 1]
                gen_x = np.random.uniform(x_min, x_max, count)
                gen_y = np.random.uniform(y_min, y_max, count)

                # add some scatter
                gen_x += np.random.uniform(-scatter_range, scatter_range, count)
                gen_y += np.random.uniform(-scatter_range, scatter_range, count)

                generated_x.extend(gen_x)
                generated_y.extend(gen_y)

        # double check the total number of points matches exactly
        generated_x = np.array(generated_x)
        generated_y = np.array(generated_y)
        if len(generated_x) > total_points:
            # Trim excess points
            indices = np.random.choice(len(generated_x), total_points, replace=False)
            generated_x = generated_x[indices]
            generated_y = generated_y[indices]
        elif len(generated_x) < total_points:
            # pad with additional random points in non-empty bins
            missing_points = total_points - len(generated_x)
            pad_x = []
            pad_y = []
            while len(pad_x) < missing_points:
                i, j = bin_indices[np.random.choice(len(bin_indices))]
                x_min, x_max = xedges[i], xedges[i + 1]
                y_min, y_max = yedges[j], yedges[j + 1]
                gen_x = np.random.uniform(x_min, x_max, 1)
                gen_y = np.random.uniform(y_min, y_max, 1)
                gen_x += np.random.uniform(-scatter_range, scatter_range, 1)
                gen_y += np.random.uniform(-scatter_range, scatter_range, 1)
                pad_x.extend(gen_x)
                pad_y.extend(gen_y)
            generated_x = np.concatenate([generated_x, pad_x[:missing_points]])
            generated_y = np.concatenate([generated_y, pad_y[:missing_points]])

        return generated_x, generated_y


    bins = 100
    # Compute histogram
    hist, xedges, yedges = np.histogram2d(dfsf['x'], dfsf['y'], bins=bins)
    # Compute number of points so each bin has at least 1 point and the densest bin has ~50 points
    total_points = compute_total_points(hist)
    N = total_points + len(sfg_add_x)
    
    # sfg data
    scatter_range = 0.01
    sfx_a, sfy_a = gen_points_2d(dfsf['x'], dfsf['y'], bins, total_points=total_points, scatter_range=scatter_range)
    # reshuffle indices
    ind_sf = np.arange(len(sfx_a))
    np.random.shuffle(ind_sf)
    sfx_a = sfx_a[ind_sf]
    sfy_a = sfy_a[ind_sf]
    # add outside points
    sfx = np.append(sfx_a, sfg_add_x)
    sfy = np.append(sfy_a, sfg_add_y)

    # agn data
    agnx_a, agny_a = gen_points_2d(dfagn['x'], dfagn['y'], bins, total_points=total_points, scatter_range=scatter_range)
    # reshuffle indices
    ind_agn = np.arange(len(agnx_a))
    np.random.shuffle(ind_agn)
    agnx_a = agnx_a[ind_agn]
    agny_a = agny_a[ind_agn]
    # add outside points
    agnx = np.append(agnx_a, agn_add_x)
    agny = np.append(agny_a, agn_add_y)

    # validate matching lengths
    print(f"Length of SF Points: {len(sfx)}, Length of AGN points: {len(agnx)}")
    print('Data have been generated')
    
    #plt.figure(figsize=(8, 6))
    plt.scatter(sfx, sfy, color='c', s=1)
    plt.scatter(agnx, agny, color='gray', s=1)
    #plt.hist2d(dfsf['x'], dfsf['y'], bins=bins, cmap='Blues', alpha=0.5)
    plt.plot(kewl_line()[0], kewl_line()[1], c='k')
    plt.plot(kauf_line()[0], kauf_line()[1], c='k')
    plt.plot(law_line()[0], law_line()[1], c='k')
    plt.xlim(-2.5, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel('N2/Ha')
    plt.ylabel('O3/Hb')
    plt.title('Generated Data w/0.01 dex scatter')
    #plt.legend()
    plt.savefig('plots/bpt_generated_data.pdf')
    #plt.show()
    
    # do the calculations
    balmer_dec = 2.86

    # H-alpha caontribution is 1 (doesn't matter bc they are all ratios)
    sfha = np.linspace(1, 1, N)
    # balmer decrement
    sfhb = sfha/balmer_dec
    # SFG N2 contribution based of the SFG points and SFG H-alpha
    sfn2 = sfha*(10**sfx)
    # same idea as sfn2
    sfo3 = sfhb*(10**sfy)

    # fractional AGN contribution
    f = np.linspace(0.01, 0.99, 981)

    # calculating the AGN O3 contribution for a given f-value
    # f = agno3/(sfo3+agno3) ----> ao3 = -f*sfo3/(f-1)
    agno3 = []
    for index, value in enumerate(sfo3):
            agno3.append((-f[:]*value)/(f[:]-1))

    # calculating agn hb, ha, & n2 based on various lines and the agn x & y components
    agnhb = []
    for index, value in enumerate(agno3):
        agnhb.append(value/(10**agny[index]))

    agnha = []
    for i in agnhb:
        agnha.append(i*balmer_dec)

    agnn2 = []
    for index, value in enumerate(agnha):
        agnn2.append(value*(10**agnx[index]))

    # calculating the total n2, ha, o3, hb contribution for a given f value 
    # n2_tot = agnn2+sfgn2, etc.
    n2 = []
    for index, value in enumerate(agnn2):
        n2.append(value+sfn2[index])

    ha = []
    for index, value in enumerate(agnha):
        ha.append(value+sfha[index])

    o3 = []
    for index, value in enumerate(agno3):
        o3.append(value+sfo3[index])

    hb = []
    for index, value in enumerate(agnhb):
        hb.append(value+sfhb[index])

    # translating n2, ha, o3, hb into log x & y space
    x = []
    for index, value in enumerate(n2):
        x.append(np.log10(value/ha[index]))

    y = []
    for index, value in enumerate(o3):
        y.append(np.log10(value/hb[index]))


    # making array with x/y and f values
    xs = []
    ys = []
    for i in range(N):
        xs.append((np.array( x[i][:]), f))                          
        ys.append((np.array( y[i][:]), f))

    #  putting x/y values in df where the index number is the fractional AGN contribution
    dfx = pd.DataFrame({i: xs[i][0] for i in range(N)}, index=f)
    dfy = pd.DataFrame({i: ys[i][0] for i in range(N)}, index=f)
    
    print('calculations completed')
    
    #plt.figure(figsize=(7, 5))
    #plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=1, alpha=0.2,)
    #for i in range(500):
    #    plt.scatter(dfx[i], dfy[i], c=dfx.index.tolist(), edgecolor='none',  s=0.5)
    #plt.colorbar().set_label(label=r'% AGN/100', size=15)
    #plt.clim(0, 1.0)
    #plt.plot(kewl_line()[0], kewl_line()[1], c='k', linestyle='dashed')
    #plt.plot(kauf_line()[0], kauf_line()[1], c='k', linestyle='dashed')
    #plt.plot(law_line()[0], law_line()[1], c='k', linestyle='dashed')
    #plt.xlim(-2.5, 0.65)
    #plt.ylim(-1.25, 1.5)
    #plt.xlabel(r'[NII]/H$\alpha$', fontsize=20)
    #plt.ylabel(r'[OIII]/H$\beta$', fontsize=20)
    #plt.text(-1.5, 0.0, 'SFGs', c='k', fontweight='bold')
    #plt.text(-0.2, -0.75, 'Comp.', c='k', fontweight='bold')
    #plt.text(0.25, -0.25, 'LINERs', c='k', fontweight='bold')
    #plt.text(-0.5, 0.75, 'Seyferts', c='k', fontweight='bold')
    #plt.savefig('plots/tracks_individual.pdf')
    #print('BPT with individual tracks created (~500 tracks).')
    #plt.show()
    
    # Optimized visualization code
    #plt.figure(figsize=(7, 5))
    # Plot the background scatter for 'eboss'
    #plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=1, alpha=0.2)
    # Prepare all points from dfx and dfy for a single scatter plot
    #all_x = np.concatenate(dfx.values)  # Flatten dfx
    #all_y = np.concatenate(dfy.values)  # Flatten dfy
    #colors = np.repeat(dfx.index, dfx.shape[1])  # Repeat indices for coloring
    # Single scatter plot for tracks
    #plt.scatter(all_x, all_y, c=colors, edgecolor='none', s=0.5)
    # Add colorbar
    #cbar = plt.colorbar()
    #cbar.set_label(label=r'% AGN/100', size=15)
    #cbar.ax.tick_params(labelsize=15)
    #plt.clim(0, 1.0)
    # Plot pre-computed lines
    #plt.plot(kewl_line()[0], kewl_line()[1], c='k', linestyle='dashed')
    #plt.plot(kauf_line()[0], kauf_line()[1], c='k', linestyle='dashed')
    #plt.plot(law_line()[0], law_line()[1], c='k', linestyle='dashed')
    # Set axis limits and labels
    #plt.xlim(-2.5, 0.65)
    #plt.ylim(-1.25, 1.5)
    #plt.xlabel(r'[NII]/H$\alpha$', fontsize=20)
    #plt.ylabel(r'[OIII]/H$\beta$', fontsize=20)
    # Annotate regions
    #plt.text(-1.5, 0.0, 'SFGs', c='k', fontweight='bold')
    #plt.text(-0.2, -0.75, 'Comp.', c='k', fontweight='bold')
    #plt.text(0.25, -0.25, 'LINERs', c='k', fontweight='bold')
    #plt.text(-0.5, 0.75, 'Seyferts', c='k', fontweight='bold')
    #plt.savefig('plots/tracks_clean.pdf')
    #print('BPT with all tracks created')
    #plt.show()
    
    def f_agn_nearest(dfx, dfy, x, y, num):
        distances = np.sqrt((dfx - x)**2 + (dfy - y)**2)
        # get nearest N points
        min_vals = distances.stack().nsmallest(num)
        min_inds = min_vals.index
        result = [index[0] for index in min_inds]
        return result

    def f_agn_radial(dfx, dfy, x, y, radius):
        distances = np.sqrt((dfx - x)**2 + (dfy - y)**2)
        mask = distances < radius
        result = distances.index[mask.any(axis=1)].tolist()
        if not result:
            return np.nan
        else:
            return result

    # Combined method: Use radius for points in model space, use nearest N for points outside
    def f_agn_combined(dfx, dfy, x, y, radius, num=20):
        distances = np.sqrt((dfx - x)**2 + (dfy - y)**2)
        # Select points within the radius
        mask = distances < radius
        result = distances.index[mask.any(axis=1)].tolist()
        # If no points are within the radius, return the nearest N points
        if not result:
            min_vals = distances.stack().nsmallest(num)
            min_inds = min_vals.index
            result = [index[0] for index in min_inds]
        return result

    # Combined method: Use radius for points in model space, use nearest N for points outside
    #def f_agn_combined(dfx, dfy, x, y, radius, num=20):
    #    dfx_array = dfx.values
    #    dfy_array = dfy.values
    
    #    # Calculate distances squared (to avoid computing square root unnecessarily for comparison)
    #    dx = dfx_array - x
    #    dy = dfy_array - y
    #    distances_squared = dx**2 + dy**2
    
    #    # Select points within the radius (using squared distances)
    #    mask = distances_squared < radius**2
    #    result = np.where(mask.any(axis=1))[0].tolist()  # Get indices of points within the radius

    #    # If no points are within the radius, return the nearest N points
    #    if not result:
    #        # Flatten distances and get the indices of the smallest distances
    #        min_indices = np.argsort(distances_squared, axis=None)[:num]
    #        result = [divmod(idx, distances_squared.shape[1])[0] for idx in min_indices]
    
    #    return result
    
    # Generate x and y values to make the lookup table
    # small table
    #x_values = np.arange(-2.25, 0.75, 0.1)
    #y_values = np.arange(-1.25, 1.35, 0.1)
    #large table
    x_values = np.arange(-2.25, 0.75, 0.025)
    y_values = np.arange(-1.25, 1.35, 0.025)
    
    # Create a meshgrid
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)

    # Flatten the meshgrid to get x and y values
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()

    # Create the DataFrame
    dfgrid = pd.DataFrame({'x': x_flat, 'y': y_flat})
    
    
    # takes ~80 minutes for 10,000 tracks & 3120 galaxies
    # Define a helper function for processing
    #def calculate_fraction(radius):
    #    start_time = time.time()
    #    result = dfgrid.apply(lambda row: f_agn_combined(dfx, dfy, row['x'], row['y'], radius=radius), axis=1)
    #    end_time = time.time()
    #    tot_time = end_time - start_time
    #    print(f'For Combined Method (r={radius}) {len(dfgrid)} galaxies & {len(dfx)} lines it takes {tot_time} seconds')
    #    return result

    
    #print('Starting lookup table creation... ')
    
    
    #start_time = time.time()
    #total_steps = len(dfgrid)
    #frac_agn_r025 = dfgrid.apply(
    #    lambda row: (print(f"Progress: {row.name / total_steps * 100:.2f}% completed", end="\r"),
    #                 f_agn_combined(dfx, dfy, row['x'], row['y'], radius=0.025))[1], axis=1)
    #end_time = time.time()
    #tot_time = end_time - start_time
    #print('\nFor Radial Method (r=0.025)', len(dfgrid), 'galaxies &', len(dfx.columns), 'tracks it takes', tot_time, 'seconds')

    # Define a function for parallel processing, n_jobs is the number of CPUs the function will use... BE CAREFUL
    def calculate_fraction(radius, n_jobs=10):
        start_time = time.time()
        
        # Parallel processing using Modified f_agn_combined
        results = Parallel(n_jobs=n_jobs)(
            delayed(f_agn_combined)(dfx, dfy, row['x'], row['y'], radius) for _, row in dfgrid.iterrows()
        )

        end_time = time.time()
        print(f'For Combined Method (r={radius}) {len(dfgrid)} galaxies & {len(dfx)} lines it takes {end_time - start_time:.2f} seconds')
        return results

    n_cpus = 10
    print(f'Starting lookup table creation... You are using {n_cpus} CPUs!')
    start_time = time.time()
    frac_agn_r025 = calculate_fraction(radius=0.025, n_jobs=n_cpus)
    
    dfgrid_r025 = dfgrid.copy()
    dfgrid_r025['frac_agn'] = frac_agn_r025
    dfgrid_r025['f_mean'] = dfgrid_r025['frac_agn'].apply(np.nanmean)
    dfgrid_r025['f_median'] = dfgrid_r025['frac_agn'].apply(np.median)
    dfgrid_r025['f_std'] = dfgrid_r025['frac_agn'].apply(np.std)
    dfgrid_r025['f_var'] = dfgrid_r025['frac_agn'].apply(np.var)

    end_time = time.time()
    tot_time = end_time - start_time
    print('\nTotal time taken:', tot_time, 'seconds')
    
    plt.figure(figsize=(7, 5))
    plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=0.1, alpha=0.4)
    plt.scatter(dfgrid_r025['x'], dfgrid_r025['y'], c=dfgrid_r025['f_mean'], cmap='cool', marker='o', s=10)
    plt.colorbar(label=r'F$_{AGN}$')
    plt.clim(0, 1)
    plt.plot(kewl_line()[0], kewl_line()[1], c='r', linestyle='dashed')
    plt.plot(kauf_line()[0], kauf_line()[1], c='r', linestyle='dashed')
    plt.plot(law_line()[0], law_line()[1], c='r', linestyle='dashed')
    plt.xlim(-2, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel(r'[NII]/H$\alpha$')
    plt.ylabel(r'[OIII]/H$\beta$')
    plt.title(f'Distribution for r025')
    plt.savefig('plots/grid_sampling_r025.pdf')
    #plt.show()   

    dfgrid_r025.to_csv('data/table_r025.csv', index=None)

    del dfgrid_r025, frac_agn_r025
    gc.collect()
    print('Trash has been taken out for r025')

    #sys.exit("Test finished")
    #########################

    #start_time = time.time()
    #total_steps = len(dfgrid)
    #frac_agn_r05 = dfgrid.apply(
    #    lambda row: (print(f"Progress: {row.name / total_steps * 100:.2f}% completed", end="\r"),
    #                 f_agn_combined(dfx, dfy, row['x'], row['y'], radius=0.05))[1], axis=1)
    #end_time = time.time()
    #tot_time = end_time - start_time
    #print('\nFor Radial Method (r=0.05)', len(dfgrid), 'galaxies &', len(dfx.columns), 'tracks it takes', tot_time, 'seconds')

    #dfgrid_r05 = dfgrid.copy()
    #dfgrid_r05['frac_agn'] = frac_agn_r05
    #dfgrid_r05['f_mean'] = dfgrid_r05['frac_agn'].apply(np.nanmean)
    #dfgrid_r05['f_median'] = dfgrid_r05['frac_agn'].apply(np.median)
    #dfgrid_r05['f_std'] = dfgrid_r05['frac_agn'].apply(np.std)
    #dfgrid_r05['f_var'] = dfgrid_r05['frac_agn'].apply(np.var)

    start_time = time.time()
    frac_agn_r05 = calculate_fraction(radius=0.05, n_jobs=n_cpus)

    dfgrid_r05 = dfgrid.copy()
    dfgrid_r05['frac_agn'] = frac_agn_r05
    dfgrid_r05['f_mean'] = dfgrid_r05['frac_agn'].apply(np.nanmean)
    dfgrid_r05['f_median'] = dfgrid_r05['frac_agn'].apply(np.median)
    dfgrid_r05['f_std'] = dfgrid_r05['frac_agn'].apply(np.std)
    dfgrid_r05['f_var'] = dfgrid_r05['frac_agn'].apply(np.var)

    end_time = time.time()
    tot_time = end_time - start_time
    print('\nTotal time taken:', tot_time, 'seconds')
    
    plt.figure(figsize=(7, 5))
    plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=0.1, alpha=0.4)
    plt.scatter(dfgrid_r05['x'], dfgrid_r05['y'], c=dfgrid_r05['f_mean'], cmap='cool', marker='o', s=10)
    plt.colorbar(label=r'F$_{AGN}$')
    plt.clim(0, 1)
    plt.plot(kewl_line()[0], kewl_line()[1], c='r', linestyle='dashed')
    plt.plot(kauf_line()[0], kauf_line()[1], c='r', linestyle='dashed')
    plt.plot(law_line()[0], law_line()[1], c='r', linestyle='dashed')
    plt.xlim(-2, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel(r'[NII]/H$\alpha$')
    plt.ylabel(r'[OIII]/H$\beta$')
    plt.title(f'Distribution for r05')
    plt.savefig('plots/grid_sampling_r05.pdf')
    #plt.show()   

    dfgrid_r05.to_csv('data/table_r05.csv', index=None)

    del dfgrid_r05, frac_agn_r05
    gc.collect()
    print('Trash has been taken out for r05')

    #####################################

    #start_time = time.time()
    #total_steps = len(dfgrid)
    #frac_agn_r075 = dfgrid.apply(
    #    lambda row: (print(f"Progress: {row.name / total_steps * 100:.2f}% completed", end="\r"),
    #                 f_agn_combined(dfx, dfy, row['x'], row['y'], radius=0.075))[1], axis=1)
    #end_time = time.time()
    #tot_time = end_time - start_time
    #print('\nFor Radial Method (r=0.075)', len(dfgrid), 'galaxies &', len(dfx.columns), 'tracks it takes', tot_time, 'seconds')

    #dfgrid_r075 = dfgrid.copy()
    #dfgrid_r075['frac_agn'] = frac_agn_r075
    #dfgrid_r075['f_mean'] = dfgrid_r075['frac_agn'].apply(np.nanmean)
    #dfgrid_r075['f_median'] = dfgrid_r075['frac_agn'].apply(np.median)
    #dfgrid_r075['f_std'] = dfgrid_r075['frac_agn'].apply(np.std)
    #dfgrid_r075['f_var'] = dfgrid_r075['frac_agn'].apply(np.var)

    start_time = time.time()
    frac_agn_r075 = calculate_fraction(radius=0.075, n_jobs=n_cpus)

    dfgrid_r075 = dfgrid.copy()
    dfgrid_r075['frac_agn'] = frac_agn_r075
    dfgrid_r075['f_mean'] = dfgrid_r075['frac_agn'].apply(np.nanmean)
    dfgrid_r075['f_median'] = dfgrid_r075['frac_agn'].apply(np.median)
    dfgrid_r075['f_std'] = dfgrid_r075['frac_agn'].apply(np.std)
    dfgrid_r075['f_var'] = dfgrid_r075['frac_agn'].apply(np.var)

    end_time = time.time()
    tot_time = end_time - start_time
    print('\nTotal time taken:', tot_time, 'seconds')
    
    plt.figure(figsize=(7, 5))
    plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=0.1, alpha=0.4)
    plt.scatter(dfgrid_r075['x'], dfgrid_r075['y'], c=dfgrid_r075['f_mean'], cmap='cool', marker='o', s=10)
    plt.colorbar(label=r'F$_{AGN}$')
    plt.clim(0, 1)
    plt.plot(kewl_line()[0], kewl_line()[1], c='r', linestyle='dashed')
    plt.plot(kauf_line()[0], kauf_line()[1], c='r', linestyle='dashed')
    plt.plot(law_line()[0], law_line()[1], c='r', linestyle='dashed')
    plt.xlim(-2, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel(r'[NII]/H$\alpha$')
    plt.ylabel(r'[OIII]/H$\beta$')
    plt.title(f'Distribution for r075')
    plt.savefig('plots/grid_sampling_r075.pdf')
    #plt.show()   

    dfgrid_r075.to_csv('data/table_r075.csv', index=None)

    del dfgrid_r075, frac_agn_r075
    gc.collect()
    print('Trash has been taken out for r075')

    #########################################

    #start_time = time.time()
    #total_steps = len(dfgrid)
    #frac_agn_r1 = dfgrid.apply(
    #    lambda row: (print(f"Progress: {row.name / total_steps * 100:.2f}% completed", end="\r"),
    #                 f_agn_combined(dfx, dfy, row['x'], row['y'], radius=0.1))[1], axis=1)
    #end_time = time.time()
    #tot_time = end_time - start_time
    #print('\nFor Radial Method (r=0.1)', len(dfgrid), 'galaxies &', len(dfx.columns), 'tracks it takes', tot_time, 'seconds')

    #print('Lookup tables created.')

    #dfgrid_r1 = dfgrid.copy()
    #dfgrid_r1['frac_agn'] = frac_agn_r1
    #dfgrid_r1['f_mean'] = dfgrid_r1['frac_agn'].apply(np.nanmean)
    #dfgrid_r1['f_median'] = dfgrid_r1['frac_agn'].apply(np.median)
    #dfgrid_r1['f_std'] = dfgrid_r1['frac_agn'].apply(np.std)
    #dfgrid_r1['f_var'] = dfgrid_r1['frac_agn'].apply(np.var)

    start_time = time.time()
    frac_agn_r1 = calculate_fraction(radius=0.1, n_jobs=n_cpus)

    dfgrid_r1 = dfgrid.copy()
    dfgrid_r1['frac_agn'] = frac_agn_r1
    dfgrid_r1['f_mean'] = dfgrid_r1['frac_agn'].apply(np.nanmean)
    dfgrid_r1['f_median'] = dfgrid_r1['frac_agn'].apply(np.median)
    dfgrid_r1['f_std'] = dfgrid_r1['frac_agn'].apply(np.std)
    dfgrid_r1['f_var'] = dfgrid_r1['frac_agn'].apply(np.var)

    end_time = time.time()
    tot_time = end_time - start_time
    print('\nTotal time taken:', tot_time, 'seconds')

    print('Lookup tables created!')
    
    plt.figure(figsize=(7, 5))
    plt.scatter(eboss['x'], eboss['y'], c='tab:gray', s=0.1, alpha=0.4)
    plt.scatter(dfgrid_r1['x'], dfgrid_r1['y'], c=dfgrid_r1['f_mean'], cmap='cool', marker='o', s=10)
    plt.colorbar(label=r'F$_{AGN}$')
    plt.clim(0, 1)
    plt.plot(kewl_line()[0], kewl_line()[1], c='r', linestyle='dashed')
    plt.plot(kauf_line()[0], kauf_line()[1], c='r', linestyle='dashed')
    plt.plot(law_line()[0], law_line()[1], c='r', linestyle='dashed')
    plt.xlim(-2, 0.65)
    plt.ylim(-1.25, 1.25)
    plt.xlabel(r'[NII]/H$\alpha$')
    plt.ylabel(r'[OIII]/H$\beta$')
    plt.title(f'Distribution for r1')
    plt.savefig('plots/grid_sampling_r1.pdf')
    #plt.show()   

    dfgrid_r1.to_csv('data/table_r1.csv', index=None)

    del dfgrid_r1, frac_agn_r1
    gc.collect()
    print('Trash has been taken out for r1')

    ######################################

    print('And we are done, great job team!')
   
# run the damn script
if __name__ == '__main__':
    main()
