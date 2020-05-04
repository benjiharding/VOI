import glob
import json
import logging
import multiprocessing as mp
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygeostat as gs
from numpy.random import RandomState
from scipy.spatial import Delaunay

#
# FUNCTIONS
#


def adjxy(x, y, xnew, theta):
    '''Calculate new (x,y) position with xnew
    and rotation theta in radians'''

    dx = x - xnew
    dy = dx * np.tan(theta)
    yadj = y + dy

    return xnew, yadj


def rotate_around_point(x, y, radians, origin=(0, 0)):
    '''Lyle Scott, III  // lyle@ls3.io'''

    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def in_hull(p, hull):

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull, qhull_options='QJ')

    return hull.find_simplex(p) >= 0


def calc_grid_xyz(grid):
    '''Return xyz coords of grid nodes
    grid format: [nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize]'''

    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid

    grid_idx = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                grid_idx.append([i, j, k])

    grid_xyz = []
    for i in range(len(grid_idx)):
        ix = grid_idx[i][0]
        iy = grid_idx[i][1]
        iz = grid_idx[i][2]
        gx = ix * xsize + xmin
        gy = iy * ysize + ymin
        gz = iz * zsize + zmin
        grid_xyz.append([gx, gy, gz])
    grid_xyz = np.array(grid_xyz)

    return grid_xyz


def fval(zsim, hull, grid_xyz, xsiz, ysiz, zsiz, SG, price, cf,
         cost_m, cost_p, cost_ga, frec, fmine, fdil, froy, fstr):
    '''Return expected value of 'hull' across all realizations in zsim'''

    in_stope = in_hull(grid_xyz, hull) * 1

    # get fixed costs based on stope geometry
    blk_vol = xsiz * ysiz * zsiz
    mined = np.sum(in_stope)
    if mined == 0:
        print('No blocks inside stope hull!')
    dil = mined * fdil
    tonnes = (mined + dil) * blk_vol * SG
    cost = tonnes * (cost_m + cost_p + cost_ga)

    # calculate value of blocks in the stope across nreals
    mined_gr = zsim[in_stope == 1]
    diluted = (mined_gr * blk_vol) / (blk_vol + fdil*blk_vol)
    rec_metal = diluted * blk_vol * SG * frec * fmine
    gross = np.nansum(rec_metal, axis=0) / cf * price * froy * fstr
    net = gross - cost
    exp_val = np.nanmean(net)

    return exp_val, np.nanmean(diluted), tonnes


def sample_ddh3d(simfl, rockfl, outfl1, outfl2, nreals, nddhout,
                 grid, dh_data, sample_space):
    '''Function to sample 3D simulated realizations by defining drillhole
    collars, azimuths and dips. Drillhole fan option will drill multiple fanned
    holes from a single collar location simulating UG drilling. Samples are
    returned as the x,y,z midpoint of the defined sample interval. If rock type
    file is defined, function returns two data frames - one with grades and the
    second with rock types. Otherwise dataframe of grades is returned. Positive dip
    is down, ie vertical holes have a dip of 90.
    dhdata format: [x,y,z,azm,dip,length]
    Code Author: Ben Harding
    Date: February 6 2020'''

    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid

    grade = gs.DataFile(simfl).data.values
    grade = grade.flatten()
    if rockfl is not None:
        rock = gs.DataFile(rockfl).data.values
        rock = rock.flatten()

    dhdata = np.loadtxt(dh_data)

    # setup ddh properties
    col = dhdata[:, 0:3]
    azm = dhdata[:, 3]
    dip = dhdata[:, 4]
    length = dhdata[:, 5]

    # get sample coordinates

    nddh = col.shape[0]
    holeid = np.arange(1, nddh+1)
    coords = []

    print('Generating sample coordinates')
    print('\n')

    for ddh in zip(col, azm, dip, length, holeid):
        col_tmp = ddh[0]
        azm_tmp = ddh[1]
        dip_tmp = ddh[2]
        len_tmp = ddh[3]
        holeid_tmp = ddh[4]
        azm_rad = azm_tmp * np.pi/180
        dip_rad = dip_tmp * np.pi/180
        samp_tmp = np.arange(sample_space/2, len_tmp, sample_space)
        coords_tmp = np.zeros((len(samp_tmp), 5))

        for i, s in enumerate(samp_tmp):
            l_plan = np.cos(dip_rad) * s
            dx = np.sin(azm_rad) * l_plan
            dy = np.cos(azm_rad) * l_plan
            dz = np.sin(dip_rad) * s
            coords_tmp[i, 0] = holeid_tmp
            coords_tmp[i, 1] = col_tmp[0] + dx
            coords_tmp[i, 2] = col_tmp[1] + dy
            coords_tmp[i, 3] = col_tmp[2] - dz
            coords_tmp[i, 4] = s

        coords.append(coords_tmp)
    coords = np.vstack(coords)

    # get 3D model index of dh coordinates
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    ix = np.ceil((x - xmin) / xsize + 0.5)
    iy = np.ceil((y - ymin) / ysize + 0.5)
    iz = np.ceil((z - zmin) / zsize + 0.5)

    # Check if on the minimum edge
    ix = np.where((ix == 0) & (x == xmin - (xsize / 2.0)), 1, ix)
    iy = np.where((iy == 0) & (y == ymin - (ysize / 2.0)), 1, iy)
    iz = np.where((iz == 0) & (z == zmin - (zsize / 2.0)), 1, iz)
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1

    # get the 1D model index from 3D model index
    idx = ix + iy * nx + iz * nx * ny
    idx = idx.astype(int)

    # set coords outside model grid to NaN
    len_real = nx*ny*nz
    idx = np.where((idx < 0) | (idx > len_real), np.nan, idx)
    out_grid = np.argwhere(np.isnan(idx) != False)  # indexes outside grid
    idx = np.delete(idx, out_grid).astype(int)  # remove indexes
    coords = np.delete(coords, out_grid, axis=0)  # remove coords

    print(f'{len(out_grid)} samples outside the grid')
    print('\n')

    # sample the gridded model data
    samp_gr = np.zeros((len(coords), nreals))
    samp_rk = np.zeros((len(coords), nreals))
    for i in range(nreals):
        ii = i*len_real
        jj = (i+1)*len_real
        samp_gr[:, i] = grade[ii:jj][idx]
        if rockfl is not None:
            samp_rk[:, i] = rock[ii:jj][idx]

    out_gr = np.concatenate((coords, samp_gr), axis=1)
    out_rk = np.concatenate((coords, samp_rk), axis=1)

    # output dataframe
    cols = [f'Real {i+1}' for i in range(nreals)]
    cols[0:0] = ['DHID', 'X', 'Y', 'Z', 'Mid Point']
    samples_gr = pd.DataFrame(out_gr, columns=cols)
    samples_rk = pd.DataFrame(out_rk, columns=cols)
    nsamp = len(samples_gr)

    print(f'{nsamp} samples generated from {nddh} holes')
    print('\n')

    ndata = np.array([nsamp, nddh])
    np.savetxt(nddhout, ndata, fmt='%i')

    # output gslib data files
    gr_gslib = gs.DataFile(data=samples_gr)
    rk_gslib = gs.DataFile(data=samples_rk)
    gs.write_gslib_f(gr_gslib, outfl1)

    if rockfl is not None:
        gs.write_gslib_f(rk_gslib, outfl2)

    return [samples_gr, samples_rk] if rockfl is not None else samples_gr


def pcorr(true, reals, nreals, thresh):
    '''Calculate probability of correct classification for a set
        of realizations against the truth for a given threshold'''

    true = true.flatten()
    pi = np.zeros(true.shape)
    pii = np.zeros(true.shape)
    pcorr = np.zeros(true.shape)

    # code true model based on threshold
    true = np.where(true >= thresh, 1, 0)
    reals = np.where(reals >= thresh, 1, 0)

    for i in range(nreals):
        sim = reals[:, i]
        pi += np.where((sim == 1) & (true == 0), 1, 0)
        pii += np.where((sim == 0) & (true == 1), 1, 0)

    pi /= nreals
    pii /= nreals
    pcorr = 1-(pi+pii)

    return pcorr


def tag_ddh(dhdata, rkdata, nreals, outfl, grid):
    '''Code drillholes by categories in rkdata'''

    ddh = gs.DataFile(dhdata)
    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid

    ix = np.ceil((ddh['X'].values - xmin) / xsize + 0.5)
    iy = np.ceil((ddh['Y'].values - ymin) / ysize + 0.5)
    iz = np.ceil((ddh['Z'].values - zmin) / zsize + 0.5)
    ix = np.where((ix == 0) & (ddh['X'].values == xmin - (xsize / 2.0)), 1, ix)
    iy = np.where((iy == 0) & (ddh['Y'].values == ymin - (ysize / 2.0)), 1, iy)
    iz = np.where((iz == 0) & (ddh['Z'].values == zmin - (zsize / 2.0)), 1, iz)
    ix, iy, iz = ix - 1, iy - 1, iz - 1

    idx = ix + iy * nx + iz * nx * ny
    idx = idx.astype(int)

    len_real = nx*ny*nz
    idx = np.where((idx < 0) | (idx > len_real), np.nan, idx)
    out_grid = np.argwhere(np.isnan(idx) != False)  # indexes outside grid
    idx = np.delete(idx, out_grid).astype(int)  # remove indexes

    for i in range(nreals):  # nreals
        mwa = gs.DataFile(rkdata[i])
        tag = mwa.data.iloc[idx].values
        ddh['GSTag'] = tag
        gs.write_gslib_f(ddh, outfl[i])


def add_coord(grid, nreals, simfl, outfl):
    '''Add xyz coordinates to realizations'''

    grid_xyz = calc_grid_xyz(grid)

    for i in range(nreals):
        sim_xyz = gs.DataFile(simfl[i])
        sim_xyz.data['X'] = grid_xyz[:, 0]
        sim_xyz.data['Y'] = grid_xyz[:, 1]
        sim_xyz.data['Z'] = grid_xyz[:, 2]
        sim_xyz.data = sim_xyz.data[['X', 'Y', 'Z', 'variable_00001']]
        sim_xyz.writefile(outfl[i])


def mwa_threshold(low, high, nreals, simfl, outfl):
    '''Threshold MWA realizations to categories'''

    for i in range(nreals):
        mwa = gs.DataFile(simfl[i])
        mwa.data.loc[mwa.data['variable_00001'] < low, 'cat'] = 0
        mwa.data.loc[(mwa.data['variable_00001'] >= low) & (
            mwa.data['variable_00001'] < high), 'cat'] = 1
        mwa.data.loc[mwa.data['variable_00001'] > high, 'cat'] = 2
        mwa.writefile(outfl[i], variables='cat', fmt='%i')


def clip_reals(nreals, grid, bxmin, bxmax, bymin, bymax, bzmin, bzmax,
               simfl, outfl):
    '''Clip realizations by a 3D bounding box'''

    # xyz locations of simulation grid
    grid_xyz = calc_grid_xyz(grid)

    # generate keyout array
    buf_box = np.array([(bxmin, bymin, bzmin),
                        (bxmin, bymax, bzmin),
                        (bxmax, bymax, bzmin),
                        (bxmax, bymin, bzmin),
                        (bxmin, bymin, bzmax),
                        (bxmin, bymax, bzmax),
                        (bxmax, bymax, bzmax),
                        (bxmax, bymin, bzmax)])

    buff = in_hull(grid_xyz, buf_box) * 1
    grid_clip_xyz = grid_xyz[buff == 1]
    np.savetxt('grid_clip_xyz.out', grid_clip_xyz, fmt='%.2f')

    # load realizations
    sim = np.zeros((len(grid_clip_xyz), nreals))

    for i in range(nreals):

        real = gs.DataFile(simfl[i])
        val = real.data.values.ravel()
        val = val[buff == 1]
        sim[:, i] = val

    # all realizations in one array
    out_flat = sim.flatten(order='F')
    outdat = gs.DataFile(data=pd.DataFrame(out_flat,
                                           columns=['sim']))
    outdat.writefile(outfl)


def fobj(x, nreals, nx, ny, nz, xsiz, ysiz, zsiz, xmin, ymin, zmin,
         SG, price, cf, cost_m, cost_p, cost_ga, frec, fmine, fdil,
         froy, fstr, min_width, max_width, grid_xyz, rot, zsim, verts):

    # get rotation in radians
    theta = rot * np.pi/180

    # get dx, dy between points and vector from population
    xlocs = verts[:, 0]
    ylocs = verts[:, 1]
    xadj, yadj = adjxy(xlocs, ylocs, x, theta)

    # update verticies with points from population
    verts[:, 0] = xadj
    verts[:, 1] = yadj

    # check geometric contraints of the stope
    # vhw = [v1,v2,v5,v6]
    # vfw = [v3,v4,v7,v8]
    # bot = [v1,v2,v3,v4]
    # top = [v5,v6,v7,v8]
    # shw = [v1,v2,v5,v6]
    # sfw = [v4,v3,v8,v7]

    vhw = [0, 1, 4, 5]
    vfw = [2, 3, 6, 7]

    bot = [0, 1, 2, 3]
    top = [4, 5, 6, 7]

    shw = [0, 1, 4, 5]
    sfw = [3, 2, 7, 6]

    # make sure hw and fw verticies are within the grid
    xmax = xmin + nx * xsiz

    for i in zip(vhw, vfw):
        # i[0] = hw, i[1] = fw
        ihw = i[0]
        ifw = i[1]

        # if vhw < xmin:
        if verts[ihw][0] < xmin:
            verts[ihw][0], verts[ihw][1] = adjxy(
                verts[ihw][0], verts[ihw][1], xmin, theta)

        # if vfw < xmin:
        if verts[ifw][0] < xmin:
            verts[ifw][0], verts[ifw][1] = adjxy(
                verts[ifw][0], verts[ifw][1], xmin+min_width, theta)

        # if vfw > xmax:
        if verts[ifw][0] > xmax:
            verts[ifw][0], verts[ifw][1] = adjxy(
                verts[ifw][0], verts[ifw][1], xmax, theta)

        # if vhw > xmax:
        if verts[ihw][0] > xmax:
            verts[ihw][0], verts[ihw][1] = adjxy(
                verts[ihw][0], verts[ihw][1], xmax-min_width, theta)

    # make sure hw and fw dont overlap
    for j in zip(vhw, vfw):
        # j[0] = hw, j[1] = fw
        ihw = j[0]
        ifw = j[1]

        if verts[ifw][0] < verts[ihw][0]:
            verts[ifw][0], verts[ifw][1] = adjxy(verts[ifw][0], verts[ifw][1],
                                                 verts[ihw][0]+min_width, theta)

    # force stope to be vertical or dip west (this is not a robust solution...)
    for k in zip(bot, top):
        # k[0] = bot, k[1] = top
        ib = k[0]
        it = k[1]

        if verts[it][0] < verts[ib][0]:
            verts[it][0], verts[it][1] = adjxy(
                verts[it][0], verts[it][1], verts[ib][0], theta)

    # get distances for maximum span constraint
    stope_len = 20
    max_span = np.sqrt(stope_len**2 + max_width**2)
    span = []
    for l in zip(shw, sfw):
        ihw = l[0]
        ifw = l[1]
        span_dist = np.sqrt((verts[ihw][0]-verts[ifw][0])
                            ** 2 + (verts[ihw][1]-verts[ifw][1])**2)
        span.append(span_dist)

    # if max span is violated adjust each vertex 1/2 req. dist to satisfy
    for s in zip(shw, sfw, span):
        ihw = s[0]
        ifw = s[1]
        spn = s[2]

        # enforce max. span width
        if spn > max_span:
            delx = np.sqrt(spn**2 - stope_len**2)
            max_dx = max_width
            dx = (delx-max_dx)/2 * np.cos(theta)
            dy = (delx-max_dx)/2 * -np.sin(theta)
            verts[ihw][0] += dx
            verts[ihw][1] += dy
            verts[ifw][0] -= dx
            verts[ifw][1] -= dy

    # check again hw and fw dont overlap
    for j in zip(vhw, vfw):
        # j[0] = hw, j[1] = fw
        ihw = j[0]
        ifw = j[1]

        if verts[ifw][0] < verts[ihw][0]:
            verts[ifw][0], verts[ifw][1] = adjxy(verts[ifw][0], verts[ifw][1],
                                                 verts[ihw][0]+min_width, theta)

    # get distances b/w HW and FW verticies for mining width constraint
    width = []
    for v in zip(vhw, vfw):
        ihw = v[0]
        ifw = v[1]
        dist = np.sqrt((verts[ihw][0]-verts[ifw][0]) **
                       2 + (verts[ihw][1]-verts[ifw][1])**2)
        width.append(dist)

    # if width constraints are violated adjust each vertex 1/2 req. dist to satisfy
    for w in zip(vhw, vfw, width):
        ihw = w[0]
        ifw = w[1]
        wdt = w[2]

        # enforce min. mining width
        if wdt < min_width:
            dx = (min_width-wdt)/2 * np.cos(theta)
            dy = (min_width-wdt)/2 * -np.sin(theta)
            verts[ihw][0] -= dx
            verts[ihw][1] -= dy
            verts[ifw][0] += dx
            verts[ifw][1] += dy

        # enforce max. mining width
        if wdt > max_width:
            dx = (wdt-max_width)/2 * np.cos(theta)
            dy = (wdt-max_width)/2 * -np.sin(theta)
            verts[ihw][0] += dx
            verts[ihw][1] += dy
            verts[ifw][0] -= dx
            verts[ifw][1] -= dy

    # check again hw and fw dont overlap
    for j in zip(vhw, vfw):
        # j[0] = hw, j[1] = fw
        ihw = j[0]
        ifw = j[1]

        if verts[ifw][0] < verts[ihw][0]:
            verts[ifw][0], verts[ifw][1] = adjxy(verts[ifw][0], verts[ifw][1],
                                                 verts[ihw][0]+min_width, theta)

    # call value function with current stope verticies
    exp_val, _, _ = fval(zsim, verts, grid_xyz, xsiz, ysiz, zsiz, SG, price, cf,
                         cost_m, cost_p, cost_ga, frec, fmine, fdil, froy, fstr)

    # return negative value for minimization of f(x)
    return exp_val * -1, verts


def de(fobj, args, bounds, seed, mut=(0.5, 1), crossp=0.9,
       popsize=10, its=500, K=25, g=10e-6):
    '''[1] https://gist.github.com/pablormier/0caff10a5f76e87857b44f63757729b0#file-differential_evolution-py
       [2] https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
       [3] https://www1.icsi.berkeley.edu/~storn/code.html#prac
       [4] https://www.sciencedirect.com/science/article/pii/S1110866514000279

       Modified for fobj arguments
       Modified for random seed repeatability
       Modified for dithering:
       mut = (0.5,1) by default as per [3]
       Modified for random restarts [4]:
       K = number of previous generations to check
       g = maximum threshold between |fcurr(x)-fprev(x)|'''

    random_state = np.random.RandomState(seed)
    low = mut[0]
    high = mut[1]
    dimensions = len(bounds)

    # initialize population
    pop = random_state.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind, *args.values())[0] for ind in pop_denorm])
    hulls = np.asarray([fobj(ind, *args.values())[1] for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    best_hull = hulls[best_idx]

    # random restart parameters
    f_its = np.zeros((popsize, its+1))
    df_its = np.zeros((popsize, its+1))
    rr = 0

    for i in range(its):           # G: generations
        for j in range(popsize):   # NP: number in population

            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[random_state.choice(idxs, 3, replace=False)]
            dith = random_state.uniform(low, high)
            mutant = np.clip(a + dith * (b - c), 0, 1)
            cross_points = random_state.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[random_state.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f, hull = fobj(trial_denorm, *args.values())

            if f < fitness[j]:
                fitness[j] = f
                # replace population vector with normalized trial vector
                pop[j] = trial
                # replace population hull with trial stope hull
                hulls[j] = hull
                if f < fitness[best_idx]:
                    best_idx = j             # update best index in population
                    best = trial_denorm      # retain denormalized trial vector
                    best_hull = hull         # retain corresponding stope hull

            # track fitness of population from generation to generation
            f_its[:, i] = fitness
            df_its[:, i] = np.abs(f_its[:, i] - f_its[:, i-1])

        # check random restart criteria but do K iterations first
        if i > K:
            if i % K == 0:
                check = np.where(df_its[:, (i-K):i] < g, 1, 0)
                if np.sum(check) == K*popsize:
                    # reinitialize entire population but the best vector
                    fit = pop[best_idx].copy()
                    pop_rr = random_state.rand(popsize, dimensions)
                    pop_rr[best_idx] = fit
                    pop = pop_rr.copy()
                    rr += 1

        # yield best, best_hull, fitness[best_idx]

    return best, hulls[best_idx], fitness[best_idx], f_its[best_idx][:-1]


#
# GENERATE JOB LIST
#


def gen_directories(dirlist, nreals_true, nphase):
    '''Generate output directories; reference model dir is first'''

    for dirs in dirlist:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    for dirs in dirlist[1:]:
        for t in range(nreals_true):
            for j in range(nphase):
                if not os.path.exists(dirs + f't{t+1}/phase{j+1}'):
                    os.makedirs(dirs + f't{t+1}/phase{j+1}')


def gen_sample_args(true_idx, phase_idx, grid, sample_space,
                    refdir, dhdir):
    '''Generate arguments for sample_ddh3d'''

    sim = refdir + f'real{true_idx}.gsb'
    rock = None
    ddh = f'p{phase_idx}_all.out'
    outfl1 = dhdir + f't{true_idx}/ddh_au_p{phase_idx}_all.out'
    outfl2 = None
    nddhout = f'ndata_p{phase_idx}.out'

    sample_args = tuple((sim, rock, outfl1, outfl2, 1, nddhout,
                         grid, ddh, sample_space))

    return sample_args


def gen_addcoord_args(true_idx, phase_idx, nreals, grid, simdir):
    '''Generate arguments for add_coord'''

    sim_list = []
    out_list = []

    for i in range(nreals):
        simfl = simdir + f't{true_idx}/phase{phase_idx}/real{i+1}.gsb'
        outfl = simdir + f't{true_idx}/phase{phase_idx}/real{i+1}_xyz.gsb'
        sim_list.append(simfl)
        out_list.append(outfl)

    coord_args = tuple((grid, nreals, sim_list, out_list))

    return coord_args


def gen_threshold_args(true_idx, phase_idx, gsdir, nreals, low, high):
    '''Generate arguments for mwa_threshold'''

    sim_list = []
    out_list = []

    for i in range(nreals):
        simfl = gsdir + f't{true_idx}/phase{phase_idx}/real{i+1}_mwa.gsb'
        outfl = simfl
        sim_list.append(simfl)
        out_list.append(outfl)

    mwa_args = tuple((low, high, nreals, sim_list, out_list))

    return mwa_args


def gen_tag_args(true_idx, phase_idx, nreals, grid, dhdir, gsdir):
    '''Generate arguments for tag_ddh'''

    dhdata = dhdir + f't{true_idx}/ddh_au_p{phase_idx}_declus.out'
    rk_list = []
    out_list = []

    for i in range(nreals):

        rk_list.append(
            gsdir + f't{true_idx}/phase{phase_idx}/real{i+1}_mwa.gsb')
        out_list.append(
            dhdir + f't{true_idx}/phase{phase_idx}/ddh_au_real{i+1}_tag.out')

    tag_args = tuple((dhdata, rk_list, nreals, out_list, grid))

    return tag_args


def gen_clip_args(true_idx, phase_idx, nreals, simdir, outdir, grid,
                  clip_bounds):
    '''Generate args for clip_reals'''

    sim_list = []

    bxmin, bxmax, bymin, bymax, bzmin, bzmax = clip_bounds

    outfl = outdir + f't{true_idx}/phase{phase_idx}/realblk_clip.gsb'

    for i in range(nreals):
        sim_list.append(
            simdir + f't{true_idx}/phase{phase_idx}/realblk{i+1}.gsb')

    clip_args = tuple((nreals, grid, bxmin, bxmax, bymin, bymax, bzmin, bzmax,
                       sim_list, outfl))

    return clip_args


def gen_de_args(true_idx, phase_idx, nreals, vert_bounds, voutfl, rot,
                sublevels, stope_len, nstope, nvert, seed, simdir, outdir,
                fval_pars, de_pars, min_width, max_width):
    '''Generate intital verticies, bounds and clip realizations for DE'''

    de_args = []

    vxmin, vxmax, vymin, vymax, vzmin, vzmax = vert_bounds

    # define initial stope verticies
    rot_rad = rot * np.pi/180
    origin = (vxmin+(vxmax-vxmin)/2, vymin+(vymax-vymin)/2)

    # bbox dimensions
    del_x = vxmax - vxmin

    # sub-levels
    # sl1, sl2, sl3  = 645, 670, 700
    levels = sublevels
    nsub = len(levels)

    # number of stopes along strike
    stp_space = np.arange(vymin, vymax + stope_len, stope_len)
    nstp = len(stp_space)

    # arrays for verticies
    xx = np.ones(nstp)*vxmin
    yy = stp_space

    verts_hw = np.zeros((nstp, 3))
    verts_fw = np.zeros((nstp, 3))
    verts_hw[:, 0] = xx
    verts_hw[:, 1] = yy
    verts_fw[:, 0] = xx + del_x
    verts_fw[:, 1] = yy

    verts = np.vstack((verts_hw, verts_fw))
    verts = np.tile(verts, (nsub, 1))

    # set z coords for sublevels
    for i, l in enumerate(levels):
        ii = i*nstp*2
        jj = (i+1)*nstp*2
        verts[ii:jj, 2] = l

    # sort verticies based on stope sequence and
    # update list as some stopes share initial verticies
    verts_tmp = []
    for j in range(0, nsub+1, 2):
        for i in range(nstp-1):
            v1 = verts[(i+j*nstp)]
            v2 = verts[(i+j*nstp)+1]
            v3 = verts[(i+j*nstp)+nstp]
            v4 = verts[(i+j*nstp)+nstp+1]
            v5 = verts[(i+j*nstp)+nstp*2]
            v6 = verts[(i+j*nstp)+nstp*2+1]
            v7 = verts[(i+j*nstp)+nstp*2+nstp]
            v8 = verts[(i+j*nstp)+nstp*2+nstp+1]
            verts_tmp.append([v1, v2, v3, v4, v5, v6, v7, v8])

    verts_tmp = [i for sub in verts_tmp for i in sub]
    verts = np.array(verts_tmp)

    # rotate coordinates
    verts[:, 0], verts[:, 1] = rotate_around_point(verts[:, 0], verts[:, 1],
                                                   rot_rad, origin)

    # output file
    np.savetxt(voutfl, verts, fmt='%.2f')

    # bounds for verticies
    tolx = del_x/2 * np.cos(rot_rad)
    bounds = []

    for v in verts:
        xx = v[0]
        bxu = xx + tolx
        bxl = xx - tolx
        bounds.append((bxl, bxu))

    # clipped grid [nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize]
    grid = [20, 39892.5, 5, 48, 19832.5, 5, 20, 643.5, 3]

    # file paths
    simfl = simdir + f't{true_idx}/phase{phase_idx}/realblk_clip.gsb'

    out_list1 = []
    out_list2 = []
    for i in range(nstope):
        outfl1 = outdir + f't{true_idx}/phase{phase_idx}/stope{i+1}.out'
        outfl2 = outdir + f't{true_idx}/phase{phase_idx}/fitness{i+1}.out'
        out_list1.append(outfl1)
        out_list2.append(outfl2)

    de_args = tuple((simfl, nreals, nstope, seed, verts, bounds, grid, fval_pars, de_pars,
                     min_width, max_width, rot, out_list1, out_list2, true_idx, phase_idx))

    return de_args


def gen_job_list(nreals_true, nreals, nphase, nstope, seed, dirlist,
                 fval_pars, min_width, max_width, rot, clip_bounds,
                 vert_bounds, true_data, de_pars, len_comp, grid1,
                 grid2, grid3, varg1, varg2):
    '''Generate job list for parallel processing of VOI workflow'''

    # generate directories
    gen_directories(dirlist, nreals_true, nphase)
    refdir, dhdir, simdir1, chksim, gsdir, simdir2, stopedir, postdir, outdir, plotdir, pardir, tmp = dirlist

    # generate seed lists
    rseed_lst0 = gs.rseed_list(nreals_true, seed+1000)
    rseed_lst1 = gs.rseed_list(nreals_true*nreals*nphase, seed+2000)
    rseed_lst2 = gs.rseed_list(nreals_true*nreals*nphase, seed+3000)
    rseed_lst3 = gs.rseed_list(nreals_true*nphase, seed+4000)

    # grid definitons
    with open(grid1, 'r') as f:
        grid = f.read()
        griddef1 = gs.GridDef(grid)
    with open(grid2, 'r') as f:
        grid = f.read()
        griddef2 = gs.GridDef(grid)
    with open(grid3, 'r') as f:
        grid = f.read()
        griddef3 = gs.GridDef(grid)

    # variogram models
    with open(varg1, 'r') as f:
        varlg = f.read()
    with open(varg2, 'r') as f:
        varhg = f.read()

    jobs_ref_sim = []
    jobs_ref_post = []
    jobs = []

    for t in range(nreals_true):

        # reference model params
        # usgsim (1)
        with open(pardir + 'usgsim1.par', 'r') as f:
            parstr_ref = f.read()
            parstr_ref = parstr_ref.format(rseed=rseed_lst0[t],
                                           griddef=griddef1,
                                           true_data=true_data,
                                           outfl=refdir+f'real{t+1}.gsb',
                                           varhg=varhg)
            parpath = pardir + f't{t+1}/usgsim_ref{t+1}.par'
            with open(parpath, 'w', newline='', encoding='utf8') as pf:
                pf.write(parstr_ref)

        jobs_ref_sim.append((dict(parstr=parstr_ref, parfile=parpath)))

        # reference model block avg params
        with open(pardir + 'ublkavg.par', 'r') as f:
            parstr_avg = f.read()
            parstr_avg = parstr_avg.format(simfl=refdir + f'real{t+1}.gsb',
                                           griddef=griddef1,
                                           griddef_block=griddef3,
                                           outfl=refdir + f'realblk{t+1}.gsb')
            parpath = pardir + f't{t+1}/blkavg_ref{t+1}.par'
            with open(parpath, 'w', newline='', encoding='utf8') as pf:
                pf.write(parstr_avg)

        jobs_ref_post.append((dict(parstr=parstr_avg, parfile=parpath)))

        for j in range(nphase):

            # resample params
            grid = [griddef1.nx, griddef1.xmn, griddef1.xsiz,
                    griddef1.ny, griddef1.ymn, griddef1.ysiz,
                    griddef1.nz, griddef1.zmn, griddef1.zsiz]
            sample_args = gen_sample_args(
                t+1, j+1, grid, len_comp, refdir, dhdir)

            # decluster params
            with open(pardir + 'declus_nn2.par', 'r') as f:
                parstr_nn = f.read()
                parstr_nn = parstr_nn.format(datafl=dhdir+f't{t+1}/ddh_au_p{j+1}_all.out',
                                             griddef=griddef1,
                                             outfl=dhdir+f't{t+1}/ddh_au_p{j+1}_declus.out')

            # simulate au
            # usgsim (2) params
            usgsim_pars2 = []
            with open(pardir + 'usgsim2.par', 'r') as f:
                parstr_usg = f.read()
                for i in range(nreals):
                    parstr_usg_fmt = parstr_usg.format(rseed=rseed_lst1[i+j*nreals+t*nphase*nreals],
                                                       griddef=griddef2,
                                                       outfl=simdir1 +
                                                       f't{t+1}/phase{j+1}/real{i+1}.gsb',
                                                       varhg=varhg,
                                                       data=dhdir + f't{t+1}/ddh_au_p{j+1}_declus.out')
                    usgsim_pars2.append(parstr_usg_fmt)

            # add coords params
            grid = [griddef2.nx, griddef2.xmn, griddef2.xsiz,
                    griddef2.ny, griddef2.ymn, griddef2.ysiz,
                    griddef2.nz, griddef2.zmn, griddef2.zsiz]
            coord_args = gen_addcoord_args(t+1, j+1, nreals, grid, simdir1)

            # mwa params
            mwa_pars = []
            with open(pardir + 'maketrend.par', 'r') as f:
                parstr_mwa = f.read()
                for i in range(nreals):
                    parstr_mwa_fmt = parstr_mwa.format(simfl=simdir1+f't{t+1}/phase{j+1}/real{i+1}_xyz.gsb',
                                                       outfl=gsdir +
                                                       f't{t+1}/phase{j+1}/real{i+1}_mwa.gsb',
                                                       griddef=griddef2)
                    mwa_pars.append(parstr_mwa_fmt)

            # threshold params
            thresh_args = gen_threshold_args(t+1, j+1, gsdir, nreals, 1.2, 3.3)

            # tag ddh params
            grid = [griddef2.nx, griddef2.xmn, griddef2.xsiz,
                    griddef2.ny, griddef2.ymn, griddef2.ysiz,
                    griddef2.nz, griddef2.zmn, griddef2.zsiz]
            tag_args = gen_tag_args(t+1, j+1, nreals, grid, dhdir, gsdir)

            # simulate au
            # usgsim (3) params
            usgsim_pars3 = []
            with open(pardir + 'usgsim3.par', 'r') as f:
                parstr_usg = f.read()
                for i in range(nreals):
                    parstr_usg_fmt = parstr_usg.format(rseed=rseed_lst2[i+j*nreals+t*nphase*nreals],
                                                       griddef=griddef2,
                                                       outfl=simdir2 +
                                                       f't{t+1}/phase{j+1}/real{i+1}.gsb',
                                                       varlg=varlg,
                                                       varhg=varhg,
                                                       data=dhdir +
                                                       f't{t+1}/phase{j+1}/ddh_au_real{i+1}_tag.out',
                                                       rockfl=gsdir + f't{t+1}/phase{j+1}/real{i+1}_mwa.gsb')
                    usgsim_pars3.append(parstr_usg_fmt)

            # block average params
            blkavg_pars = []
            with open(pardir + 'ublkavg.par', 'r') as f:
                parstr_avg = f.read()
                for i in range(nreals):
                    parstr_avg_fmt = parstr_avg.format(simfl=simdir2 + f't{t+1}/phase{j+1}/real{i+1}.gsb',
                                                       griddef=griddef2,
                                                       griddef_block=griddef3,
                                                       outfl=simdir2 + f't{t+1}/phase{j+1}/realblk{i+1}.gsb')
                    blkavg_pars.append(parstr_avg_fmt)

            # merge reals params
            with open(pardir + 'merge_reals_gsb.par', 'r') as f:
                parstr_mrg = f.read()
                parstr_mrg = parstr_mrg.format(simfl=simdir2 + f't{t+1}/phase{j+1}/realblk',
                                               nreals=nreals,
                                               outfl=postdir + f't{t+1}/phase{j+1}/real_merge.gsb')

            # postsim params
            with open(pardir + 'upostsim.par', 'r') as f:
                parstr_post = f.read()
                parstr_post = parstr_post.format(simfl=postdir + f't{t+1}/phase{j+1}/real_merge.gsb',
                                                 nreals=nreals,
                                                 outfl=postdir + f't{t+1}/phase{j+1}/postsim.gsb')

            # clipping args
            grid = [griddef3.nx, griddef3.xmn, griddef3.xsiz,
                    griddef3.ny, griddef3.ymn, griddef3.ysiz,
                    griddef3.nz, griddef3.zmn, griddef3.zsiz]
            clip_args = gen_clip_args(
                t+1, j+1, nreals, simdir2, simdir2, grid, clip_bounds)

            # optimization args
            de_args = gen_de_args(t+1, j+1, nreals, vert_bounds, 'verts.out', -3, [645, 670, 700],
                                  20, nstope, 8, rseed_lst3[j +
                                                            t*nphase], simdir2, stopedir,
                                  fval_pars, de_pars, min_width, max_width)

            jobs.append((sample_args, parstr_nn, usgsim_pars2, coord_args, mwa_pars,
                         thresh_args, tag_args, usgsim_pars3, blkavg_pars, parstr_mrg,
                         parstr_post, clip_args, de_args, t+1, j+1, simdir1, simdir2,
                         dhdir, gsdir))

    return [jobs_ref_sim, jobs_ref_post, jobs]


#
# MAIN WORKFLOW FUNCTIONS
#


def gen_reference(jobs, nprocess):
    '''Parallel process reference models'''
    try:
        usgsim = gs.Program('usgsim')
        result = []
        pool = mp.Pool(nprocess)
        for job in jobs:
            result.append(pool.apply_async(usgsim.run, (), job))
        pool.close()
        pool.join()
    except Exception as error:
        print('Error generating reference model!')
        logging.error(traceback.format_exc())


def blkavg_reference(jobs, nprocess):
    '''Parallel process reference models'''
    try:
        ublkavg = gs.Program('ublkavg')
        result = []
        pool = mp.Pool(nprocess)
        for job in jobs:
            result.append(pool.apply_async(ublkavg.run, (), job))
        pool.close()
        pool.join()
    except Exception as error:
        print('Error block averaging reference model!')
        logging.error(traceback.format_exc())


def declus_nn(parstr):
    '''Call declus_nn with specified par file'''
    try:
        declus_nn = gs.Program('declus_nn')
        parfile = gs.rand_fname('par')
        declus_nn.run(parstr, parfile)
        os.remove(parfile)
    except Exception as error:
        print('Error declustering!')
        logging.error(traceback.format_exc())


def sim_continuous(parstr):
    '''Call usgsim with specified par file'''
    nreals = len(parstr)
    try:
        usgsim = gs.Program('usgsim')
        for i in range(nreals):
            parfile = gs.rand_fname('par')
            usgsim.run(parstr[i], parfile)
            os.remove(parfile)
    except Exception as error:
        print('Error simulating the continuous variable!')
        logging.error(traceback.format_exc())


def mwa(parstr):
    '''Call custom gsb maketrend with specified par file'''
    nreals = len(parstr)
    try:
        path = os.getcwd()
        maketrend = gs.Program(path + '\\maketrend.exe')
        for i in range(nreals):
            parfile = gs.rand_fname('par')
            maketrend.run(parstr[i], parfile)
            os.remove(parfile)
    except Exception as error:
        print('Error during moving window average!')
        logging.error(traceback.format_exc())


def block_avg(parstr):
    '''Call ublkavg with specified par file'''
    nreals = len(parstr)
    try:
        ublkavg = gs.Program('ublkavg')
        for i in range(nreals):
            parfile = gs.rand_fname('par')
            ublkavg.run(parstr[i], parfile)
            os.remove(parfile)
    except Exception as error:
        print('Error during block averaging!')
        logging.error(traceback.format_exc())


def post_sim(parstr1, parstr2):
    '''Call merge_reals_gsb and upostsim with specified par files'''
    try:
        merge_reals = gs.Program('merge_reals_gsb')
        upostsim = gs.Program('upostsim')
        parfile1 = gs.rand_fname('par')
        parfile2 = gs.rand_fname('par')
        merge_reals.run(parstr1, parfile1)
        upostsim.run(parstr2, parfile2)
        os.remove(parfile1)
        os.remove(parfile2)
    except Exception as error:
        print('Error during post processing!')
        logging.error(traceback.format_exc())


def stope_opt(simfl, nreals, nstope, seed, verts, bounds, grid, fval_pars, de_pars,
              min_width, max_width, rot, outfl1, outfl2, true_idx, phase_idx):
    '''Call differential evolution function for stope optimization'''

    rseeds = gs.rseed_list(nstope, seed)
    grid_clip_xyz = np.loadtxt('grid_clip_xyz.out')
    nvert = 8

    # fixed params for value function
    SG, price, cf, cost_m, cost_p, cost_ga, frec, fmine, fdil, froy, fstr = [
        *fval_pars.values()]

    # grid params
    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid
    nxyz = nx*ny*nz

    sim = gs.DataFile(simfl)
    sim = sim.data.values
    sim[np.isnan(sim)] = 0  # catch undefined blocks
    sim = sim.reshape((nxyz, nreals), order='F')

    # full set of arguments for DE fucntion
    args = dict(nreals=nreals,
                nx=nx,
                ny=ny,
                nz=nz,
                xsiz=xsize,
                ysiz=ysize,
                zsiz=zsize,
                xmin=xmin,
                ymin=ymin,
                zmin=zmin,
                SG=SG,
                price=price,
                cf=cf,
                cost_m=cost_m,
                cost_p=cost_p,
                cost_ga=cost_ga,
                frec=frec,
                fmine=fmine,
                fdil=fdil,
                froy=froy,
                fstr=fstr,
                min_width=min_width,
                max_width=max_width,
                grid_xyz=grid_clip_xyz,
                rot=rot,
                zsim=sim)

    print('Starting Differential Evolution')
    print(f'Reference model: {true_idx}')
    print(f'    Working on phase: {phase_idx}')

    for i in range(nstope):

        print(f'        Stope: {i+1}')

        verts_de = verts[i*nvert:(i+1)*nvert]
        bounds_de = bounds[i*nvert:(i+1)*nvert]

        args['verts'] = verts_de

        x, h, f, f_its = de(fobj, args, bounds_de,
                            rseeds[i], *de_pars.values())

        np.savetxt(outfl1[i], h, fmt='%.2f')
        np.savetxt(outfl2[i], f_its, fmt='%.2f')

    print(f'    Finished Phase: {phase_idx}')


def run_voi(sample_args, parstr_nn, parstr_sim1, coord_args,
            parstr_mwa, thresh_args, tag_args, parstr_sim2,
            parstr_blkavg, merge_pars, postsim_pars, clip_args,
            de_args, true_idx, phase_idx, simdir1, simdir2,
            dhdir, gsdir):
    '''Run main VOI workflow'''

    CHECK = 0

    try:
        sample_ddh3d(*sample_args)
        declus_nn(parstr_nn)
        sim_continuous(parstr_sim1)
        add_coord(*coord_args)
        mwa(parstr_mwa)
        mwa_threshold(*thresh_args)
        tag_ddh(*tag_args)
        sim_continuous(parstr_sim2)
        block_avg(parstr_blkavg)
        post_sim(merge_pars, postsim_pars)
        clip_reals(*clip_args)

        CHECK = 1

    except BaseException as error:
        print(
            f'An exception during resampling and resimulating occurred: {error}')

    # if everything above runs, remove realizations we no longer need
    if CHECK:

        # remove reals
        for fl in glob.glob(simdir1 + f't{true_idx}/phase{phase_idx}/real*.gsb'):
            os.remove(fl)

        # remove drillhole data
        for fl in glob.glob(dhdir + f't{true_idx}/phase{phase_idx}/ddh*.out'):
            os.remove(fl)

        # remove grade shells
        for fl in glob.glob(gsdir + f't{true_idx}/phase{phase_idx}/real*.gsb'):
            os.remove(fl)

        # remove reals within grade shells, but keep block averaged ones
        for fl in glob.glob(simdir2 + f't{true_idx}/phase{phase_idx}/real?.gsb'):
            os.remove(fl)
        for fl in glob.glob(simdir2 + f't{true_idx}/phase{phase_idx}/real??.gsb'):
            os.remove(fl)
        for fl in glob.glob(simdir2 + f't{true_idx}/phase{phase_idx}/real???.gsb'):
            os.remove(fl)

    try:
        stope_opt(*de_args)

    except BaseException as error:
        print(f'An exception during optimization occurred: {error}')


#
# VOI WORKFLOW
#

def voi_reference(jobs_sim, jobs_post, nprocess):
    '''Generate reference models in parallel for voi_main()'''

    gen_reference(jobs_sim, nprocess)
    blkavg_reference(jobs_post, nprocess)


def voi_main(jobs, nprocess):
    '''Main VOI workflow set up for each data configuration
       branch to be run in parallel'''

    result = []
    pool = mp.Pool(nprocess)
    for job in jobs:
        result.append(pool.apply_async(run_voi, job))
    pool.close()
    pool.join()


def voi_calc(refdir, stopedir, outdir, plotdir, nreals_true, nphase,
             nstope, cost_per_m, len_comp, grid, fval_pars):
    '''Calculate value of information from voi_main() workflow'''

    ndata = np.zeros((nphase, 2))
    for j in range(nphase):
        dat = np.loadtxt(f'ndata_p{j+1}.out')
        ndata[j, :] = dat.T

    # grid [nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize]
    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid
    grid_xyz = calc_grid_xyz(grid)
    np.savetxt('voi_calc_gridxyz.out', grid_xyz, fmt='%.2f')

    # get cost of data
    nsamp = ndata[:, 0]
    nddh = ndata[:, 1]
    vol2d = (ny*ysize) * (nz*zsize)
    vol3d = (nx*xsize) * (ny*ysize) * (nz*zsize)
    dens = nsamp / vol2d
    dhs = np.sqrt(vol3d/(len_comp*nsamp))
    samples_per_ha = dens * 100**2
    cost = nsamp * cost_per_m * len_comp

    # get value of stopes against reference model Vc
    Vc = {}
    for t in range(nreals_true):
        Vc[f't{t+1}'] = {}
        for j in range(nphase):
            Vc[f't{t+1}'][f'phase{j+1}'] = {}

    for t in range(nreals_true):

        ref_model = gs.DataFile(refdir + f'realblk{t+1}.gsb').data.values

        for j in range(nphase):

            gross_stpval = []
            tonnes_stp = []

            for i in range(nstope):

                hull = np.loadtxt(
                    stopedir + f't{t+1}/phase{j+1}/stope{i+1}.out')

                gross, avg_grade, tonnes = fval(ref_model, hull, grid_xyz, xsize, ysize, zsize,
                                                *fval_pars.values())
                if gross >= 0:
                    gross_stpval.append(gross)
                    tonnes_stp.append(np.int(tonnes))
                    Vc[f't{t+1}'][f'phase{j+1}'][f'val_stp{i+1}'] = gross
                    Vc[f't{t+1}'][f'phase{j+1}'][f'gr_stp{i+1}'] = avg_grade
                    Vc[f't{t+1}'][f'phase{j+1}'][f'tonnes_stp{i+1}'] = tonnes

            Vc[f't{t+1}'][f'phase{j+1}'][f'num_stp'] = len(gross_stpval)
            Vc[f't{t+1}'][f'phase{j+1}'][f'avg_tonnes'] = np.int(
                np.mean(tonnes_stp))
            Vc[f't{t+1}'][f'phase{j+1}'][f'gross_val'] = np.sum(gross_stpval)

            with open(outdir + f't{t+1}/phase{j+1}.out', 'w') as f:
                f.write(json.dumps(Vc[f't{t+1}'][f'phase{j+1}'], indent=2))

    # get value of current information V0
    V0 = np.zeros(nreals_true)
    for t in range(nreals_true):
        V0[t] = Vc[f't{t+1}'][f'phase1'][f'gross_val']

    # calculate gross (GVOI) and net (NVOI) voi
    GVOI = np.zeros((nphase, nreals_true))
    NVOI = np.zeros((nphase, nreals_true))
    EGVOI = np.zeros(nphase)
    ENVOI = np.zeros(nphase)

    for t in range(nreals_true):

        for j in range(nphase):

            vc = Vc[f't{t+1}'][f'phase{j+1}'][f'gross_val']
            gvoi = vc - V0[t]
            GVOI[j, t] = gvoi

        NVOI[:, t] = GVOI[:, t] - cost
        NVOI[0, t] = 0

    EGVOI = np.mean(GVOI, axis=1)
    ENVOI = np.mean(NVOI, axis=1)

    fmt = '%.2f'
    np.savetxt(outdir + 'nvoi.out', NVOI, fmt=fmt)
    np.savetxt(outdir + 'gvoi.out', GVOI, fmt=fmt)
    np.savetxt(outdir + 'cost.out', cost, fmt=fmt)
    np.savetxt(outdir + 'envoi.out', ENVOI, fmt=fmt)
    np.savetxt(outdir + 'egvoi.out', EGVOI, fmt=fmt)

    # generate figure
    fig, ax = plt.subplots()

    ax1 = ax.twiny()
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_label_position('bottom')
    ax1.spines['bottom'].set_position(('outward', 40))

    # expected curves
    ax.plot(samples_per_ha, EGVOI,
            samples_per_ha, ENVOI,
            samples_per_ha, cost,
            marker='.', zorder=2)
    ax.axhline(0, c='k')
    ax.set_xlabel('Samples per Hectare')
    ax.set_ylabel('Dollars ($)')
    ax.set_xlim(0, samples_per_ha.max()+250)
    ax.grid(alpha=0.5)

    # all curves
    for i in range(nreals_true):
        ax.plot(samples_per_ha, GVOI[:, i], c='C0',
                alpha=0.75, lw=0.5, zorder=0)
        ax.plot(samples_per_ha, NVOI[:, i], c='C1',
                alpha=0.75, lw=0.5, zorder=0)

    xmax = np.int(np.ceil(samples_per_ha[-1]/1000) * 1000)
    step = 1000
    xx = np.arange(0, xmax+step, step)
    nsamp2 = (xx*vol2d) / 100**2
    dhs_eq = np.round(np.sqrt(vol3d/(len_comp*nsamp2)), 1)

    ax1.set_xlabel('Equivalent Data Spacing (m)')
    ax1.set_xticks(np.arange(len(xx)))
    ax1.set_xticklabels(dhs_eq)
    ax.legend(['Exp. GVOI', 'Exp. NVOI', 'Cost of Data'])

    fig.tight_layout()
    plt.savefig(plotdir + 'voi_curve.pdf')
