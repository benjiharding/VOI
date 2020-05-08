import itertools
import os

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist
from scipy import interpolate

#
# Main function to eleminiate drillholes in a spatially reasonable way
# and calculate drillhole spacing of remaining drillholes
#


def spatial_elim(datafl, sample_len, seed, grid, dhs_option, dhs_pars,
                 dhs_search, dhs_search_ang, ztol, wcent=0.01, wrand=0.001,
                 wrem=0.5, wstd=0.5, wprev=0.01, dhs_clip=0, group=0, group_tol=0.10,
                 group_lvls=0, treat_as_cent=0, header=0, delim=None, outfl_col=None,
                 outfl_dh=None, outfl_dhs=None, outfl_avgdhs=None, outdir=None):
    '''
    datafl: file drillhole collars in format [x,y,z,azm,dip,length]
    sample_len: sample length for synthetic drillholes   
    seed: random number seed    
    grid: grid definition in format [nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize]
    dhs_option: option to search and calculate DHS in 2D like dhs3d (0) or in 3D like kt3dn (1) 
    dhs_pars: if dhs_option is 0; range of samples to search for DHS calculations, ex. [2,3,4,5,6]
    dhs_search: if dhs_option is 1; search ellipse for data spacing calculation  
    dhs_search_ang: if dhs_option is 1; search ellipse angles
    ztol: tolerance in z coordinte to accept samples for dhs_option=0 DHS calculation   
    wcent: weight to distance to data centroid for elimination algorithm    
    wrand: weight to random component for elimination algorithm    
    wrem: weight to distance between remaining drillholes for elimination algorithm    
    wstd: weight to stdev of distance between remaining drillholes for elimination algorithm    
    dhs_clip: clipping of DHS model: 0 = none, 1 = convex hull, >1 = that distance to nn    
    group: boolean to group holes with same collar location (ie drillhole fans)    
    header: number of header lines to skip for datafl import    
    delim: delimiter for datafl   
    outfl_dh: output for drillhole groups after elimintion (first group contains all drillholes), 
        set to None for no output    
    outfl_dhs: output for DHS after elimintion, set to None for no output   
    outfl_avgdhs: output for elimination sequence and resulting DHS, set to None for no output
    '''

    random_state = np.random.RandomState(seed)
    epsillon = 1e-10

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # grid definition

    nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize = grid
    grid_xyz, grid_idx = calc_grid_xyz(grid, return_idx=1)
    nxyz = nx*ny*nz

    # generate sample locations from dh data

    dh, col = syn_samples(datafl, sample_len, group=group, group_tol=group_tol,
                          group_lvls=group_lvls, header=header, delim=delim)

    # set id cols based on group/no group

    dhidcol = 0

    if group == 0:
        idcol = 0
        xcol, ycol, zcol = 1, 2, 3
        x = dh[:, xcol]
        y = dh[:, ycol]
        z = dh[:, zcol]
        xyz = dh[:, xcol:zcol+1]

    else:
        idcol = 1
        xcol, ycol, zcol = 2, 3, 4
        x = dh[:, xcol]
        y = dh[:, ycol]
        z = dh[:, zcol]
        xyz = dh[:, xcol:zcol+1]

    dhids = dh[:, idcol]  # all numeric dhids, not necessarily 0 indexed
    dhid = np.unique(dhids)  # unique ids
    ndh = len(dhid)
    dhidx = np.arange(ndh)  # zero based dh index

    # calc avg. dist. to centroid of each dh or group

    xcent = np.mean(x)
    ycent = np.mean(y)
    zcent = np.mean(z)

    cent = np.array([xcent, ycent, zcent])
    dhdcent = np.zeros(ndh)

    for i, idx in enumerate(dhid):

        if treat_as_cent == 0:

            dhxyz = xyz[dh[:, idcol] == idx]
            centxyz = cent.repeat(len(dhxyz)).reshape(dhxyz.shape, order='F')
            dmat = cdist(dhxyz, centxyz)
            dhdcent[i] = np.mean(dmat)

        else:

            dhxyz = np.mean(xyz[dh[:, idcol] == idx], axis=0)
            dx = dhxyz[0] - xcent
            dy = dhxyz[1] - ycent
            dz = dhxyz[2] - zcent
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            dhdcent[i] = dist

    # calc avg. dist. between all pairs of dhs or groups

    dhdists = np.zeros((ndh, ndh))
    combo = itertools.combinations(dhidx, 2)

    for i, j in combo:

        if treat_as_cent == 0:

            idx1 = dhid[i]
            idx2 = dhid[j]
            dh1 = xyz[dh[:, idcol] == idx1]
            dh2 = xyz[dh[:, idcol] == idx2]
            dmat = cdist(dh1, dh2)
            dhdists[i, j] = np.mean(dmat)
            dhdists[j, i] = dhdists[i, j]

        else:

            idx1 = dhid[i]
            idx2 = dhid[j]
            dh1 = np.mean(xyz[dh[:, idcol] == idx1], axis=0)
            dh2 = np.mean(xyz[dh[:, idcol] == idx2], axis=0)
            dx = dh1[0] - dh2[0]
            dy = dh1[1] - dh2[1]
            dz = dh1[2] - dh2[2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            dhdists[i, j] = dist
            dhdists[j, i] = dhdists[i, j]

    # loop over dhs or groups evaluating remaining 'goodness'

    dec_idx = []

    for iloop in dhidx:

        ndh_rem = len(dhidx)
        good = np.zeros(ndh_rem)

        for i, idx in enumerate(dhidx):

            rn = random_state.rand()
            dc = dhdcent[idx]
            idxs = [dh for dh in dhidx if dh != idx]

            if len(idxs) >= 2:
                dist = dhdists[idxs, :]
                dist = dist[:, idxs]
                dd = np.mean(dist[dist != 0])
                std = np.std(dist[dist != 0])

            elif len(idxs) == 1:
                dd = dhdists[dhidx[0], dhidx[1]]
                std = 0

            else:
                dd = 0
                std = 0

            if iloop > 0:
                prev_idx = dec_idx[iloop-1]
                dp = dhdists[prev_idx, idx]

            else:
                dp = 0

            good[i] = (wcent*dc) + (wrand*rn) + \
                (wrem*dd) + (wprev*dp) - (wstd*std)
            best = np.argmax(good)

        dec_idx.append(dhidx[best])
        dhidx = np.delete(dhidx, best)

    dec_ids = dhid[dec_idx].astype(int)
    dec_ids = np.insert(dec_ids, 0, -999)  # keep original config for DHS

    # set up global variables for data spacing calcs

    if dhs_option == 0:
        print('DHS option 0: drillhole spacing')
        print('\n')

    else:

        radius = dhs_search[0]
        radius1 = dhs_search[1]
        radius2 = dhs_search[2]

        sanis1 = radius1 / radius
        sanis2 = radius2 / radius

        ang1 = dhs_search_ang[0]
        ang2 = dhs_search_ang[1]
        ang3 = dhs_search_ang[2]

        # generate rotation matrix

        R = rotmat(ang1, ang2, ang3, sanis1, sanis2)

        # generate data spacing table
        ntest = 100
        noff = 20
        dsf, _, _ = dstable(dhs_search, sample_len, ntest, noff)

        # rotate and scale grid coordinates

        rot_grid_xyz = rot_from_matrix(
            grid_xyz[:, 0], grid_xyz[:, 1], grid_xyz[:, 2], R, cent)
        rot_grid_xyz = rot_grid_xyz / radius

        print('DHS option 1: equivalent square data spacing')
        print('\n')

    # set up clipping limits using all data if required

    if dhs_clip == 1:  # convex hull

        clip_arr = in_hull(grid_xyz, xyz) * 1

        print('Clipping option: convex hull')
        print('\n')

    elif dhs_clip > 1:  # distance to nn

        clip_tree = cKDTree(xyz)
        cnn, _ = clip_tree.query(grid_xyz, k=1)
        clip_arr = np.where(cnn <= dhs_clip, 1, 0)

        print(f'Clipping option: nearest neighbour within {dhs_clip}m')
        print('\n')

    else:
        clip_arr = np.ones(len(grid_xyz))

        print('Clipping option: none')
        print('\n')

    # eliminate dhs or groups cumulatively and calculate data spacing

    avg_dhs = np.zeros(ndh-1)

    for ii in range(ndh-1):

        dh_rem = dh[dh[:, idcol] != dec_ids[ii]]
        dh = dh_rem

        col_rem = col[col[:, 0] != dec_ids[ii]]
        col = col_rem

        if outfl_dh is not None:

            if group == 0:
                np.savetxt(outdir + f'{outfl_dh}_{ii+1}.out', dh_rem,
                           fmt='%i %.2f %.2f %.2f %.2f')
            else:
                np.savetxt(outdir + f'{outfl_dh}_{ii+1}.out', dh_rem,
                           fmt='%i %i %.2f %.2f %.2f %.2f')

        if outfl_col is not None:

            np.savetxt(outdir + f'{outfl_col}_{ii+1}.out', col_rem[:, 1:],
                       fmt='%.2f %.2f %.2f %.2f %.2f %.2f')

        # calculate data spacing for each remaining group
        # sort data by dhid and z level first

        dh_rem = dh_rem[dh_rem[:, dhidcol].argsort()]
        dh_rem = dh_rem[dh_rem[:, zcol].argsort()]
        xyz_rem = dh_rem[:, xcol:zcol+1]

        #
        # 2D dhs3d like drillhole spacing calc
        #

        if dhs_option == 0:

            # find the data on each z level and build kDTrees

            trees = {}

            num_dat = np.zeros((nz))
            idx_dat = np.zeros((nz))

            dat_z = dh_rem[:, zcol]

            for i in range(nz):

                grid_z = zmin + i*zsize - zsize / 2
                num = dat_z[(dat_z > grid_z-ztol) &
                            (dat_z <= grid_z+zsize+ztol)]
                num_dat[i] = len(num)

                if len(num) > 0:
                    idx_dat[i] = np.min(np.argwhere(
                        (dat_z > grid_z-ztol) & (dat_z <= grid_z+zsize+ztol)))
                else:
                    idx_dat[i] = None

                if np.isfinite(idx_dat[i]):

                    idx1 = int(idx_dat[i])
                    idx2 = int(idx_dat[i] + num_dat[i])
                    trees[f'iz{i+1}_idx'] = (idx1, idx2)
                    trees[f'iz{i+1}_tree'] = cKDTree(xyz_rem[idx1:idx2])

                else:
                    idx1 = None
                    idx2 = None
                    trees[f'iz{i+1}'] = None

            # loop over all grid nodes and calculate dhs

            print(
                f'Starting DHS calculations for {(ndh)-ii} remaining drillholes')
            print('\n')

            tmp_dhs = np.zeros(len(dhs_pars))
            dhs = np.zeros(nxyz)

            for idx in range(nxyz):  # could loop over nz and query entire level

                loc_x = grid_xyz[idx, 0]
                loc_y = grid_xyz[idx, 1]
                loc_z = grid_xyz[idx, 2]
                iz = grid_idx[idx, 2]

                # query the tree for all neighbours

                ndat = int(num_dat[iz])

                if ndat > dhs_pars[-1]:
                    ndat = dhs_pars[-1]

                if ndat > 0:
                    tree = trees[f'iz{iz+1}_tree']
                    idx1, idx2 = trees[f'iz{iz+1}_idx']
                    nn, ni = tree.query((loc_x, loc_y, loc_z), k=ndat)
                else:
                    tmp_dhs[:] = np.nan

                tmp_dis = np.zeros(ndat)
                tmp_id = np.zeros(ndat, dtype=int)

                j = -1

                for i in range(ndat):

                    if ndat <= 1:
                        j = -1
                    else:
                        j += 1
                        tmp_dis[j] = nn[i]
                        tmp_id[j] = dh_rem[:, dhidcol][np.arange(idx1, idx2)[
                            ni]][i]

                        # only take 1 sample per dh
                        for k in range(j):
                            if np.abs(tmp_id[k]-tmp_id[j]) < epsillon:
                                j -= 1
                            elif j == ndat:
                                break
                j += 1

                if j == 0:
                    dhs[idx] == np.nan

                for i in range(len(dhs_pars)):

                    if dhs_pars[i] <= j:
                        tmp_dhs[i] = (
                            (tmp_dis[i] + tmp_dis[i+1]) / 2) * np.sqrt(2 / dhs_pars[i])
                    else:
                        tmp_dhs[i] = np.nan

                dhs[idx] = np.nanmean(tmp_dhs)

        #
        # kt3dn like data spacing calc
        #

        if dhs_option == 1:

            # rotate and scale data coordinates

            rot_xyz = rot_from_matrix(
                xyz_rem[:, 0], xyz_rem[:, 1], xyz_rem[:, 2], R, cent)

            # scale again so search is a unit sphere

            rot_xyz = rot_xyz / radius

            # build kdtree and query neighbours

            dhs = np.zeros(nxyz)
            tree = cKDTree(rot_xyz)
            neighbours = tree.query_ball_point(rot_grid_xyz, r=1)

            print(
                f'Starting DHS calculations for {(ndh)-ii} remaining drillholes')
            print('\n')

            for i, nn in enumerate(neighbours):

                num_found = len(nn)
                dhs[i] = dsf(num_found)

        # clip DHS model

        dhs = np.where(clip_arr == 1, dhs, np.nan)

        avg_dhs[ii] = np.nanmean(dhs)

    # write out files

        if outfl_dhs is not None:
            np.savetxt(outdir + f'{outfl_dhs}_{ii+1}.out', dhs, fmt='%.5f')

    out_avgdhs = np.hstack((dec_ids[:-2].reshape(-1, 1),
                            avg_dhs.reshape(-1, 1)))

    if outfl_avgdhs is not None:
        np.savetxt(outdir + f'{outfl_avgdhs}.out', out_avgdhs, fmt='%i %.5f')

    print('\n')
    print('Finished DHS calculations')


#
# Helper functions
#

def syn_samples(dh_data, sample_len, group=0, group_tol=0.10, group_lvls=0,
                header=0, delim=None):
    '''Generate synthetic drillhole samples from collar data and sample length

    dhdata format: [x,y,z,azm,dip,length]
    sample_len: composite length
    group: group drillholes with the same collar location?
    tol: tolerance for collar "closeness" in future? 
    output: [dhid, gid, x, y, z, mid_pt]
    '''

    np.set_printoptions(precision=3, suppress=True)

    dhdata = np.loadtxt(dh_data, skiprows=header, delimiter=delim)

    # setup ddh properties
    col = dhdata[:, 0:3]
    azm = dhdata[:, 3]
    dip = dhdata[:, 4]
    length = dhdata[:, 5]

    # get sample coordinates
    nddh = col.shape[0]
    holeid = np.arange(1, nddh+1)
    coords = []

    print(f'Generating sample coordinates for {nddh} drillholes')
    print('\n')

    if group == 0:

        print('Not considering drillhole groups')
        print('\n')

        collar_groups = np.hstack(
            (np.array(holeid).reshape(-1, 1), dhdata))
        np.savetxt('collar_groups.out', collar_groups,
                   fmt='%i %.2f %.2f %.2f %.2f %.2f %.2f')

        for ddh in zip(col, azm, dip, length, holeid):
            col_tmp = ddh[0]
            azm_tmp = ddh[1]
            dip_tmp = ddh[2]
            len_tmp = ddh[3]
            holeid_tmp = ddh[4]
            azm_rad = azm_tmp * np.pi/180
            dip_rad = dip_tmp * np.pi/180
            samp_tmp = np.arange(sample_len/2, len_tmp, sample_len)
            coords_tmp = np.zeros((len(samp_tmp), len(ddh)))

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

    else:

        if group_lvls == 0:

            # get group ids for shared collars
            groupid = [group]
            for i, row in enumerate(col[1:]):

                i += 1

                if np.allclose(row, col[i-1, :], atol=group_tol):
                    groupid.append(groupid[i-1])

                else:
                    groupid.append(groupid[i-1]+1)

            collar_groups = np.hstack(
                (np.array(groupid).reshape(-1, 1), dhdata))
            np.savetxt('collar_groups.out', collar_groups,
                       fmt='%i %.2f %.2f %.2f %.2f %.2f %.2f')

        else:  # group considering levels

            dhdata = dhdata[dhdata[:, 0].argsort()]  # sort x
            dhdata = dhdata[dhdata[:, 1].argsort()]  # sort y
            col = dhdata[:, 0:3]
            azm = dhdata[:, 3]
            dip = dhdata[:, 4]
            length = dhdata[:, 5]

            groupid = [group]

            for i, row in enumerate(col[1:]):  # skip the first row

                i += 1

                if np.allclose(row[0:2], col[i-1, 0:2], atol=group_tol):
                    groupid.append(groupid[i-1])

                else:
                    groupid.append(groupid[i-1]+1)

            collar_groups = np.hstack(
                (np.array(groupid).reshape(-1, 1), dhdata))
            np.savetxt('collar_groups.out', collar_groups,
                       fmt='%i %.2f %.2f %.2f %.2f %.2f %.2f')

        print(f'Considering {len(np.unique(groupid))} drillhole groups')
        print('\n')

        for ddh in zip(col, azm, dip, length, holeid, groupid):
            col_tmp = ddh[0]
            azm_tmp = ddh[1]
            dip_tmp = ddh[2]
            len_tmp = ddh[3]
            holeid_tmp = ddh[4]
            groupid_tmp = ddh[5]
            azm_rad = azm_tmp * np.pi/180
            dip_rad = dip_tmp * np.pi/180
            samp_tmp = np.arange(sample_len/2, len_tmp, sample_len)
            coords_tmp = np.zeros((len(samp_tmp), len(ddh)))

            for i, s in enumerate(samp_tmp):
                l_plan = np.cos(dip_rad) * s
                dx = np.sin(azm_rad) * l_plan
                dy = np.cos(azm_rad) * l_plan
                dz = np.sin(dip_rad) * s
                coords_tmp[i, 0] = holeid_tmp
                coords_tmp[i, 1] = groupid_tmp
                coords_tmp[i, 2] = col_tmp[0] + dx
                coords_tmp[i, 3] = col_tmp[1] + dy
                coords_tmp[i, 4] = col_tmp[2] - dz
                coords_tmp[i, 5] = s

            coords.append(coords_tmp)
        coords = np.vstack(coords)

    return coords, collar_groups


def calc_grid_xyz(grid, return_idx=0):
    '''Return xyz coords of grid nodes, optionally return grid indicies
    grid format: [nx, xmin, xsize, ny, ymin, ysize, nz, zmin, zsize]
    '''

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

    if return_idx == 0:
        return np.array(grid_xyz)
    else:
        return np.array(grid_xyz), np.array(grid_idx)


def in_hull(p, hull):
    '''Return indicies of points p inside hull '''

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull, qhull_options='QJ')

    return hull.find_simplex(p) >= 0


def dstable(search, len_comp, ntest, noff):
    '''After dataspactable subroutine by JL Deutsch 2015'''

    radius = search[0]
    radius1 = search[1]
    radius2 = search[2]

    sanis1 = radius1 / radius
    sanis2 = radius2 / radius

    lds = np.zeros(ntest)
    ndf = np.zeros(ntest)

    for i in range(ntest, 0, -1):

        lspace = radius / ntest * i
        lds[i-1] = lspace

        nmaj = int(np.ceil(radius/lspace) + 1)
        nmin = int(np.ceil(radius*sanis1/lspace) + 1)

        sumheight = 0

        for j in range(noff):

            majorig = 0 - radius - lspace / (j+1)
            minorig = 0 - radius*sanis1 - lspace / (j+1)

            for imaj in range(nmaj):

                xmaj = majorig + imaj*lspace

                for imin in range(nmin):

                    xmin = minorig + imin*lspace

                    rcheck = xmaj**2 / radius**2 + xmin**2 / (radius*sanis1)**2

                    if rcheck <= 1:

                        sumheight = sumheight + \
                            np.sqrt((1-rcheck)*(radius*sanis2)**2)

            ndf[i-1] = sumheight / (len_comp*noff)

    # return the function for interp
    dsf = interpolate.interp1d(
        ndf, lds, bounds_error=False, fill_value=(lds.max()+1, lds.max()+1))

    return dsf, ndf, lds


def rotmat(ang1, ang2, ang3, anis1, anis2):
    '''After GSLIB subroutine 'setrot' by CV Deutsch 1992'''

    epsillon = 1e-10
    R = np.eye(3)

    alpha = ang1 * np.pi / 180
    beta = ang2 * np.pi / 180
    theta = ang3 * np.pi / 180

    sina = np.sin(alpha)
    sinb = np.sin(beta)
    sint = np.sin(theta)
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cost = np.cos(theta)

    afac1 = 1 / max(anis1, epsillon)
    afac2 = 1 / max(anis2, epsillon)

    # from Geostats Lessons

    R[0, 0] = afac1 * (cosa*cost + sina*sinb*sint)
    R[1, 0] = afac1 * (sina*cosb)
    R[2, 0] = afac1 * (cosa*sint - sina*sinb*cost)
    R[0, 1] = -sina*cost + cosa*sinb*sint
    R[1, 1] = cosa*cosb
    R[2, 1] = -sina*sint - cosa*sinb*cost
    R[0, 2] = afac2 * (-cosb*sint)
    R[1, 2] = afac2 * (sinb)
    R[2, 2] = afac2 * (cosb*cost)

    return R


def rot_from_matrix(x, y, z, rotmat, origin):
    '''3D rotation from existing rotation matrix'''

    ox = origin[0]
    oy = origin[1]
    oz = origin[2]

    ax = (x - ox).reshape(-1, 1)
    ay = (y - oy).reshape(-1, 1)
    az = (z - oz).reshape(-1, 1)

    adj_xyz = np.hstack((ax, ay, az))

    rot_xyz = adj_xyz @ rotmat

    rot_xyz[:, 0] = rot_xyz[:, 0] + ox
    rot_xyz[:, 1] = rot_xyz[:, 1] + oy
    rot_xyz[:, 2] = rot_xyz[:, 2] + oz

    return rot_xyz
