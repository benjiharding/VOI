import voi_utils as voi

# global parameters
nreals_true = 2
nphase = 6
nreals = 1
nprocess = 4
seed = 17
fval_pars = dict(SG=2.80,
                 price=1250,
                 cf=31.1035,
                 cost_m=42,
                 cost_p=25,
                 cost_ga=12,
                 frec=0.94,
                 fmine=0.95,
                 fdil=0.10,
                 froy=1-0.05,
                 fstr=1-0.084)
cost_per_m = 124.52
len_comp = 3
grid = [45, 39852.5, 5.0,
        65, 19802.5, 5.0,
        39, 616.5, 3.0]
nstope = 22
nvert = 8
stope_len = 20
min_width = 10
max_width = 25
rot = -3

# differential evolution parameters
de_pars = dict(mut=(0.5, 1),
               crossp=0.9,
               popsize=10,
               its=500,
               K=25,
               g=10e-6)

# bounds to clip realizations
clip_bounds = [39890, 39990, 19830, 20070, 642, 703]
# bounds for initial vertex creation
vert_bounds = [39920, 39960, 19840, 20060, 645, 700]

# delcustered input data for reference model
true_data = 'declus_nn.out'

# directories
refdir = '01 ref models/'
dhdir = '02 ddh data/'
simdir1 = '03 realizations/'
chksim = simdir1 + 'varsim/'
gsdir = '04 grade shells/'
simdir2 = '05 final realizations/'
stopedir = '06 stopes/'
postdir = '07 post process/'
outdir = '08 outputs/'
plotdir = '09 plots/'
pardir = '99 parfiles/'
tmp = 'tmp/'
dirlist = [refdir, dhdir, simdir1, chksim, gsdir, simdir2, stopedir,
           postdir, outdir, plotdir, pardir, tmp]

# generate job list for parallel processing
job0, job1, job2 = voi.gen_job_list(nreals_true, nreals, nphase, nstope, seed, dirlist, fval_pars,
                                    min_width, max_width, rot, clip_bounds, vert_bounds, true_data,
                                    de_pars, grid1='grid1.grd', grid2='grid2.grd', grid3='grid3.grd',
                                    varg1='varlg.var', varg2='varhg.var')

# generate reference models
voi.voi_reference(job0, job1, nprocess)

# call main VOI workflow
voi.voi_main(job2, nprocess)

# calculate VOI
voi.voi_calc(refdir, stopedir, outdir, plotdir, nreals_true, nphase,
             nstope, cost_per_m, len_comp, grid, fval_pars)
