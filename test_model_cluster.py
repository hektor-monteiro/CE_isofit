import oc_tools_padova_dr3 as oc
import numpy as np

print("Loading grid...")
oc.load_mod_grid('grids/', isoc_set='GAIA_eDR3')

# model_cluster parameters
age = 8.0 # log age
dist = 1.0 # kpc
FeH = 0.0 # metallicity
Av = 0.5
bin_frac = 0.3
nstars = 100
bands = ['Gmag', 'G_BPmag', 'G_RPmag']
refMag = 'Gmag'
seed = 42

print("Generating first cluster...")
res1 = oc.model_cluster(age, dist, FeH, Av, bin_frac, nstars, bands, refMag, seed=seed)

print("Generating second cluster...")
res2 = oc.model_cluster(age, dist, FeH, Av, bin_frac, nstars, bands, refMag, seed=seed)

print("Comparing outputs...")
# check equality
if len(res1) != len(res2):
    print(f"Lengths differ: {len(res1)} vs {len(res2)}")
else:
    diff_found = False
    for name in res1.dtype.names:
        if not np.allclose(res1[name], res2[name], equal_nan=True):
            print(f"Column {name} differs!")
            diff_found = True

    if not diff_found:
        print("Outputs are identical.")
