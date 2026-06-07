import os
import math
import quaph
from quaph._compare import run_compare
from quaph._dmrg import run_dmrg_itensor

"""
 Note from Maggie to Adam: I specified this example phase diagram with the same parameters as the one I simulated on perlmutter, 
 except I removed the parallelism features. So, the current dmrg data in examples/logs and examples/plots didn't come from running 
 run_dmrg.py on local machine, but from a previoous perlmutter job with my slurm scripts below. 

 The data should be identical because they have the same specifications.

 Let me know if you want me to run a simpler/smaller one, if you indend to have users be able to run and finish these examples in time on
 their local machine.
"""
MODEL = "haldane"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"
X_RANGE = None
Y_RANGE = (0.0, 1.0, 0.1) 
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.0}
DMRG_NSWEEPS = 4
DMRG_MAXDIMS = "20,50,100,200"
DMRG_CUTOFF = 1e-9

_HERE = os.path.dirname(os.path.abspath(__file__))
_LATTICE_TAG = "x".join(str(x) for x in LATTICE)
_LOG_DIR = os.path.join(_HERE, "logs")
_PLOT_DIR = os.path.join(_HERE, "plots")

_DMRG_LOG = os.path.join(
    _LOG_DIR,
    MODEL,
    _LATTICE_TAG,
    "dmrg",
    "dmrg-{}-vs-{}.json".format(X_PARAM, Y_PARAM),
)
_COMPARE_LOG = os.path.join(
    _LOG_DIR,
    MODEL,
    _LATTICE_TAG,
    "compare-{}-vs-{}.json".format(X_PARAM, Y_PARAM),
)

if os.path.exists(_DMRG_LOG):
    print("DMRG log already exists: {}".format(_DMRG_LOG))
else:
    dmrg_result = run_dmrg_itensor(
        quaph.get_model(MODEL),
        lattice=LATTICE,
        x_param=X_PARAM,
        x_range=X_RANGE,
        y_param=Y_PARAM,
        y_range=Y_RANGE,
        n_occ=None,
        model_params=MODEL_PARAMS,
        nsweeps=DMRG_NSWEEPS,
        maxdims=DMRG_MAXDIMS,
        cutoff=DMRG_CUTOFF,
        log_dir=_LOG_DIR,
        plot_dir=_PLOT_DIR,
    )
    print("Wrote {}".format(dmrg_result["summary_path"]))

if os.path.exists(_COMPARE_LOG):
    print("Plotting from existing compare log: {}".format(_COMPARE_LOG))
    quaph.load_result(_COMPARE_LOG).plot()
else:
    compare_result = run_compare(
        quaph.get_model(MODEL),
        lattice=LATTICE,
        x_param=X_PARAM,
        x_range=X_RANGE,
        y_param=Y_PARAM,
        y_range=Y_RANGE,
        n_occ=None,
        model_params=MODEL_PARAMS,
        algorithms=["exact", "dmrg"],
        quantum_pipeline="ideal",
        dmrg_nsweeps=DMRG_NSWEEPS,
        dmrg_maxdims=DMRG_MAXDIMS,
        dmrg_cutoff=DMRG_CUTOFF,
        log_dir=_LOG_DIR,
        plot_dir=_PLOT_DIR,
    )
    print("Wrote {}".format(compare_result["summary_path"]))

# To include quantum algorithms in the comparison:
# algorithms=["exact", "vqe", "iqpe", "dmrg"]
# vqe_iters=200, vqe_layers=2, vqe_reps=1,
# iqpe_time=0.2, iqpe_trot=2, iqpe_iters=2, iqpe_reps=1,
