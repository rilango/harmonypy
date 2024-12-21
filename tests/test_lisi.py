import os
import harmonypy as hm

GPU = False
try:
    if os.environ.get('HARMONYPY_CPU', '0') == '1':
        raise ModuleNotFoundError("HARMONYPY_CPU is set to 1")
    import cudf as pd
    import cupy as np
    GPU = True
except ModuleNotFoundError:
    import pandas as pd
    import numpy as np


def test_lisi():

    X = pd.read_csv("data/lisi_x.tsv.gz", sep = "\t")
    metadata = pd.read_csv("data/lisi_metadata.tsv.gz", sep = "\t")
    label_colnames = metadata.columns
    perplexity = 30

    lisi = hm.compute_lisi(X, metadata, label_colnames, perplexity)

    lisi_test = pd.read_csv("data/lisi_lisi.tsv.gz", sep="\t")
    lisi_test = lisi_test.iloc[:,-2:].to_numpy()

    assert np.allclose(lisi, lisi_test, rtol=1.e-4)

# def timereps(reps, func):
#     from time import time
#     start = time()
#     for i in range(0, reps):
#         func()
#     end = time()
#     return (end - start) / reps
#
# # 0.3 seconds per loop (too slow)
# timereps(10, lambda: hm.compute_lisi(X, metadata, label_colnames, perplexity))
#
#
# # Try https://github.com/rkern/line_profiler
