import os
import logging
import h5py

from typing import List, Set


__HARMONY_GPU = False
__HARMONY_DISTRIBUTED = False

# Path in h5 file
H5AD_DATA = '/X/data'
H5AD_INDEX = '/X/indices'
H5AD_INDPRT = '/X/indptr'
H5AD_GENES = '/var/_index'
H5AD_BARCODES = '/obs/_index'

BATCH_SIZE = 100000


try:
    if os.environ.get('HARMONYPY_CPU', '0') == '0':
        import cupy
        __HARMONY_GPU = True
except ModuleNotFoundError:
    __HARMONY_GPU = False
    cupy = None
    logging.info("GPU libraries are not available")


try:
    import dask
    __HARMONY_DISTRIBUTED = True
except ModuleNotFoundError:
    __HARMONY_DISTRIBUTED = False
    logging.info("Dask support is not available")


def is_gpu_available():
    return __HARMONY_GPU


def is_distributed_supported():
    return __HARMONY_DISTRIBUTED


def prep_distributed_inputs(input_files: Set[str],  vars_use: List[str]):
    """
    Prepares the input files for distributed processing
    :param input_files: List of input files
    :param vars_use: List of variables to use
    :return: List of tuples with input files and variables to use
    """
    if not is_distributed_supported():
        raise ValueError("Distributed processing is not supported")
    else:
        import dask.array as da
        import dask.dataframe as dd

    if len(input_files) != len(vars_use):
        raise ValueError("Number of input files and variables to use should be the same")

    if is_gpu_available():
        import cudf as _pd
        import cupy as _np
        from cupy.sparse import csr_matrix
    else:
        import pandas as _pd
        import numpy as _np
        from scipy.sparse import csr_matrix


    @dask.delayed
    def _read_batch(read_file,
                    batch_start,
                    batch_end,
                    total_cols):
        with h5py.File(read_file, 'r') as h5f:
            attrs = dict(h5f['X'].attrs)
            encoding = None
            if 'encoding-type' in attrs:
                encoding = attrs['encoding-type']

            if encoding == 'csr_matrix':
                indptrs = h5f[H5AD_INDPRT]
                start_ptr = indptrs[batch_start]
                end_ptr = indptrs[batch_end]

                # Read all things data and index
                sub_data = _np.array(h5f[H5AD_DATA][start_ptr:end_ptr])
                sub_indices = _np.array(h5f[H5AD_INDEX][start_ptr:end_ptr])

                # recompute the row pointer for the partial dataset
                sub_indptrs  = _np.array(indptrs[batch_start:(batch_end + 1)])
                sub_indptrs = sub_indptrs - sub_indptrs[0]

                sub_data = csr_matrix(
                    (sub_data, sub_indices, sub_indptrs),
                    shape=(batch_end - batch_start, total_cols))
                sub_data = sub_data.todense()
            elif attrs['encoding-type'] == 'array':
                sub_data = h5f["X"][batch_start:batch_end]
            else:
                logging.warning(f'Find file {read_file} contains dense X.')
                raise ValueError(f'Supports CSR and dense matrix. "{encoding}" found in {read_file}')
            return sub_data

    dls = []
    dfs_metadata = []

    #TODO: Can we parallelize this? These are independent reads.
    for input_file, batch_col in list(zip(input_files, vars_use)):
        with h5py.File(input_file, 'r') as h5f:
            assert batch_col in h5f[f'obs']
            cnt_genes = len(h5f[H5AD_GENES])
            cnt_cells = len(h5f[H5AD_INDPRT]) - 1
            # Compute metadata for computing phi in harmony
            # This involves reading categorical data from h5. Move to aseparate
            # function to address the differences between Pandas and CuDF.
            # TODO: Also try using Dask from_delayed like construct.
            attrs = dict(h5f[f'obs/{batch_col}'].attrs)
            batch_names = h5f[f'obs/{batch_col}/categories']
            batch_names = [c.decode() if isinstance(c, bytes) else c for c in batch_names]

            codes = h5f[f'obs/{batch_col}/codes']
            ordered = attrs.get("ordered", False)

            if is_gpu_available():
                dtype = _pd.CategoricalDtype(categories=batch_names, ordered=ordered)
                meta_series = _pd.Series.from_categorical(dtype, codes[:])
                metadata = _pd.DataFrame({batch_col: meta_series})
            else:
                raise NotImplementedError("Pandas support is not yet available")
            dfs_metadata.append(metadata)

        BATCH_SIZE
        for batch_start in range(0, cnt_cells, BATCH_SIZE):
            actual_batch_size = min(BATCH_SIZE, cnt_cells - batch_start)
            dls.append(da.from_delayed(
                (_read_batch) (input_file, batch_start, batch_start + actual_batch_size, cnt_genes),
                dtype=_np.float32,
                shape=(actual_batch_size, cnt_genes)))

    return da.concatenate(dls), dd.concat(dfs_metadata)