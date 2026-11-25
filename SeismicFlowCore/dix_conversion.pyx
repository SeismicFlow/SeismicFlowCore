# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: infer_types=True

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt
cnp.import_array()

def interval_to_stacking_velocity(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    float sampling_interval_seconds
):
    """
    Convert interval velocity to stacking velocity using Dix equation.
    """
    cdef:
        int num_time = data.shape[0]
        int num_inline = data.shape[1]
        int num_crossline = data.shape[2]
        cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k
    cdef float v_val, cumsum, inv_time
    cdef float dt = sampling_interval_seconds
    
    with nogil:
        for j in prange(num_inline, schedule='static'):
            for k in range(num_crossline):
                cumsum = 0.0
                for i in range(num_time):
                    v_val = data_view[i, j, k]
                    cumsum = cumsum + v_val * v_val * dt
                    inv_time = 1.0 / ((i + 1) * dt)
                    output_view[i, j, k] = sqrt(cumsum * inv_time)
    
    return output