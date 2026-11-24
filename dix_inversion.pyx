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

def stacking_to_interval_velocity(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    float sampling_interval_seconds
):
    """
    Convert stacking (RMS) velocity to interval velocity using Dix equation.
    """
    cdef:
        int num_time = data.shape[0]
        int num_inline = data.shape[1]
        int num_crossline = data.shape[2]
        cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k
    cdef float t1, t2, vrms1, vrms2, vrms1_sq, vrms2_sq, numerator
    cdef float dt = sampling_interval_seconds
    cdef float inv_dt = 1.0 / dt
    
    with nogil:
        for j in prange(num_inline, schedule='static'):
            for k in range(num_crossline):
                output_view[0, j, k] = data_view[0, j, k]
        
        for i in range(1, num_time):
            t1 = i * dt
            t2 = t1 + dt
            
            for j in prange(num_inline, schedule='static'):
                for k in range(num_crossline):
                    vrms2 = data_view[i, j, k]
                    vrms1 = data_view[i - 1, j, k]
                    
                    vrms2_sq = vrms2 * vrms2
                    vrms1_sq = vrms1 * vrms1
                    
                    numerator = (vrms2_sq * t2) - (vrms1_sq * t1)
                    
                    if numerator > 0.0:
                        output_view[i, j, k] = sqrt(numerator * inv_dt)
                    else:
                        output_view[i, j, k] = output_view[i - 1, j, k]
    
    return output