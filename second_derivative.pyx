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
cnp.import_array()

def second_derivative(cnp.ndarray[cnp.float32_t, ndim=3] data):
    """
    Optimized second derivative computation using central difference.
    
    Computes f''(x) ≈ [f(x+1) - 2f(x) + f(x-1)] along the time axis.
    
    Boundary handling:
    - First time slice (i=0): forward difference f''(0) = f(2) - 2f(1) + f(0)
    - Interior slices: central difference f''(i) = f(i+1) - 2f(i) + f(i-1)
    - Last time slice (i=n-1): backward difference f''(n-1) = f(n-1) - 2f(n-2) + f(n-3)
    """
    cdef:
        int num_time = data.shape[0]
        int num_inline = data.shape[1]
        int num_crossline = data.shape[2]
        cnp.ndarray[cnp.float32_t, ndim=3] output = np.zeros_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k
    
    with nogil:
        # Parallelize over inlines
        for j in prange(num_inline, schedule='static'):
            # First time slice: forward difference
            for k in range(num_crossline):
                output_view[0, j, k] = (
                    data_view[2, j, k] - 
                    2.0 * data_view[1, j, k] + 
                    data_view[0, j, k]
                )
            
            # Interior time slices: central difference
            for i in range(1, num_time - 1):
                for k in range(num_crossline):
                    output_view[i, j, k] = (
                        data_view[i + 1, j, k] - 
                        2.0 * data_view[i, j, k] + 
                        data_view[i - 1, j, k]
                    )
            
            # Last time slice: backward difference
            for k in range(num_crossline):
                output_view[num_time - 1, j, k] = (
                    data_view[num_time - 1, j, k] - 
                    2.0 * data_view[num_time - 2, j, k] + 
                    data_view[num_time - 3, j, k]
                )
    
    return output