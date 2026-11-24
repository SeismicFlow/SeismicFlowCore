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
from libc.math cimport sqrt, fabs

cnp.import_array()

def rms_amplitude(cnp.ndarray[cnp.float32_t, ndim=3] data, int window_size=9):
    """
    Ultra-optimized RMS amplitude calculation.
    
    Parameters:
    -----------
    data : ndarray of float32, shape (num_samples, inline, crossline)
        Input seismic data - 3D array
    window_size : int, default=9
        Size of the sliding window for RMS calculation
        
    Returns:
    --------
    rms_amplitude : ndarray of float32, same shape as input
    """
    cdef int num_samples = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    cdef int half_window = window_size // 2
    
    # Allocate output array
    cdef cnp.ndarray[cnp.float32_t, ndim=3] rms_amplitude = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] rms_view = rms_amplitude
    
    cdef int i, j, k, t, start_idx, end_idx, window_len
    cdef float inv_window_len, sum_sq, val
    
    # Phase 1: Core RMS calculation - parallelize over inline
    for i in range(num_samples):
        start_idx = 0 if i < half_window else i - half_window
        end_idx = num_samples if i + half_window + 1 > num_samples else i + half_window + 1
        window_len = end_idx - start_idx
        inv_window_len = 1.0 / <float>window_len
        
        for j in prange(num_inline, nogil=True, schedule='guided'):
            for k in range(num_crossline):
                sum_sq = 0.0
                
                # Unrolled loop for SIMD
                t = start_idx
                while t + 3 < end_idx:
                    val = data_view[t, j, k]
                    sum_sq = sum_sq + val * val
                    val = data_view[t + 1, j, k]
                    sum_sq = sum_sq + val * val
                    val = data_view[t + 2, j, k]
                    sum_sq = sum_sq + val * val
                    val = data_view[t + 3, j, k]
                    sum_sq = sum_sq + val * val
                    t = t + 4
                
                while t < end_idx:
                    val = data_view[t, j, k]
                    sum_sq = sum_sq + val * val
                    t = t + 1
                
                rms_view[i, j, k] = sqrt(sum_sq * inv_window_len)
    
    # Phase 2: Edge mirroring - top edge
    cdef float weight, w_comp, rms_current, rms_mirror
    cdef int mirror_idx
    
    for i in range(half_window):
        mirror_idx = 2 * half_window - i
        if mirror_idx < num_samples:
            weight = (<float>(i + 1)) / (<float>(half_window + 1))
            w_comp = 1.0 - weight
            
            for j in prange(num_inline, nogil=True, schedule='static'):
                for k in range(num_crossline):
                    rms_current = rms_view[i, j, k]
                    rms_mirror = rms_view[mirror_idx, j, k]
                    rms_view[i, j, k] = weight * rms_current + w_comp * rms_mirror
    
    # Phase 2b: Edge mirroring - bottom edge
    cdef int bottom_idx
    
    for i in range(half_window):
        mirror_idx = num_samples - 2 * half_window + i - 1
        bottom_idx = num_samples - i - 1
        
        if mirror_idx >= 0 and mirror_idx < num_samples and bottom_idx >= 0:
            weight = (<float>(i + 1)) / (<float>(half_window + 1))
            w_comp = 1.0 - weight
            
            for j in prange(num_inline, nogil=True, schedule='static'):
                for k in range(num_crossline):
                    rms_current = rms_view[bottom_idx, j, k]
                    rms_mirror = rms_view[mirror_idx, j, k]
                    rms_view[bottom_idx, j, k] = weight * rms_current + w_comp * rms_mirror
    
    # Phase 3: Post-processing with Welford's algorithm
    cdef double mean_rms = 0.0
    cdef double M2 = 0.0
    cdef double delta, delta2
    cdef long long n = 0
    cdef float abs_val
    
    for i in range(num_samples):
        for j in range(num_inline):
            for k in range(num_crossline):
                abs_val = fabs(rms_view[i, j, k])
                rms_view[i, j, k] = abs_val
                
                n = n + 1
                delta = <double>abs_val - mean_rms
                mean_rms = mean_rms + delta / <double>n
                delta2 = <double>abs_val - mean_rms
                M2 = M2 + delta * delta2
    
    cdef float std_rms = <float>sqrt(M2 / <double>n)
    cdef float threshold = <float>mean_rms + 5.0 * std_rms
    
    # Apply threshold
    for i in prange(num_samples, nogil=True, schedule='static'):
        for j in range(num_inline):
            for k in range(num_crossline):
                if rms_view[i, j, k] > threshold:
                    rms_view[i, j, k] = threshold
    
    return rms_amplitude