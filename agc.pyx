# distutils: language = c++
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
from libc.math cimport sqrt, fabs, fmax

cnp.import_array()

def trace_agc(cnp.ndarray[cnp.float32_t, ndim=3] data, int window_size=9):
    """
    TraceAGC: Automatic Gain Control using RMS amplitude normalization.
    
    Implements Petrel's TraceAGC algorithm:
    TraceAGC(T) = F(T) * (1.5 - RMS/max)
    
    where:
    - F(T) is the original input seismic data at time T
    - RMS is the root-mean-square amplitude computed over a sliding window
    - max is the maximum RMS value computed over the entire survey
    
    Based on scientific literature:
    - Standard AGC applies a gain scalar to the input trace (SEG Wiki, ScienceDirect)
    - RMS amplitude is computed as sqrt(mean(x^2)) over a sliding window
    - The gain factor modulates the original data based on local RMS relative to global max
    
    Parameters:
    -----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic data
    window_size : int, default=9
        Window size in samples for RMS computation (1-1000)
        
    Returns:
    --------
    output : ndarray of float32, shape (time, inline, crossline)
        TraceAGC processed seismic data
        
    References:
    -----------
    SEG Wiki: RMS amplitude AGC and Automatic Gain Control
    ScienceDirect Topics: Automatic Gain Control in seismic processing
    """
    cdef int num_samples = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    cdef int half_window = window_size // 2
    
    # Allocate arrays
    cdef cnp.ndarray[cnp.float32_t, ndim=3] rms_amplitude = np.empty_like(data)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] rms_view = rms_amplitude
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k, t, start_idx, end_idx, window_len, mirror_idx, bottom_idx
    cdef float inv_window_len, sum_sq, val, rms_val, max_rms, gain_factor
    cdef float weight, w_comp, rms_current, rms_mirror
    cdef double mean_rms, M2, delta, delta2
    cdef long long n
    cdef float abs_val, std_rms, threshold
    
    # Phase 1: Core RMS calculation (matching the perfect RMS code)
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
    mean_rms = 0.0
    M2 = 0.0
    n = 0
    
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
    
    std_rms = <float>sqrt(M2 / <double>n)
    threshold = <float>mean_rms + 5.0 * std_rms
    
    # Apply threshold
    for i in prange(num_samples, nogil=True, schedule='static'):
        for j in range(num_inline):
            for k in range(num_crossline):
                if rms_view[i, j, k] > threshold:
                    rms_view[i, j, k] = threshold
    
    # Phase 4: Find maximum RMS value across entire survey
    max_rms = 0.0
    for i in range(num_samples):
        for j in range(num_inline):
            for k in range(num_crossline):
                max_rms = fmax(max_rms, rms_view[i, j, k])
    
    # Phase 5: Apply TraceAGC formula: F(T) * (1.5 - RMS/max)
    if max_rms > 1e-8:
        for i in prange(num_samples, nogil=True, schedule='static'):
            for j in range(num_inline):
                for k in range(num_crossline):
                    gain_factor = 1.5 - (rms_view[i, j, k] / max_rms)
                    if gain_factor < 0.0:
                        gain_factor = 0.0
                    output_view[i, j, k] = data_view[i, j, k] * gain_factor
    else:
        for i in prange(num_samples, nogil=True, schedule='static'):
            for j in range(num_inline):
                for k in range(num_crossline):
                    output_view[i, j, k] = data_view[i, j, k] * 1.5
    
    return output