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
from libc.math cimport sin, cos, M_PI, sqrt, exp, fabs, log
from libc.stdlib cimport malloc, free
cimport openmp

cnp.import_array()


cdef void convolve_time_domain_optimized(float* trace, int num_time, 
                                          float* wavelet,
                                          int wavelet_samples,
                                          float* output) noexcept nogil:
    """
    Real convolution with branch elimination for bulk processing.
    """
    cdef int i, k, trace_idx
    cdef int half_len = wavelet_samples // 2
    cdef float sum_val
    
    # Edge region: start (requires bounds checking)
    for i in range(min(half_len, num_time)):
        sum_val = 0.0
        for k in range(wavelet_samples):
            trace_idx = i - half_len + k
            if trace_idx >= 0 and trace_idx < num_time:
                sum_val += trace[trace_idx] * wavelet[k]
        output[i] = sum_val
    
    # Bulk region: middle (no bounds checking needed - fully inside)
    for i in range(half_len, num_time - half_len):
        sum_val = 0.0
        for k in range(wavelet_samples):
            sum_val += trace[i - half_len + k] * wavelet[k]
        output[i] = sum_val
    
    # Edge region: end (requires bounds checking)
    for i in range(max(half_len, num_time - half_len), num_time):
        sum_val = 0.0
        for k in range(wavelet_samples):
            trace_idx = i - half_len + k
            if trace_idx >= 0 and trace_idx < num_time:
                sum_val += trace[trace_idx] * wavelet[k]
        output[i] = sum_val


def generalized_spectral_decomposition(cnp.ndarray[cnp.float32_t, ndim=3] data,
                                       float frequency=30.0,
                                       int num_cycles=2,
                                       float phase_deg=90.0):
    """
    Generalized Spectral Decomposition
    
    Extracts frequency content using phase-rotated Gabor wavelet convolution
    with L1-weighted normalization.
    
    Parameters:
    -----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic volume
    frequency : float (Hz), default=30.0
        Target frequency to extract
    num_cycles : int (1-15), default=2
        Number of wavelet cycles (controls frequency selectivity)
    phase_deg : float (-180 to 180), default=90.0
        Phase rotation in degrees (90° = sine phase, 0° = cosine phase)
        
    Returns:
    --------
    filtered : ndarray of float32
        Phase-rotated spectral decomposition at specified frequency
    """
    cdef int num_time = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] filtered = np.zeros_like(data)
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] filt_view = filtered
    
    # Construct wavelet - frequency-adaptive length
    # 250.0 is the "wavelet sampling rate" - independent of data sampling
    cdef float samples_per_cycle = 250.0 / frequency
    cdef int wavelet_samples = <int>(num_cycles * samples_per_cycle)
    
    # Ensure odd length for symmetric filter
    if wavelet_samples % 2 == 0:
        wavelet_samples += 1
    if wavelet_samples > num_time:
        wavelet_samples = num_time if num_time % 2 == 1 else num_time - 1
    if wavelet_samples < 3:
        wavelet_samples = 3
    
    cdef int half_len = wavelet_samples // 2
    cdef float omega = 2.0 * M_PI * frequency / 250.0
    cdef float sigma = <float>wavelet_samples / 4.0
    cdef float phase_rad = phase_deg * M_PI / 180.0
    
    cdef int i, j, k, w_idx
    cdef float n, sinusoid, gaussian_window
    cdef float normalization_factor = 1.0
    
    # L1-weighted normalization statistics
    cdef float wavelet_l1_sum = 0.0
    cdef float wavelet_weighted_l1 = 0.0
    cdef float distance_weight
    
    # Allocate wavelet (shared across all threads)
    cdef float* wavelet = <float*>malloc(wavelet_samples * sizeof(float))
    
    if wavelet == NULL:
        raise MemoryError("Failed to allocate wavelet buffer")
    
    # Build phase-rotated Gabor wavelet and compute statistics
    for w_idx in range(wavelet_samples):
        n = <float>(w_idx - half_len)
        gaussian_window = exp(-0.5 * (n / sigma) * (n / sigma))
        sinusoid = cos(omega * n + phase_rad)
        wavelet[w_idx] = sinusoid * gaussian_window
        
        # Weighted L1: weight by inverse distance from center
        distance_weight = exp(-0.1 * (n / sigma) * (n / sigma))
        wavelet_weighted_l1 += fabs(wavelet[w_idx]) * distance_weight
    
    # Apply L1-weighted normalization
    if wavelet_weighted_l1 > 0.0:
        normalization_factor = 1.0 / wavelet_weighted_l1
        for w_idx in range(wavelet_samples):
            wavelet[w_idx] *= normalization_factor
    
    # Allocate thread-local buffers (one pair per thread)
    cdef int num_threads = openmp.omp_get_max_threads()
    cdef float** trace_buffers = <float**>malloc(num_threads * sizeof(float*))
    cdef float** output_buffers = <float**>malloc(num_threads * sizeof(float*))
    
    if trace_buffers == NULL or output_buffers == NULL:
        free(wavelet)
        if trace_buffers != NULL:
            free(trace_buffers)
        if output_buffers != NULL:
            free(output_buffers)
        raise MemoryError("Failed to allocate thread buffers")
    
    # Initialize thread-local buffers
    for i in range(num_threads):
        trace_buffers[i] = <float*>malloc(num_time * sizeof(float))
        output_buffers[i] = <float*>malloc(num_time * sizeof(float))
        if trace_buffers[i] == NULL or output_buffers[i] == NULL:
            # Cleanup on failure
            for k in range(i + 1):
                if trace_buffers[k] != NULL:
                    free(trace_buffers[k])
                if output_buffers[k] != NULL:
                    free(output_buffers[k])
            free(trace_buffers)
            free(output_buffers)
            free(wavelet)
            raise MemoryError("Failed to allocate thread-local buffers")
    
    # Process all traces in parallel (thread-safe with local buffers)
    cdef int thread_id
    
    for j in prange(num_inline, nogil=True, schedule='static'):
        thread_id = openmp.omp_get_thread_num()
        
        for k in range(num_crossline):
            # Copy trace to thread-local buffer
            for i in range(num_time):
                trace_buffers[thread_id][i] = data_view[i, j, k]
            
            # Apply optimized phase-rotated Gabor filter
            convolve_time_domain_optimized(trace_buffers[thread_id], num_time, 
                                          wavelet, wavelet_samples, 
                                          output_buffers[thread_id])
            
            # Write filtered result
            for i in range(num_time):
                filt_view[i, j, k] = output_buffers[thread_id][i]
    
    # Cleanup thread-local buffers
    for i in range(num_threads):
        free(trace_buffers[i])
        free(output_buffers[i])
    free(trace_buffers)
    free(output_buffers)
    free(wavelet)
    
    return filtered