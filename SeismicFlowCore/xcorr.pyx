# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
cimport cython
from libc.math cimport sqrt

# FFTW interface
cdef extern from "fftw3.h":
    ctypedef struct fftwf_plan_s:
        pass
    ctypedef fftwf_plan_s* fftwf_plan
    
    ctypedef float fftwf_complex[2]
    
    cdef enum:
        FFTW_FORWARD = -1
        FFTW_BACKWARD = 1
        FFTW_ESTIMATE = 1 << 6
    
    void* fftwf_malloc(size_t n) nogil
    void fftwf_free(void* p) nogil
    
    fftwf_plan fftwf_plan_dft_r2c_1d(int n, float* input, fftwf_complex* output, unsigned flags) nogil
    fftwf_plan fftwf_plan_dft_c2r_1d(int n, fftwf_complex* input, float* output, unsigned flags) nogil
    
    void fftwf_execute(fftwf_plan plan) nogil
    void fftwf_execute_dft_r2c(fftwf_plan plan, float* input, fftwf_complex* output) nogil
    void fftwf_execute_dft_c2r(fftwf_plan plan, fftwf_complex* input, float* output) nogil
    void fftwf_destroy_plan(fftwf_plan plan) nogil

ctypedef cnp.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cross_correlation(cnp.ndarray[DTYPE_t, ndim=3] data1,
                        cnp.ndarray[DTYPE_t, ndim=3] data2,
                        int max_lag=100,
                        float sample_rate=4.0):
    """
    FFT-accelerated normalized cross-correlation using sum tables (Lewis's algorithm).
    
    This achieves IDENTICAL results to time-domain version but with O(N log N) complexity.
    
    Algorithm:
    1. Use FFT to compute raw cross-correlation numerator - O(N log N)
    2. Use cumulative sum tables for fast mean/variance computation - O(1) per lag
    
    Parameters
    ----------
    data1 : ndarray, shape (n_samples, n_inlines, n_crosslines), dtype=float32
        First input seismic cube. Time is first dimension.
    data2 : ndarray, shape (n_samples, n_inlines, n_crosslines), dtype=float32
        Second input seismic cube. Time is first dimension.
    max_lag : int, optional
        Maximum lag in MILLISECONDS. Default is 100 ms.
    sample_rate : float, optional
        Sampling interval in milliseconds. Default is 4.0 ms.
        
    Returns
    -------
    result : ndarray, shape (n_samples, n_inlines, n_crosslines), dtype=float32
        Normalized cross-correlation result, zero-padded to match input size.
    """
    
    # Convert max_lag from milliseconds to samples
    cdef int max_lag_samples = int(max_lag / sample_rate + 0.5)
    
    cdef int n_samples = data1.shape[0]
    cdef int n_inlines = data1.shape[1]
    cdef int n_crosslines = data1.shape[2]
    cdef int output_len = 2 * max_lag_samples + 1
    cdef int total_padding = n_samples - output_len
    cdef int top_pad = total_padding // 2
    cdef int i, j, k, lag_idx, lag, t, valid_start, valid_end, n_valid
    cdef DTYPE_t mean1, mean2, sum_sq1, sum_sq2, denom
    
    # Validate inputs
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(f"data1 and data2 must have same n_samples")
    if data1.shape[1] != data2.shape[1]:
        raise ValueError(f"data1 and data2 must have same n_inlines")
    if data1.shape[2] != data2.shape[2]:
        raise ValueError(f"data1 and data2 must have same n_crosslines")
    if 2 * max_lag_samples >= n_samples:
        raise ValueError(f"Total displacement cannot exceed vertical length")
    
    # FFT size: must be large enough to avoid circular wraparound
    cdef int fft_size = 2 * n_samples
    cdef int fft_complex_size = fft_size // 2 + 1
    
    # Allocate output array
    cdef cnp.ndarray[DTYPE_t, ndim=3] result = np.zeros(
        (n_samples, n_inlines, n_crosslines), dtype=np.float32)
    
    cdef DTYPE_t[:, :, :] data1_view = data1
    cdef DTYPE_t[:, :, :] data2_view = data2
    cdef DTYPE_t[:, :, :] result_view = result
    
    # Declare all variables used in parallel region
    cdef float* data1_padded
    cdef float* data2_padded
    cdef fftwf_complex* fft1
    cdef fftwf_complex* fft2
    cdef float* xcorr_raw
    cdef float* sum_table1  # Cumulative sum for data1
    cdef float* sum_sq_table1  # Cumulative sum of squares for data1
    cdef float* sum_table2  # Cumulative sum for data2
    cdef float* sum_sq_table2  # Cumulative sum of squares for data2
    cdef float real1, real2, imag1, imag2, scale, raw_sum, numerator
    cdef float sum1, sum2, sumsq1, sumsq2
    cdef int fft_idx
    
    # Create FFTW plans (outside parallel region, with GIL)
    cdef float* temp_real = <float*>fftwf_malloc(fft_size * sizeof(float))
    cdef fftwf_complex* temp_freq = <fftwf_complex*>fftwf_malloc(
        fft_complex_size * sizeof(fftwf_complex))
    
    if temp_real == NULL or temp_freq == NULL:
        if temp_real != NULL:
            fftwf_free(temp_real)
        if temp_freq != NULL:
            fftwf_free(temp_freq)
        raise MemoryError("Failed to allocate FFTW buffers")
    
    cdef fftwf_plan plan_forward = fftwf_plan_dft_r2c_1d(
        fft_size, temp_real, temp_freq, FFTW_ESTIMATE)
    cdef fftwf_plan plan_backward = fftwf_plan_dft_c2r_1d(
        fft_size, temp_freq, temp_real, FFTW_ESTIMATE)
    
    if plan_forward == NULL or plan_backward == NULL:
        if plan_forward != NULL:
            fftwf_destroy_plan(plan_forward)
        if plan_backward != NULL:
            fftwf_destroy_plan(plan_backward)
        fftwf_free(temp_real)
        fftwf_free(temp_freq)
        raise RuntimeError("Failed to create FFTW plans")
    
    fftwf_free(temp_real)
    fftwf_free(temp_freq)
    
    # Process in parallel over inlines
    for i in prange(n_inlines, nogil=True, schedule='static'):
        # Allocate thread-local buffers
        data1_padded = <float*>fftwf_malloc(fft_size * sizeof(float))
        data2_padded = <float*>fftwf_malloc(fft_size * sizeof(float))
        fft1 = <fftwf_complex*>fftwf_malloc(fft_complex_size * sizeof(fftwf_complex))
        fft2 = <fftwf_complex*>fftwf_malloc(fft_complex_size * sizeof(fftwf_complex))
        xcorr_raw = <float*>fftwf_malloc(fft_size * sizeof(float))
        
        # Allocate sum tables (size n_samples + 1 for easier indexing)
        sum_table1 = <float*>fftwf_malloc((n_samples + 1) * sizeof(float))
        sum_sq_table1 = <float*>fftwf_malloc((n_samples + 1) * sizeof(float))
        sum_table2 = <float*>fftwf_malloc((n_samples + 1) * sizeof(float))
        sum_sq_table2 = <float*>fftwf_malloc((n_samples + 1) * sizeof(float))
        
        if (data1_padded == NULL or data2_padded == NULL or 
            fft1 == NULL or fft2 == NULL or xcorr_raw == NULL or
            sum_table1 == NULL or sum_sq_table1 == NULL or
            sum_table2 == NULL or sum_sq_table2 == NULL):
            if data1_padded != NULL:
                fftwf_free(data1_padded)
            if data2_padded != NULL:
                fftwf_free(data2_padded)
            if fft1 != NULL:
                fftwf_free(fft1)
            if fft2 != NULL:
                fftwf_free(fft2)
            if xcorr_raw != NULL:
                fftwf_free(xcorr_raw)
            if sum_table1 != NULL:
                fftwf_free(sum_table1)
            if sum_sq_table1 != NULL:
                fftwf_free(sum_sq_table1)
            if sum_table2 != NULL:
                fftwf_free(sum_table2)
            if sum_sq_table2 != NULL:
                fftwf_free(sum_sq_table2)
            continue
        
        # Process each crossline
        for j in range(n_crosslines):
            # Build cumulative sum tables for data1 and data2
            # sum_table[k] = sum of trace[0:k]
            sum_table1[0] = 0.0
            sum_sq_table1[0] = 0.0
            sum_table2[0] = 0.0
            sum_sq_table2[0] = 0.0
            
            for k in range(n_samples):
                sum_table1[k + 1] = sum_table1[k] + data1_view[k, i, j]
                sum_sq_table1[k + 1] = sum_sq_table1[k] + data1_view[k, i, j] * data1_view[k, i, j]
                sum_table2[k + 1] = sum_table2[k] + data2_view[k, i, j]
                sum_sq_table2[k + 1] = sum_sq_table2[k] + data2_view[k, i, j] * data2_view[k, i, j]
            
            # Zero-pad traces for FFT
            for k in range(n_samples):
                data1_padded[k] = data1_view[k, i, j]
                data2_padded[k] = data2_view[k, i, j]
            for k in range(n_samples, fft_size):
                data1_padded[k] = 0.0
                data2_padded[k] = 0.0
            
            # FFT both traces
            fftwf_execute_dft_r2c(plan_forward, data1_padded, fft1)
            fftwf_execute_dft_r2c(plan_forward, data2_padded, fft2)
            
            # Cross-correlation in frequency domain: conj(FFT1) * FFT2
            for k in range(fft_complex_size):
                real1 = fft1[k][0]
                imag1 = fft1[k][1]
                real2 = fft2[k][0]
                imag2 = fft2[k][1]
                
                # Complex conjugate of fft1 times fft2
                fft1[k][0] = real1 * real2 + imag1 * imag2
                fft1[k][1] = real1 * imag2 - imag1 * real2
            
            # Inverse FFT to get raw cross-correlation
            fftwf_execute_dft_c2r(plan_backward, fft1, xcorr_raw)
            
            # Normalize by FFT size (FFTW doesn't do this automatically)
            scale = 1.0 / <float>fft_size
            for k in range(fft_size):
                xcorr_raw[k] *= scale
            
            # Now compute normalized cross-correlation for each lag using sum tables
            for lag_idx in range(output_len):
                lag = lag_idx - max_lag_samples
                
                # Determine valid sample range for this lag
                if lag >= 0:
                    valid_start = 0
                    valid_end = n_samples - lag
                else:
                    valid_start = -lag
                    valid_end = n_samples
                
                n_valid = valid_end - valid_start
                
                # Use sum tables to compute means in O(1) time
                # sum(data1[valid_start:valid_end])
                sum1 = sum_table1[valid_end] - sum_table1[valid_start]
                mean1 = sum1 / n_valid
                
                # sum(data2[valid_start+lag:valid_end+lag])
                sum2 = sum_table2[valid_end + lag] - sum_table2[valid_start + lag]
                mean2 = sum2 / n_valid
                
                # Use sum tables to compute sum of squares in O(1) time
                sumsq1 = sum_sq_table1[valid_end] - sum_sq_table1[valid_start]
                sumsq2 = sum_sq_table2[valid_end + lag] - sum_sq_table2[valid_start + lag]
                
                # Variance = E[X²] - E[X]²
                # sum_sq = sum((x - mean)²) = sumsq - n*mean²
                sum_sq1 = sumsq1 - n_valid * mean1 * mean1
                sum_sq2 = sumsq2 - n_valid * mean2 * mean2
                
                # Get raw cross-correlation value from FFT result
                if lag >= 0:
                    fft_idx = lag
                else:
                    fft_idx = fft_size + lag
                
                # Adjust for mean-centering
                raw_sum = xcorr_raw[fft_idx]
                numerator = raw_sum - n_valid * mean1 * mean2
                
                # Normalize (Pearson coefficient)
                if sum_sq1 > 0.0 and sum_sq2 > 0.0:
                    result_view[lag_idx + top_pad, i, j] = numerator / sqrt(sum_sq1 * sum_sq2)
                else:
                    result_view[lag_idx + top_pad, i, j] = 0.0
        
        # Free thread-local buffers
        fftwf_free(data1_padded)
        fftwf_free(data2_padded)
        fftwf_free(fft1)
        fftwf_free(fft2)
        fftwf_free(xcorr_raw)
        fftwf_free(sum_table1)
        fftwf_free(sum_sq_table1)
        fftwf_free(sum_table2)
        fftwf_free(sum_sq_table2)
    
    # Cleanup plans
    fftwf_destroy_plan(plan_forward)
    fftwf_destroy_plan(plan_backward)
    
    return result