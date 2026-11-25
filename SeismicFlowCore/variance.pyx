# xb570.pyx - Ultra-optimized Variance Edge (Maximum Performance)
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fabs

# Initialize numpy C-API
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def variance_edge_cython(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    int inline_range=3,
    int crossline_range=3,
    int vertical_smoothing=15
):
    """
    Ultra-optimized variance edge detection on 3D seismic data.
    
    Optimizations:
    - OpenMP parallel processing
    - SIMD-friendly memory layout
    - Cache-optimized data access patterns
    - Reduced redundant computations
    """
    cdef:
        cnp.intp_t n_time = data.shape[0]
        cnp.intp_t n_inline = data.shape[1]
        cnp.intp_t n_crossline = data.shape[2]
        
        int inline_half = inline_range // 2
        int crossline_half = crossline_range // 2
        int vertical_half = vertical_smoothing // 2
        int max_time_samples = vertical_smoothing
        
        # Pre-allocate output
        cnp.ndarray[cnp.float32_t, ndim=3] result = np.empty(
            (n_time, n_inline, n_crossline), dtype=np.float32
        )
        
        # Pre-compute normalized weights (shared, read-only)
        cnp.ndarray[cnp.float32_t, ndim=2] normalized_weights = np.zeros(
            (n_time, vertical_smoothing), dtype=np.float32
        )
        
        # Precompute Bartlett window once
        cnp.ndarray[cnp.float32_t, ndim=1] bartlett = np.bartlett(vertical_smoothing).astype(np.float32)
        float* bartlett_ptr = <float*> cnp.PyArray_DATA(bartlett)
        
        # Direct memory pointers
        float* data_ptr = <float*> cnp.PyArray_DATA(data)
        float* result_ptr = <float*> cnp.PyArray_DATA(result)
        float* weights_ptr = <float*> cnp.PyArray_DATA(normalized_weights)
        
        # Precompute strides for faster indexing
        cnp.intp_t stride_time = n_inline * n_crossline
        cnp.intp_t stride_inline = n_crossline
        
        # Variables for weight computation
        cnp.intp_t t, i
        cnp.intp_t weight_start
        float weight_sum, inv_weight_sum
        cnp.intp_t t_start, t_end, n_samples
    
    # Pre-compute normalized weights (serial, done once)
    for t in range(n_time):
        t_start = max(<cnp.intp_t>0, t - vertical_half)
        t_end = min(n_time, t + vertical_half + 1)
        n_samples = t_end - t_start
        
        weight_start = vertical_half - <int>(t - t_start)
        
        # Sum weights
        weight_sum = 0.0
        for i in range(n_samples):
            weight_sum += bartlett_ptr[weight_start + i]
        
        # Normalize (use inverse for faster multiplication)
        inv_weight_sum = 1.0 / weight_sum if weight_sum > 1e-10 else 0.0
        for i in range(n_samples):
            weights_ptr[t * vertical_smoothing + i] = bartlett_ptr[weight_start + i] * inv_weight_sum
    
    # Main parallel computation
    _compute_variance_edge_parallel(
        data_ptr, result_ptr, weights_ptr,
        n_time, n_inline, n_crossline,
        stride_time, stride_inline,
        inline_half, crossline_half, vertical_half,
        max_time_samples
    )
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef void _compute_variance_edge_parallel(
    float* data_ptr,
    float* result_ptr,
    float* weights_ptr,
    cnp.intp_t n_time,
    cnp.intp_t n_inline,
    cnp.intp_t n_crossline,
    cnp.intp_t stride_time,
    cnp.intp_t stride_inline,
    int inline_half,
    int crossline_half,
    int vertical_half,
    int max_time_samples
) noexcept nogil:
    """Core computation kernel - fully optimized and parallelized."""
    
    cdef:
        cnp.intp_t il, xl, t, i, j, k
        cnp.intp_t il_start, il_end, xl_start, xl_end
        cnp.intp_t t_start, t_end, n_samples
        cnp.intp_t base_idx, idx, output_idx
        int wi, n_traces
        
        # Thread-local arrays
        float* trace_sum
        float* trace_sq_sum
        
        # Computation variables
        float val, numerator, denominator, semblance, variance_edge
        float max_abs, weight, sum_val, sq_sum_val
        float inv_n_traces
    
    # Parallel loop over inlines (coarse-grained parallelism)
    for il in prange(n_inline, schedule='dynamic'):
        # Allocate memory for this thread
        trace_sum = <float*> malloc(max_time_samples * sizeof(float))
        trace_sq_sum = <float*> malloc(max_time_samples * sizeof(float))
        
        if trace_sum == NULL or trace_sq_sum == NULL:
            if trace_sum != NULL:
                free(trace_sum)
            if trace_sq_sum != NULL:
                free(trace_sq_sum)
            continue
        
        # Compute inline window bounds once
        il_start = max(<cnp.intp_t>0, il - inline_half)
        il_end = min(n_inline, il + inline_half + 1)
        
        # Loop over crosslines
        for xl in range(n_crossline):
            # Compute crossline window bounds
            xl_start = max(<cnp.intp_t>0, xl - crossline_half)
            xl_end = min(n_crossline, xl + crossline_half + 1)
            
            n_traces = <int>((il_end - il_start) * (xl_end - xl_start))
            inv_n_traces = 1.0 / <float>n_traces if n_traces > 0 else 0.0
            
            # Loop over time samples
            for t in range(n_time):
                # Compute time window bounds
                t_start = max(<cnp.intp_t>0, t - vertical_half)
                t_end = min(n_time, t + vertical_half + 1)
                n_samples = t_end - t_start
                
                # Fast reset using memset
                memset(trace_sum, 0, n_samples * sizeof(float))
                memset(trace_sq_sum, 0, n_samples * sizeof(float))
                
                max_abs = 0.0
                
                # Accumulate traces - SIMD-optimizable innermost loops
                for k in range(t_start, t_end):
                    wi = <int>(k - t_start)
                    
                    for i in range(il_start, il_end):
                        # Compute base index once per inline
                        base_idx = k * stride_time + i * stride_inline + xl_start
                        
                        # Innermost loop - compiler auto-vectorizes this
                        for j in range(xl_start, xl_end):
                            idx = base_idx + (j - xl_start)
                            val = data_ptr[idx]
                            
                            # Track max absolute value using fabs
                            if fabs(val) > max_abs:
                                max_abs = fabs(val)
                            
                            # Accumulate sums (SIMD-friendly)
                            trace_sum[wi] = trace_sum[wi] + val
                            trace_sq_sum[wi] = trace_sq_sum[wi] + val * val
                
                # Compute output index once
                output_idx = t * stride_time + il * stride_inline + xl
                
                # Check for empty region
                if max_abs < 1e-10:
                    # Empty region: variance_edge = 0
                    result_ptr[output_idx] = 0.0
                else:
                    # Compute weighted semblance
                    numerator = 0.0
                    denominator = 0.0
                    
                    # Weighted accumulation (SIMD-friendly)
                    for wi in range(<int>n_samples):
                        weight = weights_ptr[t * max_time_samples + wi]
                        sum_val = trace_sum[wi]
                        sq_sum_val = trace_sq_sum[wi]
                        
                        numerator = numerator + weight * sum_val * sum_val
                        denominator = denominator + weight * sq_sum_val
                    
                    # Multiply denominator by n_traces
                    denominator = denominator * <float>n_traces
                    
                    # Compute semblance with bounds checking
                    if denominator > 1e-10:
                        semblance = numerator / denominator
                        
                        # Clamp semblance to [0, 1]
                        if semblance > 1.0:
                            semblance = 1.0
                        elif semblance < 0.0:
                            semblance = 0.0
                    else:
                        semblance = 0.0
                    
                    # Convert to variance edge: 1 - coherence
                    variance_edge = 1.0 - semblance
                    
                    # Clamp final result to [0, 1]
                    if variance_edge < 0.0:
                        variance_edge = 0.0
                    elif variance_edge > 1.0:
                        variance_edge = 1.0
                    
                    result_ptr[output_idx] = variance_edge
        
        # Free thread-local memory
        free(trace_sum)
        free(trace_sq_sum)