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
from libc.stdlib cimport malloc, free
from libc.math cimport fmax
from openmp cimport omp_get_thread_num, omp_get_max_threads

cnp.import_array()

cdef inline float linear_interp(float x, float* xp, float* fp, 
                                int n, float left_val, float right_val) noexcept nogil:
    """Fast linear interpolation with binary search"""
    cdef int i, low, high, mid
    cdef float x0, x1, y0, y1
    
    if x <= xp[0]:
        return left_val
    if x >= xp[n-1]:
        return right_val
    
    # Binary search
    low = 0
    high = n - 1
    
    while low <= high:
        mid = (low + high) // 2
        if xp[mid] == x:
            return fp[mid]
        elif xp[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    
    # Linear interpolation between high and low
    if high < 0:
        high = 0
    if low >= n:
        low = n - 1
    
    x0 = xp[high]
    x1 = xp[low]
    y0 = fp[high]
    y1 = fp[low]
    
    if x1 == x0:
        return y0
    
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

cdef inline int make_monotonic(float* depth_trace, float* amp_trace,
                                float* out_depth, float* out_amp,
                                int n) noexcept nogil:
    """Make depth values strictly increasing and remove duplicates"""
    cdef int i, j = 0
    cdef float eps = 1e-7
    
    if n == 0:
        return 0
    
    # First pass: make strictly increasing
    for i in range(1, n):
        if depth_trace[i] <= depth_trace[i-1]:
            depth_trace[i] = depth_trace[i-1] + eps
    
    # Second pass: remove duplicates
    out_depth[0] = depth_trace[0]
    out_amp[0] = amp_trace[0]
    j = 1
    
    for i in range(1, n):
        if depth_trace[i] != out_depth[j-1]:
            out_depth[j] = depth_trace[i]
            out_amp[j] = amp_trace[i]
            j += 1
    
    return j

def time_to_depth_conversion(
    cnp.ndarray[cnp.float32_t, ndim=3] velocity_cube,
    float dt_sec,
    cnp.ndarray[cnp.float32_t, ndim=3] seismic_time
):
    """
    Convert seismic data from time to depth domain.
    
    Parameters:
    -----------
    velocity_cube : 3D array (nt, ni, nx) - float32
        Velocity model in time domain
    dt_sec : float
        Time sampling interval in seconds
    seismic_time : 3D array (nt, ni, nx) - float32
        Seismic amplitude data in time domain
        
    Returns:
    --------
    result : 3D array (nt, ni, nx) - float32
        Seismic data in depth domain
    dz : float
        Depth sampling interval
    """
    cdef:
        int nt = velocity_cube.shape[0]
        int ni = velocity_cube.shape[1]
        int nx = velocity_cube.shape[2]
        # ONLY ONE depth grid - we reuse it!
        cnp.ndarray[cnp.float32_t, ndim=3] depth_grid = np.zeros((nt, ni, nx), dtype=np.float32)
        cnp.ndarray[cnp.float32_t, ndim=3] result = np.zeros((nt, ni, nx), dtype=np.float32)
        cnp.ndarray[cnp.float32_t, ndim=1] depth_output_axis = np.zeros(nt, dtype=np.float32)
        
    cdef cnp.float32_t[:, :, :] vel_view = velocity_cube
    cdef cnp.float32_t[:, :, :] seis_view = seismic_time
    cdef cnp.float32_t[:, :, :] depth_view = depth_grid
    cdef cnp.float32_t[:, :, :] result_view = result
    cdef cnp.float32_t[:] depth_axis_view = depth_output_axis
    
    cdef int i, j, k, thread_id, unique_size, max_threads
    cdef float half_sample_corr, cumsum_val, max_depth, dz
    cdef float factor = dt_sec / 2.0
    
    # Allocate working arrays - only for actual number of threads, not ni!
    cdef float** depth_trace_arr
    cdef float** amp_trace_arr
    cdef float** unique_depth_arr
    cdef float** unique_amp_arr
    
    # Step 1: Calculate Left Riemann depth grid and find max_depth
    with nogil:
        # Calculate Left Riemann sum: depth[1:] = cumsum(velocity[:-1] * dt/2)
        for j in prange(ni, schedule='static'):
            for k in range(nx):
                # Calculate half-sample correction for this trace
                half_sample_corr = vel_view[0, j, k] * factor * 0.5
                
                # First sample stays zero
                depth_view[0, j, k] = 0.0
                
                # Left Riemann: use velocity[i-1]
                cumsum_val = 0.0
                for i in range(1, nt):
                    cumsum_val = cumsum_val + vel_view[i-1, j, k] * factor
                    depth_view[i, j, k] = cumsum_val
                
                # Apply half-sample correction and clip
                for i in range(nt):
                    depth_view[i, j, k] = fmax(depth_view[i, j, k] - half_sample_corr, 0.0)
    
    # Find max_depth
    max_depth = 0.0
    for i in range(nt):
        for j in range(ni):
            for k in range(nx):
                if depth_grid[i, j, k] > max_depth:
                    max_depth = depth_grid[i, j, k]
    
    # Step 2: Create output depth axis
    if nt > 1:
        dz = max_depth / (nt - 1)
    else:
        dz = 0.0
    
    for i in range(nt):
        depth_output_axis[i] = i * dz
    
    # Step 3: REUSE depth_grid - overwrite with Right Riemann for interpolation
    with nogil:
        for j in prange(ni, schedule='static'):
            for k in range(nx):
                # Calculate half-sample correction (same as before)
                half_sample_corr = vel_view[0, j, k] * factor * 0.5
                
                # Right Riemann: use velocity[i]
                cumsum_val = 0.0
                for i in range(nt):
                    cumsum_val = cumsum_val + vel_view[i, j, k] * factor
                    depth_view[i, j, k] = cumsum_val
                
                # Apply half-sample correction and clip
                for i in range(nt):
                    depth_view[i, j, k] = fmax(depth_view[i, j, k] - half_sample_corr, 0.0)
    
    # Step 4: Allocate per-thread working arrays (NOT per inline!)
    max_threads = 1
    with nogil:
        max_threads = omp_get_max_threads()
    
    depth_trace_arr = <float**>malloc(max_threads * sizeof(float*))
    amp_trace_arr = <float**>malloc(max_threads * sizeof(float*))
    unique_depth_arr = <float**>malloc(max_threads * sizeof(float*))
    unique_amp_arr = <float**>malloc(max_threads * sizeof(float*))
    
    if not (depth_trace_arr and amp_trace_arr and unique_depth_arr and unique_amp_arr):
        if depth_trace_arr: free(depth_trace_arr)
        if amp_trace_arr: free(amp_trace_arr)
        if unique_depth_arr: free(unique_depth_arr)
        if unique_amp_arr: free(unique_amp_arr)
        raise MemoryError("Could not allocate memory for thread arrays")
    
    for i in range(max_threads):
        depth_trace_arr[i] = <float*>malloc(nt * sizeof(float))
        amp_trace_arr[i] = <float*>malloc(nt * sizeof(float))
        unique_depth_arr[i] = <float*>malloc(nt * sizeof(float))
        unique_amp_arr[i] = <float*>malloc(nt * sizeof(float))
        
        if not (depth_trace_arr[i] and amp_trace_arr[i] and unique_depth_arr[i] and unique_amp_arr[i]):
            # Clean up on failure
            for thread_id in range(i + 1):
                if depth_trace_arr[thread_id]: free(depth_trace_arr[thread_id])
                if amp_trace_arr[thread_id]: free(amp_trace_arr[thread_id])
                if unique_depth_arr[thread_id]: free(unique_depth_arr[thread_id])
                if unique_amp_arr[thread_id]: free(unique_amp_arr[thread_id])
            free(depth_trace_arr)
            free(amp_trace_arr)
            free(unique_depth_arr)
            free(unique_amp_arr)
            raise MemoryError("Could not allocate memory for working arrays")
    
    # Step 5: Perform interpolation
    try:
        with nogil:
            for j in prange(ni, schedule='static'):
                thread_id = omp_get_thread_num()
                
                for k in range(nx):
                    # Extract traces for this (j, k) position
                    for i in range(nt):
                        depth_trace_arr[thread_id][i] = depth_view[i, j, k]
                        amp_trace_arr[thread_id][i] = seis_view[i, j, k]
                    
                    # Make monotonic and remove duplicates
                    unique_size = make_monotonic(depth_trace_arr[thread_id], amp_trace_arr[thread_id],
                                                 unique_depth_arr[thread_id], unique_amp_arr[thread_id], nt)
                    
                    # Handle degenerate cases
                    if unique_size < 2:
                        if unique_size > 0:
                            for i in range(nt):
                                result_view[i, j, k] = unique_amp_arr[thread_id][0]
                        else:
                            for i in range(nt):
                                result_view[i, j, k] = 0.0
                    else:
                        # Interpolate for each output depth
                        for i in range(nt):
                            result_view[i, j, k] = linear_interp(
                                depth_axis_view[i],
                                unique_depth_arr[thread_id],
                                unique_amp_arr[thread_id],
                                unique_size,
                                unique_amp_arr[thread_id][0],  # left value
                                0.0                            # right value
                            )
    finally:
        # Clean up memory
        for i in range(max_threads):
            free(depth_trace_arr[i])
            free(amp_trace_arr[i])
            free(unique_depth_arr[i])
            free(unique_amp_arr[i])
        free(depth_trace_arr)
        free(amp_trace_arr)
        free(unique_depth_arr)
        free(unique_amp_arr)
    
    return result, dz