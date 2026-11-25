# cython: language_level=3
# xb572.pyx - FINAL: Memory-optimized, no GUI freezing
# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp, M_PI

# Eigen C++ interface for eigenvalue computation
cdef extern from *:
    """
    #include <Eigen/Dense>
    
    void compute_eigenvalues_3x3(float g11, float g12, float g13,
                                  float g22, float g23, float g33,
                                  float* eigenvalues) {
        Eigen::Matrix3f mat;
        mat << g11, g12, g13,
               g12, g22, g23,
               g13, g23, g33;
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(mat, Eigen::EigenvaluesOnly);
        Eigen::Vector3f evals = solver.eigenvalues();
        
        // Store in descending order
        eigenvalues[0] = evals(2);
        eigenvalues[1] = evals(1);
        eigenvalues[2] = evals(0);
    }
    """
    void compute_eigenvalues_3x3(float g11, float g12, float g13,
                                  float g22, float g23, float g33,
                                  float* eigenvalues) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void compute_gaussian_kernel(float* kernel, int size, float sigma) noexcept nogil:
    """Optimized Gaussian kernel computation."""
    cdef int i
    cdef int center = size // 2
    cdef float x
    cdef float sum_val = 0.0
    cdef float inv_variance = 1.0 / (sigma * sigma)
    cdef float coeff = 1.0 / (sigma * sqrt(2.0 * M_PI))
    
    for i in range(size):
        x = <float>(i - center)
        kernel[i] = coeff * exp(-0.5 * x * x * inv_variance)
        sum_val += kernel[i]
    
    # Normalize
    cdef float inv_sum = 1.0 / sum_val
    for i in range(size):
        kernel[i] *= inv_sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline int reflect_index(int idx, int size) noexcept nogil:
    """Fast reflection boundary condition."""
    if idx >= 0 and idx < size:
        return idx
    if size == 1:
        return 0
    if idx < 0:
        idx = -idx
        if idx < size:
            return idx
    if idx >= size:
        idx = 2 * size - idx - 2
        if idx >= 0 and idx < size:
            return idx
    cdef int period = 2 * (size - 1)
    idx = idx % period
    return period - idx if idx >= size else idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void filter_1d_time_inplace(float* data, float* temp,
                                  int n_t, int n_il, int n_xl,
                                  float* kernel, int ksize) noexcept nogil:
    """1D filter along time dimension - uses single temp buffer."""
    cdef int t, il, xl, k, neighbor, idx_in, idx_out
    cdef int khalf = ksize // 2
    cdef float sum_val
    cdef int stride_t = n_il * n_xl
    cdef int stride_il = n_xl
    
    for il in range(n_il):
        for xl in range(n_xl):
            for t in range(n_t):
                sum_val = 0.0
                for k in range(ksize):
                    neighbor = reflect_index(t + k - khalf, n_t)
                    idx_in = neighbor * stride_t + il * stride_il + xl
                    sum_val += data[idx_in] * kernel[k]
                idx_out = t * stride_t + il * stride_il + xl
                temp[idx_out] = sum_val
    
    # Copy back
    for il in range(n_il):
        for xl in range(n_xl):
            for t in range(n_t):
                idx_out = t * stride_t + il * stride_il + xl
                data[idx_out] = temp[idx_out]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void filter_1d_inline_inplace(float* data, float* temp,
                                    int n_t, int n_il, int n_xl,
                                    float* kernel, int ksize) noexcept nogil:
    """1D filter along inline dimension - uses single temp buffer."""
    cdef int t, il, xl, k, neighbor, idx_in, idx_out
    cdef int khalf = ksize // 2
    cdef float sum_val
    cdef int stride_t = n_il * n_xl
    cdef int stride_il = n_xl
    
    for t in range(n_t):
        for xl in range(n_xl):
            for il in range(n_il):
                sum_val = 0.0
                for k in range(ksize):
                    neighbor = reflect_index(il + k - khalf, n_il)
                    idx_in = t * stride_t + neighbor * stride_il + xl
                    sum_val += data[idx_in] * kernel[k]
                idx_out = t * stride_t + il * stride_il + xl
                temp[idx_out] = sum_val
    
    # Copy back
    for t in range(n_t):
        for xl in range(n_xl):
            for il in range(n_il):
                idx_out = t * stride_t + il * stride_il + xl
                data[idx_out] = temp[idx_out]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void filter_1d_crossline_inplace(float* data, float* temp,
                                       int n_t, int n_il, int n_xl,
                                       float* kernel, int ksize) noexcept nogil:
    """1D filter along crossline dimension - uses single temp buffer."""
    cdef int t, il, xl, k, neighbor, idx_in, idx_out
    cdef int khalf = ksize // 2
    cdef float sum_val
    cdef int stride_t = n_il * n_xl
    cdef int stride_il = n_xl
    
    for t in range(n_t):
        for il in range(n_il):
            for xl in range(n_xl):
                sum_val = 0.0
                for k in range(ksize):
                    neighbor = reflect_index(xl + k - khalf, n_xl)
                    idx_in = t * stride_t + il * stride_il + neighbor
                    sum_val += data[idx_in] * kernel[k]
                idx_out = t * stride_t + il * stride_il + xl
                temp[idx_out] = sum_val
    
    # Copy back
    for t in range(n_t):
        for il in range(n_il):
            for xl in range(n_xl):
                idx_out = t * stride_t + il * stride_il + xl
                data[idx_out] = temp[idx_out]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void separable_gaussian_3d_inplace(float* data, float* temp,
                                         int n_t, int n_il, int n_xl,
                                         float* kernel_t, int ksize_t,
                                         float* kernel_il, int ksize_il,
                                         float* kernel_xl, int ksize_xl) noexcept nogil:
    """Fast 3D separable Gaussian filter - in-place with single temp buffer."""
    filter_1d_time_inplace(data, temp, n_t, n_il, n_xl, kernel_t, ksize_t)
    filter_1d_inline_inplace(data, temp, n_t, n_il, n_xl, kernel_il, ksize_il)
    filter_1d_crossline_inplace(data, temp, n_t, n_il, n_xl, kernel_xl, ksize_xl)

def chaos(cnp.ndarray[cnp.float32_t, ndim=3] data, 
          float sigma_t=1.5, float sigma_il=1.5, float sigma_xl=1.5,
          float truncate=4.0):
    """
    Maximum performance chaos attribute - memory optimized, no GUI freezing.
    
    Parameters:
    -----------
    data : ndarray
        3D seismic data (time, inline, crossline)
    sigma_t : float
        Gaussian smoothing sigma for time dimension (default: 1.5)
    sigma_il : float
        Gaussian smoothing sigma for inline dimension (default: 1.5)
    sigma_xl : float
        Gaussian smoothing sigma for crossline dimension (default: 1.5)
    truncate : float
        Truncation parameter for Gaussian kernel (default: 4.0)
        
    Returns:
    --------
    chaos_out : ndarray
        Chaos attribute volume (values in [0, 1])
    """
    cdef int n_time = data.shape[0]
    cdef int n_inline = data.shape[1]
    cdef int n_crossline = data.shape[2]
    cdef int total = n_time * n_inline * n_crossline
    
    cdef float* data_ptr = <float*> cnp.PyArray_DATA(data)
    
    # Use numpy arrays for better memory management (prevents GUI freezing)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T11_np = np.empty(total, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T12_np = np.empty(total, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T13_np = np.empty(total, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T22_np = np.empty(total, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T23_np = np.empty(total, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] T33_np = np.empty(total, dtype=np.float32)
    
    cdef float* T11 = <float*> cnp.PyArray_DATA(T11_np)
    cdef float* T12 = <float*> cnp.PyArray_DATA(T12_np)
    cdef float* T13 = <float*> cnp.PyArray_DATA(T13_np)
    cdef float* T22 = <float*> cnp.PyArray_DATA(T22_np)
    cdef float* T23 = <float*> cnp.PyArray_DATA(T23_np)
    cdef float* T33 = <float*> cnp.PyArray_DATA(T33_np)
    
    # Single temp buffer reused for all filtering - minimal memory footprint
    cdef cnp.ndarray[cnp.float32_t, ndim=1] temp_np = np.empty(total, dtype=np.float32)
    cdef float* temp = <float*> cnp.PyArray_DATA(temp_np)
    
    # Output array
    cdef cnp.ndarray[cnp.float32_t, ndim=3] chaos_out = np.zeros(
        (n_time, n_inline, n_crossline), dtype=np.float32
    )
    cdef float* chaos_ptr = <float*> cnp.PyArray_DATA(chaos_out)
    
    cdef int t, il, xl, idx
    cdef float gt, gil, gxl
    cdef float evals[3]
    cdef float chaos_val
    cdef int stride_t = n_inline * n_crossline
    cdef int stride_il = n_crossline
    
    # STEP 1: Gradient computation + tensor products
    for idx in prange(total, nogil=True, schedule='guided'):
        # Decompose flat index
        t = idx // stride_t
        il = (idx % stride_t) // stride_il
        xl = idx % stride_il
        
        # Time gradient
        if t == 0:
            gt = data_ptr[idx + stride_t] - data_ptr[idx]
        elif t == n_time - 1:
            gt = data_ptr[idx] - data_ptr[idx - stride_t]
        else:
            gt = data_ptr[idx + stride_t] - data_ptr[idx - stride_t]
        
        # Inline gradient
        if il == 0:
            gil = data_ptr[idx + stride_il] - data_ptr[idx]
        elif il == n_inline - 1:
            gil = data_ptr[idx] - data_ptr[idx - stride_il]
        else:
            gil = data_ptr[idx + stride_il] - data_ptr[idx - stride_il]
        
        # Crossline gradient
        if xl == 0:
            gxl = data_ptr[idx + 1] - data_ptr[idx]
        elif xl == n_crossline - 1:
            gxl = data_ptr[idx] - data_ptr[idx - 1]
        else:
            gxl = data_ptr[idx + 1] - data_ptr[idx - 1]
        
        # Structure tensor products
        T11[idx] = gt * gt
        T12[idx] = gt * gil
        T13[idx] = gt * gxl
        T22[idx] = gil * gil
        T23[idx] = gil * gxl
        T33[idx] = gxl * gxl
    
    # STEP 2: Setup Gaussian kernels
    cdef int ksize_t = <int>(2 * truncate * sigma_t + 1)
    cdef int ksize_il = <int>(2 * truncate * sigma_il + 1)
    cdef int ksize_xl = <int>(2 * truncate * sigma_xl + 1)
    
    if ksize_t % 2 == 0:
        ksize_t += 1
    if ksize_il % 2 == 0:
        ksize_il += 1
    if ksize_xl % 2 == 0:
        ksize_xl += 1
    
    cdef float* kernel_t = <float*> malloc(ksize_t * sizeof(float))
    cdef float* kernel_il = <float*> malloc(ksize_il * sizeof(float))
    cdef float* kernel_xl = <float*> malloc(ksize_xl * sizeof(float))
    
    compute_gaussian_kernel(kernel_t, ksize_t, sigma_t)
    compute_gaussian_kernel(kernel_il, ksize_il, sigma_il)
    compute_gaussian_kernel(kernel_xl, ksize_xl, sigma_xl)
    
    # STEP 3: Smooth all tensor components
    with nogil:
        separable_gaussian_3d_inplace(T11, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
        separable_gaussian_3d_inplace(T12, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
        separable_gaussian_3d_inplace(T13, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
        separable_gaussian_3d_inplace(T22, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
        separable_gaussian_3d_inplace(T23, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
        separable_gaussian_3d_inplace(T33, temp, n_time, n_inline, n_crossline, 
                                      kernel_t, ksize_t, kernel_il, ksize_il, 
                                      kernel_xl, ksize_xl)
    
    # STEP 4: Compute chaos
    for idx in prange(total, nogil=True, schedule='static'):
        compute_eigenvalues_3x3(T11[idx], T12[idx], T13[idx], 
                               T22[idx], T23[idx], T33[idx], evals)
        
        if evals[0] > 1e-10:
            chaos_val = evals[1] / evals[0]
            
            if chaos_val < 0.0:
                chaos_val = 0.0
            elif chaos_val > 1.0:
                chaos_val = 1.0
        else:
            chaos_val = 0.0
        
        chaos_ptr[idx] = chaos_val
    
    # Cleanup
    free(kernel_t)
    free(kernel_il)
    free(kernel_xl)
    
    return chaos_out