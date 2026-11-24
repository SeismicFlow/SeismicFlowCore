# cython: language_level=3
# xb571.pyx - GST Coherence with Eigen eigenvalue solver
# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp, M_PI, fabs

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
cdef void compute_gaussian_derivative_kernel(float* kernel, int size, float sigma) noexcept nogil:
    """Compute 1D Gaussian derivative kernel matching scipy.ndimage."""
    cdef int i, center = size // 2
    cdef float x, sum_abs = 0.0
    cdef float sigma3 = sigma * sigma * sigma
    cdef float denom = 2.0 * sigma * sigma
    cdef float coeff = 1.0 / (sqrt(2.0 * M_PI) * sigma3)
    
    for i in range(size):
        x = <float>(i - center)
        kernel[i] = -x * coeff * exp(-x * x / denom)
        sum_abs += fabs(kernel[i])
    
    if sum_abs > 1e-10:
        for i in range(size):
            kernel[i] /= sum_abs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline int reflect_index(int idx, int size) noexcept nogil:
    """NumPy's mode='reflect' - optimized version."""
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

def gst_coherence(cnp.ndarray[cnp.float32_t, ndim=3] data, tuple window=(5, 5, 5), float sigma=1.0):
    """
    Compute GST coherence using Eigen for eigenvalue computation.
    """
    cdef int n_time = data.shape[0]
    cdef int n_inline = data.shape[1]
    cdef int n_crossline = data.shape[2]
    cdef int window_t = window[0]
    cdef int window_il = window[1]
    cdef int window_xl = window[2]
    cdef int half_t = window_t // 2
    cdef int half_il = window_il // 2
    cdef int half_xl = window_xl // 2
    cdef int kernel_size, kernel_half
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] coherence = np.zeros((n_time, n_inline, n_crossline), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] grad_t = np.zeros((n_time, n_inline, n_crossline), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] grad_il = np.zeros((n_time, n_inline, n_crossline), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] grad_xl = np.zeros((n_time, n_inline, n_crossline), dtype=np.float32)
    
    cdef float* data_ptr = <float*> cnp.PyArray_DATA(data)
    cdef float* coherence_ptr = <float*> cnp.PyArray_DATA(coherence)
    cdef float* grad_t_ptr = <float*> cnp.PyArray_DATA(grad_t)
    cdef float* grad_il_ptr = <float*> cnp.PyArray_DATA(grad_il)
    cdef float* grad_xl_ptr = <float*> cnp.PyArray_DATA(grad_xl)
    
    cdef float* derivative_kernel
    cdef int t, il, xl, i, k, refl_idx, win_t, win_il, win_xl
    cdef int actual_t, actual_il, actual_xl
    cdef int refl_t, refl_il, refl_xl
    cdef float val, sum_val
    cdef float gxx, gxy, gxz, gyy, gyz, gzz
    cdef float eigenvalues[3]
    cdef float coherence_val
    cdef float gt, gil, gxl
    cdef int idx_out
    
    kernel_size = <int>(8 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_half = kernel_size // 2
    
    derivative_kernel = <float*> malloc(kernel_size * sizeof(float))
    compute_gaussian_derivative_kernel(derivative_kernel, kernel_size, sigma)
    
    # ============================================================================
    # STEP 1: Compute Gradients (parallelized with private variables)
    # ============================================================================
    
    # Gradient in time direction
    for il in prange(n_inline, nogil=True, schedule='static'):
        for xl in range(n_crossline):
            for t in range(n_time):
                sum_val = 0.0
                for k in range(kernel_size):
                    i = t + k - kernel_half
                    refl_idx = reflect_index(i, n_time)
                    val = data_ptr[refl_idx * n_inline * n_crossline + il * n_crossline + xl]
                    sum_val = sum_val + val * derivative_kernel[k]
                
                idx_out = t * n_inline * n_crossline + il * n_crossline + xl
                grad_t_ptr[idx_out] = sum_val
    
    # Gradient in inline direction
    for il in prange(n_inline, nogil=True, schedule='static'):
        for xl in range(n_crossline):
            for t in range(n_time):
                sum_val = 0.0
                for k in range(kernel_size):
                    i = il + k - kernel_half
                    refl_idx = reflect_index(i, n_inline)
                    val = data_ptr[t * n_inline * n_crossline + refl_idx * n_crossline + xl]
                    sum_val = sum_val + val * derivative_kernel[k]
                
                idx_out = t * n_inline * n_crossline + il * n_crossline + xl
                grad_il_ptr[idx_out] = sum_val
    
    # Gradient in crossline direction
    for il in prange(n_inline, nogil=True, schedule='static'):
        for xl in range(n_crossline):
            for t in range(n_time):
                sum_val = 0.0
                for k in range(kernel_size):
                    i = xl + k - kernel_half
                    refl_idx = reflect_index(i, n_crossline)
                    val = data_ptr[t * n_inline * n_crossline + il * n_crossline + refl_idx]
                    sum_val = sum_val + val * derivative_kernel[k]
                
                idx_out = t * n_inline * n_crossline + il * n_crossline + xl
                grad_xl_ptr[idx_out] = sum_val
    
    # ============================================================================
    # STEP 2: Compute GST with CENTERED window and Eigen eigenvalues
    # ============================================================================
    
    for il in prange(n_inline, nogil=True, schedule='dynamic'):
        for xl in range(n_crossline):
            for t in range(n_time):
                # Initialize GST (private to this thread)
                gxx = 0.0
                gxy = 0.0
                gxz = 0.0
                gyy = 0.0
                gyz = 0.0
                gzz = 0.0
                
                # Window is CENTERED around (t, il, xl)
                for win_t in range(window_t):
                    actual_t = t - half_t + win_t
                    refl_t = reflect_index(actual_t, n_time)
                    
                    for win_il in range(window_il):
                        actual_il = il - half_il + win_il
                        refl_il = reflect_index(actual_il, n_inline)
                        
                        for win_xl in range(window_xl):
                            actual_xl = xl - half_xl + win_xl
                            refl_xl = reflect_index(actual_xl, n_crossline)
                            
                            i = refl_t * n_inline * n_crossline + refl_il * n_crossline + refl_xl
                            gt = grad_t_ptr[i]
                            gil = grad_il_ptr[i]
                            gxl = grad_xl_ptr[i]
                            
                            # Accumulate GST
                            gxx = gxx + gt * gt
                            gxy = gxy + gt * gil
                            gxz = gxz + gt * gxl
                            gyy = gyy + gil * gil
                            gyz = gyz + gil * gxl
                            gzz = gzz + gxl * gxl
                
                # Compute eigenvalues using Eigen (after accumulation is complete)
                compute_eigenvalues_3x3(gxx, gxy, gxz, gyy, gyz, gzz, eigenvalues)
                
                # Compute coherence
                if eigenvalues[0] + eigenvalues[1] > 1e-10:
                    coherence_val = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1])
                    if coherence_val < 0.0:
                        coherence_val = 0.0
                    elif coherence_val > 1.0:
                        coherence_val = 1.0
                else:
                    coherence_val = 0.0
                
                coherence_ptr[t * n_inline * n_crossline + il * n_crossline + xl] = coherence_val
    
    free(derivative_kernel)
    return coherence