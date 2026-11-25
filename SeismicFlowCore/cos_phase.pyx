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
from libc.math cimport atan2, cos, M_PI

cnp.import_array()

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
    
    fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int howmany,
                                    fftwf_complex* input, const int* inembed,
                                    int istride, int idist,
                                    fftwf_complex* output, const int* onembed,
                                    int ostride, int odist,
                                    int sign, unsigned flags)
    
    void fftwf_execute_dft(fftwf_plan plan, fftwf_complex* input, fftwf_complex* output) nogil
    void fftwf_destroy_plan(fftwf_plan plan)

cdef inline float unwrap_phase(float current_phase, float previous_phase) nogil:
    """Unwrap phase by detecting jumps > pi"""
    cdef float diff = current_phase - previous_phase
    
    # Detect phase jumps
    if diff > M_PI:
        return current_phase - 2.0 * M_PI
    elif diff < -M_PI:
        return current_phase + 2.0 * M_PI
    else:
        return current_phase

def cosine_of_phase(cnp.ndarray[cnp.float32_t, ndim=3] data):
    """
    Ultra-fast cosine of unwrapped phase using FFTW.
    
    Parameters:
    -----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic data
        
    Returns:
    --------
    cos_phase : ndarray of float32, shape (time, inline, crossline)
        Cosine of unwrapped instantaneous phase
    """
    cdef int num_time = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k, half_n
    cdef fftwf_complex* buffer_in
    cdef fftwf_complex* buffer_out
    cdef float real_part, imag_part, phase_rad, unwrapped_phase, prev_phase
    cdef float scale
    
    half_n = num_time // 2
    scale = 1.0 / <float>num_time
    
    # FFTW batch parameters
    cdef int rank = 1
    cdef int n[1]
    n[0] = num_time
    cdef int howmany = num_crossline
    cdef int idist = num_time
    cdef int odist = num_time
    cdef int istride = 1
    cdef int ostride = 1
    
    # Create reusable plans before parallel region
    cdef fftwf_complex* temp_in = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
    cdef fftwf_complex* temp_out = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
    
    if temp_in == NULL or temp_out == NULL:
        if temp_in != NULL:
            fftwf_free(temp_in)
        if temp_out != NULL:
            fftwf_free(temp_out)
        raise MemoryError("Failed to allocate FFTW buffers")
    
    cdef fftwf_plan plan_forward = fftwf_plan_many_dft(rank, n, howmany,
                                       temp_in, NULL, istride, idist,
                                       temp_out, NULL, ostride, odist,
                                       FFTW_FORWARD, FFTW_ESTIMATE)
    
    cdef fftwf_plan plan_backward = fftwf_plan_many_dft(rank, n, howmany,
                                        temp_out, NULL, istride, idist,
                                        temp_in, NULL, ostride, odist,
                                        FFTW_BACKWARD, FFTW_ESTIMATE)
    
    if plan_forward == NULL or plan_backward == NULL:
        if plan_forward != NULL:
            fftwf_destroy_plan(plan_forward)
        if plan_backward != NULL:
            fftwf_destroy_plan(plan_backward)
        fftwf_free(temp_in)
        fftwf_free(temp_out)
        raise RuntimeError("Failed to create FFTW plans")
    
    fftwf_free(temp_in)
    fftwf_free(temp_out)
    
    # Process each inline in parallel
    for j in prange(num_inline, nogil=True, schedule='static'):
        buffer_in = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
        buffer_out = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
        
        if buffer_in == NULL or buffer_out == NULL:
            if buffer_in != NULL:
                fftwf_free(buffer_in)
            if buffer_out != NULL:
                fftwf_free(buffer_out)
            continue
        
        # Load inline slice
        for k in range(num_crossline):
            for i in range(num_time):
                buffer_in[k * num_time + i][0] = data_view[i, j, k]
                buffer_in[k * num_time + i][1] = 0.0
        
        # Forward FFT
        fftwf_execute_dft(plan_forward, buffer_in, buffer_out)
        
        # Apply Hilbert transform
        for k in range(num_crossline):
            if num_time % 2 == 0:
                for i in range(1, half_n):
                    buffer_out[k * num_time + i][0] *= 2.0
                    buffer_out[k * num_time + i][1] *= 2.0
                for i in range(half_n + 1, num_time):
                    buffer_out[k * num_time + i][0] = 0.0
                    buffer_out[k * num_time + i][1] = 0.0
            else:
                for i in range(1, half_n + 1):
                    buffer_out[k * num_time + i][0] *= 2.0
                    buffer_out[k * num_time + i][1] *= 2.0
                for i in range(half_n + 1, num_time):
                    buffer_out[k * num_time + i][0] = 0.0
                    buffer_out[k * num_time + i][1] = 0.0
        
        # Inverse FFT
        fftwf_execute_dft(plan_backward, buffer_out, buffer_in)
        
        # Extract phase, unwrap, and compute cosine
        for k in range(num_crossline):
            prev_phase = 0.0
            
            for i in range(num_time):
                real_part = buffer_in[k * num_time + i][0] * scale
                imag_part = buffer_in[k * num_time + i][1] * scale
                phase_rad = atan2(imag_part, real_part)
                
                # Unwrap phase along time axis
                if i == 0:
                    unwrapped_phase = phase_rad
                else:
                    unwrapped_phase = unwrap_phase(phase_rad, prev_phase)
                
                prev_phase = unwrapped_phase
                
                # Compute cosine
                output_view[i, j, k] = cos(unwrapped_phase)
        
        fftwf_free(buffer_in)
        fftwf_free(buffer_out)
    
    fftwf_destroy_plan(plan_forward)
    fftwf_destroy_plan(plan_backward)
    
    return output