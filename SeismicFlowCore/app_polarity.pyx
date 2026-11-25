# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: infer_types=True

"""
Apparent Polarity Attribute Computation
Based on Taner et al. (1979) - Complex Seismic Trace Analysis

Apparent polarity is defined as the sign of f(t) when A(t) has a local maximum,
where f(t) is the seismic trace and A(t) is the instantaneous amplitude (envelope).

Mathematical Definition:
- Analytic signal: F(t) = f(t) + jf_H(t)
- Instantaneous amplitude: A(t) = |F(t)| = sqrt(f(t)^2 + f_H(t)^2)
- Apparent polarity: P(t) = sign(f(t)) when dA/dt = 0 and d²A/dt² < 0

Reference:
Taner, M. T., Koehler, F., & Sheriff, R. E. (1979). Complex seismic trace analysis.
Geophysics, 44(6), 1041-1063. https://doi.org/10.1190/1.1440994
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free

cnp.import_array()

# FFTW3 interface for fast Fourier transforms
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

def apparent_polarity(cnp.ndarray[cnp.float32_t, ndim=3] data,
                      float threshold=1e-6,
                      bint use_masking=True,
                      int window_size=33):
    """
    Compute apparent polarity attribute using complex trace analysis.
    
    Algorithm (Taner et al. 1979):
    1. Construct analytic signal F(t) = f(t) + jf_H(t) via Hilbert transform
       where f_H(t) is obtained by:
       - FFT of real signal f(t)
       - Zero negative frequencies, double positive frequencies
       - Inverse FFT yields quadrature component f_H(t)
    
    2. Calculate instantaneous amplitude (envelope):
       A(t) = |F(t)| = sqrt(f(t)^2 + f_H(t)^2)
    
    3. Detect local maxima of A(t):
       - Interior maximum at t_i when A(t_i) > A(t_i-1) and A(t_i) > A(t_i+1)
       - Boundary maximum at t=0 when A(0) > A(1)
       - Boundary maximum at t=n-1 when A(n-1) > A(n-2)
    
    4. Extract polarity at envelope maxima:
       P(t_i) = sign(f(t_i)) = +1 if f(t_i) > 0
                              = -1 if f(t_i) < 0
                              =  0 if f(t_i) = 0
    
    Parameters
    ----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic amplitude data f(t)
    threshold : float, optional
        Minimum amplitude for peak detection (default: 1e-6)
        Envelope maxima below this value are ignored
    use_masking : bool, optional
        Apply local window masking to handle dead/muted traces (default: True)
    window_size : int, optional
        Size of local window for masking in samples (default: 33)
        
    Returns
    -------
    polarity : ndarray of float32, shape (time, inline, crossline)
        Apparent polarity attribute:
        +1.0 where envelope has maximum and f(t) > 0
        -1.0 where envelope has maximum and f(t) < 0
         0.0 elsewhere (not at envelope maximum)
         
    Notes
    -----
    - Output is sparse: non-zero only at envelope maxima
    - Produces characteristic "dotted" appearance in seismic displays
    - Uses FFTW batch transforms for computational efficiency
    - Parallel processing over inline direction
    - Properly handles boundary conditions: first and last time samples
      can be detected as peaks if they exceed their single neighbor
    """
    cdef int num_time = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] polarity_output = np.zeros_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] polarity_view = polarity_output
    
    cdef int i, j, k, half_n, t_start, t_end, t, half_window
    cdef fftwf_complex* buffer_in
    cdef fftwf_complex* buffer_out
    cdef float real_part, imag_part, envelope, scale, polarity_sign
    cdef float max_amp, abs_val, trace_val
    cdef bint is_maximum
    cdef float* envelope_buffer
    
    half_n = num_time // 2
    scale = 1.0 / <float>num_time  # IFFT normalization factor
    half_window = window_size // 2
    
    # FFTW batch transform configuration
    # Transforms num_crossline traces simultaneously for each inline
    cdef int rank = 1
    cdef int n[1]
    n[0] = num_time
    cdef int howmany = num_crossline
    cdef int idist = num_time  # Distance between consecutive traces in array
    cdef int odist = num_time
    cdef int istride = 1  # Distance between consecutive elements in trace
    cdef int ostride = 1
    
    # Allocate temporary buffers for plan creation
    cdef fftwf_complex* temp_in = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
    cdef fftwf_complex* temp_out = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
    
    if temp_in == NULL or temp_out == NULL:
        if temp_in != NULL:
            fftwf_free(temp_in)
        if temp_out != NULL:
            fftwf_free(temp_out)
        raise MemoryError("Failed to allocate FFTW buffers")
    
    # Create forward and backward transform plans
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
    
    # Parallel processing over inline slices
    for j in prange(num_inline, nogil=True, schedule='static'):
        # Allocate buffers for this thread
        buffer_in = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
        buffer_out = <fftwf_complex*>fftwf_malloc(num_crossline * num_time * sizeof(fftwf_complex))
        envelope_buffer = <float*>malloc(num_time * sizeof(float))
        
        if buffer_in == NULL or buffer_out == NULL or envelope_buffer == NULL:
            if buffer_in != NULL:
                fftwf_free(buffer_in)
            if buffer_out != NULL:
                fftwf_free(buffer_out)
            if envelope_buffer != NULL:
                free(envelope_buffer)
            continue
        
        # Load inline slice into FFT input buffer
        # Initialize real part with seismic data, imaginary part with zeros
        for k in range(num_crossline):
            for i in range(num_time):
                buffer_in[k * num_time + i][0] = data_view[i, j, k]
                buffer_in[k * num_time + i][1] = 0.0
        
        # Step 1: Forward FFT to transform to frequency domain
        fftwf_execute_dft(plan_forward, buffer_in, buffer_out)
        
        # Step 2: Apply Hilbert transform in frequency domain
        # For analytic signal: zero negative frequencies, double positive frequencies
        # DC (i=0) and Nyquist (i=N/2 for even N) remain unchanged
        for k in range(num_crossline):
            if num_time % 2 == 0:
                # Even length: [DC] [positive freqs] [Nyquist] [negative freqs]
                for i in range(1, half_n):
                    buffer_out[k * num_time + i][0] *= 2.0
                    buffer_out[k * num_time + i][1] *= 2.0
                for i in range(half_n + 1, num_time):
                    buffer_out[k * num_time + i][0] = 0.0
                    buffer_out[k * num_time + i][1] = 0.0
            else:
                # Odd length: [DC] [positive freqs] [negative freqs]
                for i in range(1, half_n + 1):
                    buffer_out[k * num_time + i][0] *= 2.0
                    buffer_out[k * num_time + i][1] *= 2.0
                for i in range(half_n + 1, num_time):
                    buffer_out[k * num_time + i][0] = 0.0
                    buffer_out[k * num_time + i][1] = 0.0
        
        # Step 3: Inverse FFT to obtain analytic signal in time domain
        # Result: F(t) = f(t) + jf_H(t)
        fftwf_execute_dft(plan_backward, buffer_out, buffer_in)
        
        # Step 4: Process each trace to find envelope maxima and extract polarity
        for k in range(num_crossline):
            # Compute instantaneous amplitude A(t) = |F(t)|
            for i in range(num_time):
                real_part = buffer_in[k * num_time + i][0] * scale
                imag_part = buffer_in[k * num_time + i][1] * scale
                envelope = sqrt(real_part * real_part + imag_part * imag_part)
                envelope_buffer[i] = envelope
            
            # Detect local maxima in envelope and extract polarity sign
            
            # First time sample (boundary condition)
            if num_time > 1:
                is_maximum = False
                if envelope_buffer[0] > envelope_buffer[1] and \
                   envelope_buffer[0] > threshold:
                    is_maximum = True
                
                if is_maximum:
                    trace_val = data_view[0, j, k]
                    if trace_val > 0.0:
                        polarity_sign = 1.0
                    elif trace_val < 0.0:
                        polarity_sign = -1.0
                    else:
                        polarity_sign = 0.0
                    
                    if use_masking:
                        t_start = 0
                        t_end = half_window + 1 if half_window + 1 <= num_time else num_time
                        
                        max_amp = 0.0
                        for t in range(t_start, t_end):
                            abs_val = fabs(data_view[t, j, k])
                            if abs_val > max_amp:
                                max_amp = abs_val
                        
                        if max_amp >= threshold:
                            polarity_view[0, j, k] = polarity_sign
                    else:
                        polarity_view[0, j, k] = polarity_sign
            
            # Interior time samples (standard local maximum criterion)
            for i in range(1, num_time - 1):
                is_maximum = False
                
                # Local maximum criterion: A(t_i) > A(t_i-1) and A(t_i) > A(t_i+1)
                if envelope_buffer[i] > envelope_buffer[i - 1] and \
                   envelope_buffer[i] > envelope_buffer[i + 1] and \
                   envelope_buffer[i] > threshold:
                    is_maximum = True
                
                if is_maximum:
                    # Extract sign of original trace at envelope maximum
                    trace_val = data_view[i, j, k]
                    if trace_val > 0.0:
                        polarity_sign = 1.0
                    elif trace_val < 0.0:
                        polarity_sign = -1.0
                    else:
                        polarity_sign = 0.0
                    
                    # Optional: Apply masking to handle dead/muted traces
                    if use_masking:
                        # Check if local window contains significant amplitude
                        t_start = i - half_window if i >= half_window else 0
                        t_end = i + half_window + 1 if i + half_window + 1 <= num_time else num_time
                        
                        max_amp = 0.0
                        for t in range(t_start, t_end):
                            abs_val = fabs(data_view[t, j, k])
                            if abs_val > max_amp:
                                max_amp = abs_val
                        
                        if max_amp >= threshold:
                            polarity_view[i, j, k] = polarity_sign
                        # else: remains zero (dead trace)
                    else:
                        polarity_view[i, j, k] = polarity_sign
            
            # Last time sample (boundary condition)
            if num_time > 1:
                is_maximum = False
                if envelope_buffer[num_time - 1] > envelope_buffer[num_time - 2] and \
                   envelope_buffer[num_time - 1] > threshold:
                    is_maximum = True
                
                if is_maximum:
                    trace_val = data_view[num_time - 1, j, k]
                    if trace_val > 0.0:
                        polarity_sign = 1.0
                    elif trace_val < 0.0:
                        polarity_sign = -1.0
                    else:
                        polarity_sign = 0.0
                    
                    if use_masking:
                        t_start = num_time - 1 - half_window if num_time - 1 >= half_window else 0
                        t_end = num_time
                        
                        max_amp = 0.0
                        for t in range(t_start, t_end):
                            abs_val = fabs(data_view[t, j, k])
                            if abs_val > max_amp:
                                max_amp = abs_val
                        
                        if max_amp >= threshold:
                            polarity_view[num_time - 1, j, k] = polarity_sign
                    else:
                        polarity_view[num_time - 1, j, k] = polarity_sign
        
        # Release thread-local buffers
        fftwf_free(buffer_in)
        fftwf_free(buffer_out)
        free(envelope_buffer)
    
    # Cleanup FFTW plans
    fftwf_destroy_plan(plan_forward)
    fftwf_destroy_plan(plan_backward)
    
    return polarity_output