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
from libc.math cimport sqrt, atan2, fabs, M_PI
from libc.stdlib cimport malloc, free, qsort

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

# Comparison function for qsort - must be noexcept
cdef int compare_floats(const void* a, const void* b) noexcept nogil:
    cdef float diff = (<float*>a)[0] - (<float*>b)[0]
    if diff < 0:
        return -1
    elif diff > 0:
        return 1
    else:
        return 0

cdef inline float unwrap_phase(float current_phase, float previous_unwrapped) nogil:
    """Unwrap phase by detecting jumps > pi and accumulating corrections"""
    cdef float diff = current_phase - previous_unwrapped
    cdef float correction = 0.0
    
    # Add multiples of 2*pi to minimize the jump
    while diff > M_PI:
        diff -= 2.0 * M_PI
        correction -= 2.0 * M_PI
    while diff < -M_PI:
        diff += 2.0 * M_PI
        correction += 2.0 * M_PI
    
    return current_phase + correction

def sweetness(cnp.ndarray[cnp.float32_t, ndim=3] data, float sampling_interval_seconds, bint debug=False, int debug_inline=-1, int debug_xline=-1):
    """
    Ultra-fast sweetness computation using FFTW.
    
    Sweetness = Envelope / sqrt(InstantaneousFrequency)
    Uses adaptive per-trace frequency thresholding (42nd percentile, minimum 1 Hz).
    
    Parameters:
    -----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic data
    sampling_interval_seconds : float
        Sampling interval in seconds
    debug : bool, optional
        Enable debug output for specific trace
    debug_inline : int, optional
        Inline index for debug output (-1 to disable)
    debug_xline : int, optional
        Crossline index for debug output (-1 to disable)
        
    Returns:
    --------
    sweetness : ndarray of float32, shape (time, inline, crossline)
        Sweetness attribute
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
    cdef float* envelope_buffer
    cdef float* inst_freq_buffer
    cdef float* freq_copy
    cdef float real_part, imag_part, envelope_val
    cdef float phase_rad, unwrapped_phase, prev_phase, phase_diff
    cdef float inst_freq_hz, freq_threshold, sweetness_val
    cdef float scale, two_pi_inv
    cdef int percentile_idx
    cdef float percentile_pos, fraction
    cdef int lower_idx, upper_idx
    cdef float inst_freq_hz_thresh
    
    half_n = num_time // 2
    scale = 1.0 / <float>num_time
    two_pi_inv = 1.0 / (2.0 * M_PI)
    
    # 42nd percentile index
    percentile_idx = <int>(0.42 * <float>num_time)
    if percentile_idx >= num_time:
        percentile_idx = num_time - 1
    
    # FFTW batch parameters
    cdef int rank = 1
    cdef int n[1]
    n[0] = num_time
    cdef int howmany = num_crossline
    cdef int idist = num_time
    cdef int odist = num_time
    cdef int istride = 1
    cdef int ostride = 1
    
    # Create reusable plans
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
        envelope_buffer = <float*>fftwf_malloc(num_crossline * num_time * sizeof(float))
        inst_freq_buffer = <float*>fftwf_malloc(num_crossline * num_time * sizeof(float))
        freq_copy = <float*>fftwf_malloc(num_time * sizeof(float))
        
        if buffer_in == NULL or buffer_out == NULL or envelope_buffer == NULL or inst_freq_buffer == NULL or freq_copy == NULL:
            if buffer_in != NULL:
                fftwf_free(buffer_in)
            if buffer_out != NULL:
                fftwf_free(buffer_out)
            if envelope_buffer != NULL:
                fftwf_free(envelope_buffer)
            if inst_freq_buffer != NULL:
                fftwf_free(inst_freq_buffer)
            if freq_copy != NULL:
                fftwf_free(freq_copy)
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
        
        # Compute envelope, unwrap phase, and instantaneous frequency
        for k in range(num_crossline):
            prev_phase = 0.0
            
            # Pass 1: Compute envelope and unwrap phase
            for i in range(num_time):
                real_part = buffer_in[k * num_time + i][0] * scale
                imag_part = buffer_in[k * num_time + i][1] * scale
                
                # Envelope
                envelope_buffer[k * num_time + i] = sqrt(real_part * real_part + imag_part * imag_part)
                
                # Phase and unwrap
                phase_rad = atan2(imag_part, real_part)
                if i == 0:
                    unwrapped_phase = phase_rad
                    prev_phase = phase_rad
                else:
                    unwrapped_phase = unwrap_phase(phase_rad, prev_phase)
                    prev_phase = unwrapped_phase
                
                # Store unwrapped phase temporarily in buffer_in imaginary part
                buffer_in[k * num_time + i][1] = unwrapped_phase
            
            # Pass 2: Compute instantaneous frequency using gradient
            for i in range(num_time):
                if i == 0:
                    # Forward difference for first point
                    phase_diff = (buffer_in[k * num_time + 1][1] - buffer_in[k * num_time + 0][1]) / sampling_interval_seconds
                elif i == num_time - 1:
                    # Backward difference for last point
                    phase_diff = (buffer_in[k * num_time + i][1] - buffer_in[k * num_time + i - 1][1]) / sampling_interval_seconds
                else:
                    # Central difference for middle points
                    phase_diff = (buffer_in[k * num_time + i + 1][1] - buffer_in[k * num_time + i - 1][1]) / (2.0 * sampling_interval_seconds)
                
                inst_freq_hz = fabs(phase_diff) * two_pi_inv
                inst_freq_buffer[k * num_time + i] = inst_freq_hz
        
        # Compute per-trace adaptive threshold and sweetness
        for k in range(num_crossline):
            # Declare all variables at the start of the block
            percentile_pos = 0.42 * (<float>(num_time - 1))
            lower_idx = <int>percentile_pos
            upper_idx = lower_idx + 1
            fraction = percentile_pos - <float>lower_idx
            
            # Copy frequencies for sorting to find percentile
            for i in range(num_time):
                freq_copy[i] = inst_freq_buffer[k * num_time + i]
            
            # Sort to find 42nd percentile with linear interpolation (matches np.percentile default)
            qsort(freq_copy, num_time, sizeof(float), compare_floats)
            
            # Linear interpolation for percentile (matching numpy default behavior)
            if upper_idx >= num_time:
                freq_threshold = freq_copy[num_time - 1]
            else:
                freq_threshold = freq_copy[lower_idx] * (1.0 - fraction) + freq_copy[upper_idx] * fraction
            
            # Apply minimum threshold of 1 Hz
            if freq_threshold < 1.0:
                freq_threshold = 1.0
            

            # Compute sweetness
            for i in range(num_time):
                inst_freq_hz = inst_freq_buffer[k * num_time + i]
                
                # Apply threshold
                if inst_freq_hz < freq_threshold:
                    inst_freq_hz = freq_threshold
                
                # Sweetness = Envelope / sqrt(InstFreq)
                sweetness_val = envelope_buffer[k * num_time + i] / sqrt(inst_freq_hz)
                output_view[i, j, k] = sweetness_val
        
        fftwf_free(buffer_in)
        fftwf_free(buffer_out)
        fftwf_free(envelope_buffer)
        fftwf_free(inst_freq_buffer)
        fftwf_free(freq_copy)
    
    fftwf_destroy_plan(plan_forward)
    fftwf_destroy_plan(plan_backward)
    
    return output