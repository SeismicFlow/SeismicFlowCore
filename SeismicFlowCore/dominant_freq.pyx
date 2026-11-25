# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: infer_types=True

"""
DOMINANT FREQUENCY VIA WEIGHTED AVERAGE INSTANTANEOUS FREQUENCY
================================================================

Implementation of Barnes' weighted average method for stabilizing
instantaneous frequency, combined with spectral bandwidth computation.

References:
Barnes, A.E. (2007). A tutorial on complex seismic trace analysis.
Geophysics, 72(6), W33-W43.

Barnes, A.E. (1993). Instantaneous spectral bandwidth and dominant frequency
with applications to seismic reflection data. Geophysics, 58(3), 419-428.

Method:
1. Hilbert transform → analytic signal z(t) = R(t) + iI(t)
2. Instantaneous frequency: ω(t) = [R(t)·I'(t) - I(t)·R'(t)] / [R²(t) + I²(t)]
3. Weighted averaging using envelope squared:
   f_WAIF(t) = Σ[E²(τ)·f(τ)] / Σ[E²(τ)]
4. Instantaneous bandwidth: BW(t) = |dE/dt| / (2πE)
5. Dominant frequency: f_dom(t) = √[f_WAIF²(t) + BW²(t)]
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt, fabs, M_PI
from libc.stdlib cimport malloc, free

cnp.import_array()

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
    
    void fftwf_execute_dft(fftwf_plan plan, fftwf_complex* input, 
                           fftwf_complex* output) nogil
    void fftwf_destroy_plan(fftwf_plan plan)


cdef inline void envelope_squared_weighted_average(float* freq_data, float* envelope_data, 
                                                   float* output_data, int n, 
                                                   int window) noexcept nogil:
    """
    Weighted averaging using envelope squared (instantaneous power).
    
    f_WAIF(t) = Σ[E²(τ)·f(τ)] / Σ[E²(τ)]
    
    where E is the envelope (instantaneous amplitude) from the Hilbert transform.
    """
    cdef int i, j, half_win
    cdef float weighted_sum, weight_sum, weight
    
    half_win = window // 2
    
    for i in range(n):
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for j in range(max(0, i - half_win), min(n, i + half_win + 1)):
            weight = envelope_data[j] * envelope_data[j]
            weighted_sum += freq_data[j] * weight
            weight_sum += weight
        
        if weight_sum > 1e-12:
            output_data[i] = weighted_sum / weight_sum
        else:
            output_data[i] = 0.0


def dominant_frequency(cnp.ndarray[cnp.float32_t, ndim=3] data,
                       float sampling_interval_seconds,
                       float threshold=1e-6,
                       bint use_masking=True,
                       int window_size=33,
                       int waif_window=5):
    """
    Compute dominant frequency using Barnes weighted average method.
    
    Parameters:
    -----------
    data : ndarray, shape (num_time, num_inline, num_crossline)
        Input 3D seismic volume
        
    sampling_interval_seconds : float
        Time sampling interval (dt) in seconds
        
    threshold : float, default=1e-6
        Minimum amplitude threshold for masking
        
    use_masking : bool, default=True
        Apply amplitude-based quality control mask
        
    window_size : int, default=33
        Temporal window for amplitude masking
        
    waif_window : int, default=5
        Window size for weighted average (should be odd)
    
    Returns:
    --------
    dom_freq_output : ndarray, shape (num_time, num_inline, num_crossline)
        Dominant frequency in Hz
    """
    cdef int num_time = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] real_part = np.empty_like(data)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] imag_part = np.empty_like(data)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] envelope_output = np.empty_like(data)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] inst_freq = np.empty_like(data)
    cdef cnp.ndarray[cnp.float32_t, ndim=3] dom_freq_output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] real_view = real_part
    cdef cnp.float32_t[:, :, :] imag_view = imag_part
    cdef cnp.float32_t[:, :, :] envelope_view = envelope_output
    cdef cnp.float32_t[:, :, :] inst_freq_view = inst_freq
    cdef cnp.float32_t[:, :, :] dom_freq_view = dom_freq_output
    
    cdef int i, j, k, half_n, t_start, t_end, t, half_window
    cdef float re, im, env, re_deriv, im_deriv, envelope_sq
    cdef float freq_rad_per_sec, freq_hz
    cdef float scale, dt, max_amp, abs_val
    cdef float two_pi = 2.0 * M_PI
    cdef float nyquist_freq
    cdef float inst_bw, gradient_val
    cdef float dominant_freq, inst_freq_sq, inst_bw_sq, waif_freq
    
    cdef fftwf_complex* buffer_in
    cdef fftwf_complex* buffer_out
    cdef float* freq_trace
    cdef float* envelope_trace
    cdef float* smoothed_trace
    
    half_n = num_time // 2
    scale = 1.0 / <float>num_time
    half_window = window_size // 2
    dt = sampling_interval_seconds
    nyquist_freq = 0.5 / dt
    
    cdef int rank = 1
    cdef int n[1]
    n[0] = num_time
    cdef int howmany = num_crossline
    cdef int idist = num_time
    cdef int odist = num_time
    cdef int istride = 1
    cdef int ostride = 1
    
    cdef fftwf_complex* temp_in = <fftwf_complex*>fftwf_malloc(
        num_crossline * num_time * sizeof(fftwf_complex))
    cdef fftwf_complex* temp_out = <fftwf_complex*>fftwf_malloc(
        num_crossline * num_time * sizeof(fftwf_complex))
    
    if temp_in == NULL or temp_out == NULL:
        if temp_in != NULL:
            fftwf_free(temp_in)
        if temp_out != NULL:
            fftwf_free(temp_out)
        raise MemoryError("Failed to allocate FFTW planning buffers")
    
    cdef fftwf_plan plan_forward = fftwf_plan_many_dft(
        rank, n, howmany,
        temp_in, NULL, istride, idist,
        temp_out, NULL, ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE)
    
    cdef fftwf_plan plan_backward = fftwf_plan_many_dft(
        rank, n, howmany,
        temp_out, NULL, istride, odist,
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
    
    for j in prange(num_inline, nogil=True, schedule='static'):
        buffer_in = <fftwf_complex*>fftwf_malloc(
            num_crossline * num_time * sizeof(fftwf_complex))
        buffer_out = <fftwf_complex*>fftwf_malloc(
            num_crossline * num_time * sizeof(fftwf_complex))
        
        if buffer_in == NULL or buffer_out == NULL:
            if buffer_in != NULL:
                fftwf_free(buffer_in)
            if buffer_out != NULL:
                fftwf_free(buffer_out)
            continue
        
        for k in range(num_crossline):
            for i in range(num_time):
                buffer_in[k * num_time + i][0] = data_view[i, j, k]
                buffer_in[k * num_time + i][1] = 0.0
        
        fftwf_execute_dft(plan_forward, buffer_in, buffer_out)
        
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
        
        fftwf_execute_dft(plan_backward, buffer_out, buffer_in)
        
        for k in range(num_crossline):
            for i in range(num_time):
                re = buffer_in[k * num_time + i][0] * scale
                im = buffer_in[k * num_time + i][1] * scale
                
                real_view[i, j, k] = re
                imag_view[i, j, k] = im
                envelope_view[i, j, k] = sqrt(re * re + im * im)
        
        fftwf_free(buffer_in)
        fftwf_free(buffer_out)
    
    fftwf_destroy_plan(plan_forward)
    fftwf_destroy_plan(plan_backward)
    
    for j in prange(num_inline, nogil=True, schedule='static'):
        for k in range(num_crossline):
            for i in range(num_time):
                re = real_view[i, j, k]
                im = imag_view[i, j, k]
                envelope_sq = re * re + im * im
                
                if envelope_sq < 1e-12:
                    inst_freq_view[i, j, k] = 0.0
                    continue
                
                if i == 0:
                    re_deriv = (real_view[i+1, j, k] - re) / dt
                    im_deriv = (imag_view[i+1, j, k] - im) / dt
                elif i == num_time - 1:
                    re_deriv = (re - real_view[i-1, j, k]) / dt
                    im_deriv = (im - imag_view[i-1, j, k]) / dt
                else:
                    re_deriv = (real_view[i+1, j, k] - real_view[i-1, j, k]) / (2.0 * dt)
                    im_deriv = (imag_view[i+1, j, k] - imag_view[i-1, j, k]) / (2.0 * dt)
                
                freq_rad_per_sec = (re * im_deriv - im * re_deriv) / envelope_sq
                freq_hz = fabs(freq_rad_per_sec / two_pi)
                
                if freq_hz > nyquist_freq:
                    freq_hz = nyquist_freq
                
                inst_freq_view[i, j, k] = freq_hz
    
    for j in prange(num_inline, nogil=True, schedule='static'):
        freq_trace = <float*>malloc(num_time * sizeof(float))
        envelope_trace = <float*>malloc(num_time * sizeof(float))
        smoothed_trace = <float*>malloc(num_time * sizeof(float))
        
        if freq_trace == NULL or envelope_trace == NULL or smoothed_trace == NULL:
            if freq_trace != NULL:
                free(freq_trace)
            if envelope_trace != NULL:
                free(envelope_trace)
            if smoothed_trace != NULL:
                free(smoothed_trace)
            continue
        
        for k in range(num_crossline):
            for i in range(num_time):
                freq_trace[i] = inst_freq_view[i, j, k]
                envelope_trace[i] = envelope_view[i, j, k]
            
            envelope_squared_weighted_average(freq_trace, envelope_trace, 
                                            smoothed_trace, num_time, waif_window)
            
            for i in range(num_time):
                waif_freq = smoothed_trace[i]
                env = envelope_view[i, j, k]
                
                if env > 1e-8 and waif_freq > 0.0:
                    if i == 0:
                        gradient_val = (envelope_view[i+1, j, k] - env) / dt
                    elif i == num_time - 1:
                        gradient_val = (env - envelope_view[i-1, j, k]) / dt
                    else:
                        gradient_val = (envelope_view[i+1, j, k] - envelope_view[i-1, j, k]) / (2.0 * dt)
                    
                    inst_bw = fabs(gradient_val) / (two_pi * env)
                    
                    inst_freq_sq = waif_freq * waif_freq
                    inst_bw_sq = inst_bw * inst_bw
                    dominant_freq = sqrt(inst_freq_sq + inst_bw_sq)
                    
                    if dominant_freq > nyquist_freq:
                        dominant_freq = nyquist_freq
                    
                    dom_freq_view[i, j, k] = dominant_freq
                else:
                    dom_freq_view[i, j, k] = 0.0
        
        free(freq_trace)
        free(envelope_trace)
        free(smoothed_trace)
    
    if use_masking:
        for j in prange(num_inline, nogil=True, schedule='static'):
            for k in range(num_crossline):
                for i in range(num_time):
                    t_start = i - half_window if i >= half_window else 0
                    t_end = i + half_window + 1 if i + half_window + 1 <= num_time else num_time
                    
                    max_amp = 0.0
                    for t in range(t_start, t_end):
                        abs_val = fabs(data_view[t, j, k])
                        if abs_val > max_amp:
                            max_amp = abs_val
                    
                    if max_amp < threshold:
                        dom_freq_view[i, j, k] = 0.0
    
    return dom_freq_output