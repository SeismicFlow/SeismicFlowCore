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
from libc.math cimport pow

cnp.import_array()

def time_gain(cnp.ndarray[cnp.float32_t, ndim=3] data, float gain_exponent=1.5, float sample_rate=0.004):
    """
    Time Gain: Applies exponential gain correction to compensate for amplitude decay.
    
    Implements Petrel's Time Gain algorithm:
    output(t) = input(t) * (1 + t)^g
    
    where:
    - t is the two-way travel time (TWT) in seconds
    - g is the gain exponent defined by the user
    - g = 0 yields the input trace (no gain)
    - g values between 1.5 to 2.0 are typical for spherical divergence correction
    
    Based on scientific literature:
    - Two-way travel time (TWT) is measured in seconds (SEG Wiki, ResearchGate)
    - Seismic data is typically sampled at 2ms (0.002s) or 4ms (0.004s) intervals
    - Power gain compensates for spherical divergence and attenuation (SEG Wiki)
    - Typical power values range from 1.8 to 2.4 for divergence correction (Xsgeo)
    - t² (g=2) gain is commonly used: gain = t² for geometric spreading (Stanford SEP)
    - The gain increases with time to compensate for amplitude decay at depth
    
    Parameters:
    -----------
    data : ndarray of float32, shape (time, inline, crossline)
        Input seismic data
    gain_exponent : float, default=1.5
        Gain exponent 'g' in the formula (1 + t)^g
        Valid range: 0.0 to 10.0
        Typical values: 1.5 to 2.0
    sample_rate : float, default=0.004
        Sample interval in seconds (e.g., 0.002 for 2ms, 0.004 for 4ms)
        
    Returns:
    --------
    output : ndarray of float32, shape (time, inline, crossline)
        Time gain processed seismic data
        
    References:
    -----------
    SEG Wiki: Geometric spreading correction and Gain applications
    Xsgeo: Gain corrections in seismic processing (power gain 1.8-2.4)
    Stanford SEP: t-squared divergence correction
    ResearchGate: Two-way travel time (TWT) in milliseconds
    """
    cdef int num_samples = data.shape[0]
    cdef int num_inline = data.shape[1]
    cdef int num_crossline = data.shape[2]
    
    # Allocate output array
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty_like(data)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] output_view = output
    
    cdef int i, j, k
    cdef float time_seconds, gain_factor
    
    # Apply time gain: output(t) = input(t) * (1 + t)^g
    # where t is the two-way travel time in seconds
    for i in prange(num_samples, nogil=True, schedule='static'):
        # Convert sample index to time in seconds
        time_seconds = <float>i * sample_rate
        
        # Compute gain factor: (1 + t)^g
        gain_factor = <float>pow(1.0 + time_seconds, gain_exponent)
        
        for j in range(num_inline):
            for k in range(num_crossline):
                output_view[i, j, k] = data_view[i, j, k] * gain_factor
    
    return output