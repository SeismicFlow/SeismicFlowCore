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
from scipy.signal import butter, sosfiltfilt
cnp.import_array()

def relative_acoustic_impedance(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    float low_cut_freq,
    float sampling_interval_ms
):
    """
    Relative acoustic impedance computation by trace integration and high-pass filtering.
    
    Physics: Integration of seismic trace approximates relative acoustic impedance.
    High-pass filtering removes low-frequency drift from integration.
    """
    cdef:
        int num_time = data.shape[0]
        int num_inline = data.shape[1]
        int num_crossline = data.shape[2]
        int total_traces = num_inline * num_crossline
        cnp.ndarray[cnp.float32_t, ndim=3] integrated = np.empty_like(data)
        float sample_rate = 1000.0 / sampling_interval_ms
        float nyquist = sample_rate / 2.0
        float normalized_cutoff = min(low_cut_freq / nyquist, 0.95)
    
    cdef cnp.float32_t[:, :, :] data_view = data
    cdef cnp.float32_t[:, :, :] integrated_view = integrated
    
    cdef int i, j, k, trace_idx
    cdef float cumsum
    
    # Step 1: Parallel cumulative sum with better loop structure
    # Flattening the j,k loops can sometimes improve scheduling
    with nogil:
        for trace_idx in prange(total_traces, schedule='static'):
            j = trace_idx // num_crossline
            k = trace_idx % num_crossline
            
            cumsum = 0.0
            for i in range(num_time):
                cumsum = cumsum + data_view[i, j, k]
                integrated_view[i, j, k] = cumsum
    
    # Step 2 & 3: Use SciPy for filtering (already optimized)
    sos = butter(4, normalized_cutoff, btype='high', output='sos').astype(np.float32)
    filtered = sosfiltfilt(sos, integrated, axis=0).astype(np.float32)
    
    return filtered