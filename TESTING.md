# SeismicFlowCore - cross-platform build and test record

This file documents how the 28 extension modules are built on Windows (MSVC) and Linux (GCC), the test suite
that qualifies a build (`tests/run_suite.py`), and the numerical agreement between platforms and build modes.
Precompiled wheels for Windows and Linux are available from the GitHub repository; to build from source,
use the scripts described below.

## 1. Build environments used for the results in the paper

| Item | Linux | Windows |
|---|---|---|
| Operating system | Ubuntu 24.04 (glibc 2.39), x86-64 | Windows 10 (19045), x86-64 |
| Compiler | GCC 13.3.0 | MSVC |
| Python / NumPy | CPython 3.12.9 / NumPy 2.5.5 | CPython 3.12.10 / NumPy 2.4.6 |
| Libraries | FFTW 3.3.10 (single precision), Eigen 3.4.0 | FFTW 3.3.5 (single precision), Eigen 3.4.0 |
| Flags | -O3 -mavx2 -mfma -flto -fopenmp, fast-math | /Ox /GL /fp:fast /arch:AVX2, OpenMP |
| Result | 28 of 28 modules, exit status 0 (3 min 27 s on one core) | 28 of 28 modules |

## 2. How to build

Linux: place `eigen/` (Eigen 3.4 headers), `fftw/` (FFTW 3.3 single precision) and a CPython 3.12 interpreter
in `python312/` beside `build_linux.py`, then run

    python312/bin/python3 -m pip install cython numpy scipy setuptools
    ./build_linux.py -j 4

The script checks the toolchain (Python 3.12, Cython, NumPy, setuptools, a C++17 compiler, Eigen, FFTW and all
28 sources) before compiling. Windows: run the `setup_<module>.py` scripts with `build_ext --inplace` from an
MSVC developer prompt, with FFTW and Eigen available as described in each setup script.

## 3. How to test

    python tests/run_suite.py run --pkg-dir <build directory> --json out.json --csv out.csv
    python tests/run_suite.py run --pkg-dir <build directory> --compare tests/linux_reference_outputs.npz

The suite reads seeded input cubes from `tests/test_inputs.npz` and refuses to run if their SHA-256 differs from
the reference. Each function runs in its own process. Checks: the extension imports; the function runs; the
output is finite and non-trivial; two calls give bit-identical output without modifying the input;
OMP_NUM_THREADS=1 and OMP_NUM_THREADS=4 give identical output; and, with --compare, the output agrees with the
reference within the tolerance below.

Tolerance (fixed before testing): |a - b| <= 1e-5 * max|ref| + 1e-4 * |ref| per element; NaN/Inf positions must
match; angle outputs (instantaneous phase, azimuth) are compared with wrap-around.

## 4. Compatibility statistics

| Check | Result |
|---|---|
| Modules compiled (Linux) | 28 / 28 |
| Extension modules import; package import exposes all public functions | 28 / 28 |
| Function runs on the test cube; output finite and non-trivial | 28 / 28 |
| Bit-identical repeat call; input array unmodified | 28 / 28 |
| Bit-identical output, OMP_NUM_THREADS = 1 vs. 4 (Linux and Windows) | 28 / 28 |
| Linux wheel installed into a clean environment, full suite | 28 / 28 |
| Windows wheel installed into a clean environment, full suite incl. comparison with Linux reference | 28 / 28 |
| NumPy 1.26.4 + SciPy 1.13.1 environment, full suite within tolerance | 28 / 28 |
| Strict-IEEE rebuild vs. fast-math build: max difference / dynamic range | 8.5e-06 |
| Windows vs. Linux within tolerance; max difference / dynamic range | 28 / 28; 2.7e-06 (6 of 28 bitwise identical) |

## 5. Test cases (one per public function)

Test data: seeded synthetic cube of 128 x 16 x 16 samples (time, inline, crossline), dt = 4 ms: a 30 Hz Ricker
wavelet convolved with dipping reflectivity plus 1% noise (seis); seis delayed by 3 samples plus noise (seis2);
a smooth interval velocity of 1435-2835 m/s (vel) and the matching RMS velocity (vrms).

| Name | Function | Inputs | Parameters | Strict vs fast (max diff / range) | Windows vs Linux (max diff / range) |
|---|---|---|---|---|---|
| agc | trace_agc | seis | window_size=9 | 1.5e-07 | 1.5e-07 |
| app_polarity | apparent_polarity | seis | threshold=1e-06, use_masking=True, window_size=33 | 0.0e+00 | 0.0e+00 |
| azimuth | local_structural_azimuth | seis | defaults | 0.0e+00 | 7.2e-07 |
| chaos | chaos | seis | sigma_t=1.5, sigma_il=1.5, sigma_xl=1.5, truncate=4.0 | 6.1e-07 | 6.1e-07 |
| coherence | gst_coherence | seis | window=(5, 5, 5), sigma=1.0 | 3.7e-07 | 4.4e-07 |
| cos_phase | cosine_of_phase | seis | defaults | 0.0e+00 | 9.4e-07 |
| depth_conversion | time_to_depth_conversion | vel, 0.004, seis | defaults | 8.5e-06 | 0.0e+00 |
| dip | local_structural_dip | seis | defaults | 8.5e-08 | 1.1e-06 |
| dix_conversion | interval_to_stacking_velocity | vel, 0.004 | defaults | 1.6e-07 | 1.1e-07 |
| dix_inversion | stacking_to_interval_velocity | vrms, 0.004 | defaults | 2.1e-06 | 0.0e+00 |
| dominant_freq | dominant_frequency | seis, 0.004 | threshold=1e-06, use_masking=True, window_size=33, waif_window=5 | 1.2e-07 | 1.8e-07 |
| envelope | envelope | seis | defaults | 9.8e-08 | 9.8e-08 |
| first_derivative | first_derivative | seis | defaults | 0.0e+00 | 7.4e-08 |
| gradient_magnitude | gradient_magnitude | seis | defaults | 6.2e-08 | 1.2e-07 |
| gsd | generalized_spectral_decomposition | seis | frequency=30.0, num_cycles=2, phase_deg=90.0 | 2.5e-07 | 2.5e-07 |
| inst_bandwidth | instantaneous_bandwidth | seis, 0.004 | threshold=1e-06, use_masking=True, window_size=33 | 1.6e-07 | 2.6e-07 |
| inst_freq | instantaneous_frequency | seis, 0.004 | threshold=1e-06, use_masking=True, window_size=33, waif_window=5 | 2.5e-07 | 2.5e-07 |
| inst_phase | instantaneous_phase | seis | threshold=1e-06, use_masking=True, window_size=33 | 0.0e+00 | 8.5e-08 |
| phase_shift | phase_shift | seis, 45.0 | defaults | 1.1e-07 | 0.0e+00 |
| quad_amp | quadrature_amplitude | seis | defaults | 0.0e+00 | 0.0e+00 |
| reflection_intensity | reflection_intensity | seis | defaults | 1.8e-07 | 1.8e-07 |
| relative_ai | relative_acoustic_impedance | seis, 10.0, 4.0 | defaults | 2.5e-06 | 0.0e+00 |
| rms_amp | rms_amplitude | seis | window_size=9 | 8.2e-08 | 8.1e-08 |
| second_derivative | second_derivative | seis | defaults | 0.0e+00 | 1.3e-07 |
| sweetness | sweetness | seis, 0.004 | defaults | 2.1e-07 | 2.7e-06 |
| time_gain | time_gain | seis | gain_exponent=1.5, sample_rate=0.004 | 0.0e+00 | 1.3e-07 |
| variance | variance_edge_cython | seis | inline_range=3, crossline_range=3, vertical_smoothing=15 | 6.4e-07 | 6.4e-07 |
| xcorr | cross_correlation | seis, seis2 | max_lag=32, sample_rate=4.0 | 2.4e-07 | 2.4e-07 |

## 6. Build requirements and resolutions

| # | Requirement | Resolution |
|---|---|---|
| 1 | build_linux.py re-executes itself under the CPython 3.12 interpreter in python312/. | Use a standalone CPython placed beside the script, as described in the script header. |
| 2 | relative_ai uses scipy.signal (Butterworth SOS filtering). | SciPy is a declared runtime dependency; the wheel installs it (from source: pip install scipy). |
| 3 | Twelve FFTW-based modules load the single-precision FFTW runtime. | The Linux build places it in fftw/lib beside the modules and sets an $ORIGIN-relative RUNPATH; LD_LIBRARY_PATH also works. On Windows, keep the FFTW DLL beside the modules or on PATH. |
| 4 | Relocated copies of the package must find the FFTW runtime. | The relative RUNPATH resolves it in any location (verified on a relocated copy, 28/28). |
| 5 | Minimum platform of a Linux build. | glibc >= 2.29, libgomp with GOMP_4.5, CPU with AVX2 and FMA; for other targets, rebuild from source with build_linux.py. |
| 6 | Line endings of the build script. | build_linux.py is stored with LF line endings. |

