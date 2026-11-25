# SeismicFlowCore

**High-performance, industry-validated Python library for 3D seismic attribute computation.**

SeismicFlowCore implements 27 seismic attributes as compiled Cython extension modules with OpenMP parallelism, FFTW3-accelerated Hilbert transforms, and Eigen-based eigenvalue decomposition — delivering throughput structurally inaccessible to pure-Python implementations, validated against Petrel 2023 on the F3 Netherlands dataset.

---

## Why SeismicFlowCore?

No production-grade, open-source Python library exists for seismic attribute computation that simultaneously provides comprehensive attribute coverage, compiled performance, and quantitative validation against an industry reference platform. SeismicFlowCore addresses all three.

- **27 attributes** spanning complex trace, discontinuity, spectral, amplitude, geometric, and processing utility categories
- **Petrel 2023 validated** — Pearson correlation coefficients from 0.90 to 0.9999 across all attributes on real 3D seismic data (F3 Netherlands)
- **190× faster than a leading open-source seismic Python library** for GST coherence (168 seconds vs 8.9 hours on a 1,000×1,000×1,000 float32 volume)
- **OpenMP parallelism** across all 27 attributes via Cython's nogil prange
- **FFTW3 batched Hilbert transforms** replacing scipy.signal.hilbert for all complex trace attributes
- **Eigen eigenvalue decomposition** for structure-tensor attributes (chaos, GST coherence)
- **NumPy-native API** — input and output are float32 NumPy arrays, no format conversion required

---

## Installation

### Option 1 — Prebuilt wheel (recommended)

Prebuilt wheels for **CPython 3.12 (64-bit)** are attached to each [release](https://github.com/SeismicFlow/SeismicFlowCore/releases). No compiler, FFTW or Eigen installation is needed: the FFTW runtime is bundled in the wheel (together with the OpenMP runtime on Windows), and `numpy` (≥ 1.26) and `scipy` (≥ 1.11) are installed automatically.

| Platform | Wheel | Requirements |
|---|---|---|
| Windows x86-64 | `seismicflowcore-0.1.0-cp312-cp312-win_amd64.whl` | CPU with AVX2 and FMA (Intel Haswell, AMD Zen, or newer). Tested on Windows 10. |
| Linux x86-64 | `seismicflowcore-0.1.0-cp312-cp312-linux_x86_64.whl` | glibc ≥ 2.29, libgomp, CPU with AVX2 and FMA. Tested on Ubuntu 24.04. |

Install the wheel for your platform (a virtual environment is recommended).

Windows:

```
pip install https://github.com/SeismicFlow/SeismicFlowCore/releases/download/v0.1.0/seismicflowcore-0.1.0-cp312-cp312-win_amd64.whl
```

Linux:

```
pip install https://github.com/SeismicFlow/SeismicFlowCore/releases/download/v0.1.0/seismicflowcore-0.1.0-cp312-cp312-linux_x86_64.whl
```

Alternatively, download the `.whl` file from the release page and run `pip install <path to the file>`.

Verify the installation:

```python
import numpy as np
import SeismicFlowCore as sfc

data = np.random.randn(100, 50, 50).astype(np.float32)
dt   = 0.004

print("envelope:    ", sfc.envelope(data).shape)
print("inst_phase:  ", sfc.inst_phase(data).shape)
print("inst_freq:   ", sfc.inst_freq(data, dt).shape)
print("variance:    ", sfc.variance(data).shape)
print("coherence:   ", sfc.coherence(data).shape)
print("agc:         ", sfc.agc(data).shape)
print("dip:         ", sfc.dip(data).shape)
print()
print("OK — all extensions loaded successfully")
```

If every line prints a shape, the installation is complete.

If the import fails:

- **Windows, `DLL load failed while importing ...`** — install the latest Microsoft Visual C++ Redistributable (x64) and try again.
- **Linux, `version 'GLIBC_...' not found`, or "Illegal instruction"** — your glibc is older than 2.29 or your CPU lacks AVX2/FMA. Build from source instead (Option 2).

---

### Option 2 — Build from source

Use this if no wheel fits your platform or Python version, or if you want to modify the code. The sources live in the `SeismicFlowCore/` package folder inside the repository; the build places the compiled modules next to the `.pyx` files in that folder.

#### Windows (MSVC)

Verify that you have **all four** of the following. Do not skip any step.

**1 — Microsoft Visual Studio Build Tools 2022**

Download free from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

During install, check **"Desktop development with C++"**; nothing else is required.

> You do **not** need full Visual Studio. Build Tools alone is sufficient.

Verify after install: open **"x64 Native Tools Command Prompt for VS 2022"** from your Start menu. If it exists, you're good.

**2 — Python 3.12 (64-bit)**

Download from: https://www.python.org/downloads/

Must be the **64-bit installer**. The 32-bit version will fail silently.

Verify:
```
python --version
```
Should print `Python 3.12.x`. Then install the required packages:
```
pip install numpy scipy cython
```

**3 — FFTW3 (Windows prebuilt binaries)**

Required by all complex trace attributes and cross-correlation.

1. Download the 64-bit prebuilt binaries from: https://www.fftw.org/install/windows.html
   → get the file named `fftw-3.3.5-dll64.zip`

2. Extract the contents to `C:/fftw-3.3.5`
   The folder must contain: `libfftw3f-3.dll`, `libfftw3f-3.def`

3. Generate the `.lib` file. Open **"x64 Native Tools Command Prompt for VS 2022"**, navigate to the folder, and run:
   ```
   cd C:/fftw-3.3.5
   lib /machine:x64 /def:libfftw3f-3.def /out:libfftw3f-3.lib
   ```
   You should now see `libfftw3f-3.lib` in that folder.

**4 — Eigen 3.4 (header-only, no compilation needed)**

Required only by `coherence` and `chaos`. All other modules build without it.

Option A — clone with git (version 3.4.0):
```
git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git C:/eigen-3.4.0
```

Option B — download the 3.4.0 zip from https://eigen.tuxfamily.org and extract to `C:/eigen-3.4.0`

No build step is needed. Eigen is header-only — the compiler just needs to find the folder.

**Step 1 — Check the install paths**

The setup scripts expect FFTW in `C:/fftw-3.3.5` and Eigen in `C:/eigen-3.4.0`. If you installed them elsewhere, update the paths: the `FFTW_DIR = "..."` line near the top of each FFTW-based setup file, and the Eigen path in the include list of `setup_chaos.py` and `setup_coherence.py`. To list every line that needs checking, run this in the package folder (`SeismicFlowCore\SeismicFlowCore` after cloning, see Step 2):

```
findstr /n "fftw-3.3.5 eigen-3.4.0" setup_*.py
```

**Step 2 — Build all extensions**

Open a regular PowerShell or Command Prompt (not the VS prompt), then clone the repository:

```
git clone https://github.com/SeismicFlow/SeismicFlowCore
cd SeismicFlowCore\SeismicFlowCore
```

Then run each build line from inside this package folder:

<details>
<summary>Build commands for all 28 modules</summary>

```
python setup_agc.py build_ext --inplace
python setup_app_polarity.py build_ext --inplace
python setup_azimuth.py build_ext --inplace
python setup_chaos.py build_ext --inplace
python setup_coherence.py build_ext --inplace
python setup_cos_phase.py build_ext --inplace
python setup_depth_conversion.py build_ext --inplace
python setup_dip.py build_ext --inplace
python setup_dix_conversion.py build_ext --inplace
python setup_dix_inversion.py build_ext --inplace
python setup_dominant_freq.py build_ext --inplace
python setup_envelope.py build_ext --inplace
python setup_first_derivative.py build_ext --inplace
python setup_gradient_magnitude.py build_ext --inplace
python setup_gsd.py build_ext --inplace
python setup_inst_bandwidth.py build_ext --inplace
python setup_inst_freq.py build_ext --inplace
python setup_inst_phase.py build_ext --inplace
python setup_phase_shift.py build_ext --inplace
python setup_quad_amp.py build_ext --inplace
python setup_reflection_intensity.py build_ext --inplace
python setup_relative_ai.py build_ext --inplace
python setup_rms_amp.py build_ext --inplace
python setup_second_derivative.py build_ext --inplace
python setup_sweetness.py build_ext --inplace
python setup_time_gain.py build_ext --inplace
python setup_variance.py build_ext --inplace
python setup_xcorr.py build_ext --inplace
```

</details>

Each module prints `copying ... ->` on success. If any module fails, the others are unaffected — fix and rerun that one individually.

**Step 3 — Use and verify the build**

The compiled modules sit next to the `.pyx` files in the package folder. Run Python from the repository root, the folder that **contains** the `SeismicFlowCore` package folder (or add the repository root to `PYTHONPATH`):

```
cd ..
python -c "import SeismicFlowCore as sfc; print(sfc.__file__)"
```

At run time Windows must find `libfftw3f-3.dll`. Do one of the following: add `C:/fftw-3.3.5` to your system PATH, copy `libfftw3f-3.dll` into the package folder next to the compiled modules, or add this to the top of any script that uses SeismicFlowCore:

```python
import os
os.add_dll_directory("C:/fftw-3.3.5")
```

> **If you get `DLL load failed while importing`**, Windows cannot find `libfftw3f-3.dll`. The `os.add_dll_directory` fix above solves it without touching system PATH.

Then run the verification snippet from Option 1.

#### Linux (GCC)

Requirements: x86-64 Linux, GCC/G++ with C++17 support, and the three folders below, placed next to `build_linux.py` inside the package folder of the cloned repository:

```
SeismicFlowCore/                 repository root
└── SeismicFlowCore/             package folder
    ├── build_linux.py
    ├── *.pyx
    ├── eigen/Eigen/Dense        Eigen 3.4 headers (header-only)
    ├── fftw/include/fftw3.h     FFTW 3.3.x, single precision, shared library
    ├── fftw/lib/libfftw3f.so*
    └── python312/bin/python3    CPython 3.12 interpreter
```

Eigen (run inside the package folder): `git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git eigen`

FFTW: download the FFTW 3.3.x source, then build it into the `fftw/` folder with `./configure --enable-single --enable-shared --prefix=<path to the package folder>/fftw && make && make install`.

Then install the Python packages and build:

```
git clone https://github.com/SeismicFlow/SeismicFlowCore
cd SeismicFlowCore/SeismicFlowCore
# ... place eigen/, fftw/ and python312/ here, as shown above ...

python312/bin/python3 -m pip install cython numpy scipy setuptools
python312/bin/python3 build_linux.py -j 4
```

The script checks the toolchain and all dependencies before compiling (`--check` runs only this preflight; `--list` prints the module list). Building all 28 modules takes about 3.5 minutes on one core. The compiled `.so` files are placed next to the `.pyx` files and find FFTW through a relative RUNPATH, so keep `fftw/lib` inside the package folder; the package folder can then be moved or copied freely.

To use the build, run Python from the repository root (or add it to `PYTHONPATH`) with any CPython 3.12 that has numpy ≥ 1.26 and scipy ≥ 1.11:

```
cd ..
SeismicFlowCore/python312/bin/python3 -c "import SeismicFlowCore as sfc; print(sfc.__file__)"
```

Then run the verification snippet from Option 1.

---

## Quick Start

```python
import numpy as np
import SeismicFlowCore as sfc

# Seismic volume: (time, inline, crossline), float32
data = np.random.randn(500, 200, 200).astype(np.float32)
dt   = 0.004  # seconds

env     = sfc.envelope(data)
phase   = sfc.inst_phase(data)
freq    = sfc.inst_freq(data, dt)
var     = sfc.variance(data)
coh     = sfc.coherence(data)
agc     = sfc.agc(data)
dip     = sfc.dip(data)
```

All functions accept `(time, inline, crossline)` float32 arrays and return float32 arrays of the same shape unless otherwise noted.

---

## Attributes

### Complex Trace Attributes
| Function | Description |
|---|---|
| `sfc.envelope(data)` | Instantaneous amplitude |
| `sfc.inst_phase(data)` | Instantaneous phase |
| `sfc.inst_freq(data, dt)` | Instantaneous frequency (Vesnaver 2017) |
| `sfc.inst_bandwidth(data, dt)` | Instantaneous bandwidth (Barnes 1993) |
| `sfc.cos_phase(data)` | Cosine of instantaneous phase |
| `sfc.quad_amp(data)` | Quadrature amplitude |
| `sfc.app_polarity(data)` | Apparent polarity |
| `sfc.reflection_intensity(data)` | Reflection intensity |
| `sfc.sweetness(data, dt)` | Sweetness |

### Discontinuity and Structural Continuity
| Function | Description |
|---|---|
| `sfc.variance(data)` | Semblance-based variance (Marfurt et al. 1998) |
| `sfc.coherence(data)` | GST coherence (Eigen eigenvalue decomposition) |
| `sfc.chaos(data)` | Chaos attribute |

### Spectral and Frequency
| Function | Description |
|---|---|
| `sfc.dominant_freq(data, dt)` | Dominant frequency (Barnes 1993) |
| `sfc.gsd(data)` | Generalized spectral decomposition (Gabor wavelet) |

### Amplitude and Reflectivity
| Function | Description |
|---|---|
| `sfc.rms_amp(data)` | RMS amplitude |
| `sfc.relative_ai(data, low_cut_freq, dt_ms)` | Relative acoustic impedance |
| `sfc.agc(data)` | Automatic gain control (Petrel TraceAGC) |
| `sfc.time_gain(data)` | Time gain (spherical divergence correction) |

### Geometric and Structural
| Function | Description |
|---|---|
| `sfc.dip(data)` | Local structural dip (4th-order finite difference) |
| `sfc.gradient_magnitude(data)` | Gradient magnitude |
| `sfc.azimuth(data)` | Local structural azimuth |

### Processing Utilities
| Function | Description |
|---|---|
| `sfc.phase_shift(data, degrees)` | Constant phase rotation |
| `sfc.first_derivative(data)` | First derivative (central difference) |
| `sfc.second_derivative(data)` | Second derivative (central difference) |
| `sfc.xcorr(data1, data2)` | FFT-accelerated normalized cross-correlation |
| `sfc.dix_conversion(data, dt)` | Interval → stacking velocity (Dix 1955) |
| `sfc.dix_inversion(data, dt)` | Stacking → interval velocity (inverse Dix) |
| `sfc.depth_conversion(vel, dt, data)` | Time-to-depth conversion |

---

## Validation Against Petrel 2023

All 27 attributes validated on the F3 Netherlands 3D seismic dataset against Petrel 2023 reference outputs using Pearson correlation (r), 3D SSIM, KS distributional distance (D_KS), and NRMSE.

### Complex Trace Attributes
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| Envelope | Envelope | 0.9972 | 0.9909 | 0.0013 | 0.0181 |
| Instantaneous Phase | Instantaneous Phase | 0.9507 | 0.9502 | 0.0052 | 0.0945 |
| Instantaneous Frequency | Instantaneous Frequency | 0.9323 | 0.8751 | 0.0873 | 0.0969 |
| Instantaneous Bandwidth | Instantaneous Bandwidth | 0.9369 | 0.9381 | 0.0141 | 0.0882 |
| Cosine of Phase | Cosine of Instantaneous Phase | 0.9967 | 0.9891 | 0.0044 | 0.0288 |
| Quadrature Amplitude | Quadrature Amplitude | 0.9960 | 0.9889 | 0.0087 | 0.0194 |
| Apparent Polarity | Apparent Polarity | 0.9020 | 0.8812 | 0.0018 | 0.0896 |
| Reflection Intensity | Reflection Intensity | 0.9864 | 0.9156 | 0.0102 | 0.0397 |
| Sweetness | Sweetness | 0.9901 | 0.9658 | 0.0214 | 0.0350 |

### Discontinuity and Structural Continuity
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| Variance | Variance Edge | 0.9997 | 0.9985 | 0.0007 | 0.0054 |
| Chaos | Chaos | 0.9827 | 0.9583 | 0.0127 | 0.0426 |

### Spectral and Frequency
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| Dominant Frequency | Dominant Frequency | 0.9047 | 0.8602 | 0.0018 | 0.1225 |
| Spectral Decomposition | Generalized Spectral Decomposition (Convolution) | 0.9920 | 0.9815 | 0.0008 | 0.0463 |

### Amplitude and Reflectivity
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| RMS Amplitude | RMS Amplitude | 0.9935 | 0.9870 | 0.0021 | 0.0273 |
| Relative Acoustic Impedance | Relative Acoustic Impedance 10Hz | 0.9786 | 0.9205 | 0.0043 | 0.0450 |
| AGC | AGC | 0.9962 | 0.9848 | 0.0263 | 0.0234 |
| Time Gain | Time Gain | 0.9937 | 0.9851 | 0.0153 | 0.0247 |

### Geometric and Structural
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| Structural Dip | Local Structural Dip (Gradient) | 0.9920 | 0.9815 | 0.0008 | 0.0463 |
| Gradient Magnitude | Gradient Magnitude | 0.9832 | 0.9792 | 0.0050 | 0.0440 |
| Azimuth Gradient | Local Structural Azimuth (Gradient) | 0.9922 | 0.9900 | 0.0010 | 0.0370 |

### Processing Utilities
| Attribute | Petrel Name | r | SSIM | D_KS | NRMSE |
|---|---|---|---|---|---|
| Phase Shift 90° | Phase Shift | 0.9999 | 0.9996 | 0.0006 | 0.0029 |
| First Derivative | First Derivative | 0.9866 | 0.9823 | 0.0088 | 0.0351 |
| Second Derivative | Second Derivative | 0.9851 | 0.9836 | 0.0009 | 0.0365 |
| Cross-Correlation | Cross-Correlation | 0.9992 | 0.9982 | 0.0044 | 0.0063 |
| Dix Conversion | Dix Equation (Forward/Inverse) | 0.9956 | 0.9925 | 0.0357 | 0.0551 |
| Time-to-Depth | Time-to-Depth Conversion | 0.9722 | 0.9508 | 0.0490 | 0.0395 |

Note: GST coherence (r = 0.9827) was excluded from the table above as it has no direct Petrel equivalent; validation methodology for this attribute is described in the accompanying manuscript.

---

## Performance

Benchmarked on Intel Core i7-7700K (8 threads, AVX2) on a 1,000×1,000×1,000 float32 volume:

| Attribute | SeismicFlowCore | Reference library | Speedup |
|---|---|---|---|
| GST Coherence | 168 s | 32,086 s | **190×** |
| Quadrature Amplitude | 5.9 s | 62.2 s | **10.5×** |
| Envelope | 6.1 s | 14.5 s | **2.4×** |
| Instantaneous Phase | 14.4 s | 29.1 s | **2.0×** |

---

## Architecture

Three implementation tiers selected on engineering grounds:

- **Tier 1 — Full Cython + OpenMP**: structure-tensor attributes (chaos, GST coherence), gradient-based attributes, discontinuity attributes
- **Tier 2 — FFTW3 + Cython + OpenMP**: all 9 complex trace attributes via batched per-inline DFT plans with thread-safe `fftwf_execute_dft`
- **Tier 3 — Cython bottleneck + SciPy**: relative acoustic impedance (parallelized integration in Cython, zero-phase Butterworth filtering via `sosfiltfilt`)

---

## Testing and Reproducibility

[TESTING.md](TESTING.md) documents the build environments, the test procedure, the numerical tolerance and the per-function results. In summary, on a seeded synthetic test cube (128 × 16 × 16 samples):

- All 28 extension modules compile on Linux (GCC 13.3) and Windows (MSVC), and pass the import, execution, finite-output and repeatability checks.
- Output is bit-identical with `OMP_NUM_THREADS=1` and `4` on both platforms (28 / 28).
- The Linux and Windows wheels were each installed into a clean environment and passed the full suite (28 / 28).
- Windows and Linux outputs agree within the stated tolerance for all 28 functions; the largest difference is 2.7e-06 of the dynamic range, and 6 of 28 outputs are bitwise identical.
- A strict-IEEE rebuild differs from the fast-math build by at most 8.5e-06 of the dynamic range.
- The suite also passes with NumPy 1.26.4 and SciPy 1.13.1.

---

## Citation

If you use SeismicFlowCore in your research, please cite:

> Anonymous. (2026). *SeismicFlowCore: A High-Performance, Industry-Validated Open-Source Python Library for 3D Seismic Attribute Computation*. Manuscript submitted for publication.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
