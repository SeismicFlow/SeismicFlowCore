# setup_xcorr.py - Ultra-optimized setup with FFTW
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np
import os

# Enable Cython compiler optimizations
Options.docstrings = False
Options.annotate = False
Options.emit_code_comments = False

# FFTW paths - CORRECT paths for precompiled Windows DLLs
FFTW_DIR = "C:/fftw-3.3.5"  # Adjust this to where you extracted fftw-3.3.5-dll64.zip

# MSVC-specific compiler and linker flags
extra_compile_args = [
    "/Ox",              # Maximum optimization
    "/GL",              # Whole program optimization
    "/GF",              # String pooling
    "/Gy",              # Function-level linking
    "/std:c++17",       # C++17 standard
    "/DNDEBUG",         # Disable debug code
    "/fp:fast",         # Fast floating-point model
    "/arch:AVX2",       # Use AVX2 instructions
    "/MP",              # Multi-processor compilation
    "/EHsc",            # Exception handling model
    "/W0",              # Disable warnings
    "/openmp",          # OpenMP support
    "/Oi",              # Enable intrinsic functions
    "/Ot",              # Favor fast code
    "/favor:INTEL64",   # Optimize for 64-bit Intel
]

extra_link_args = [
    "/LTCG",           # Link-time code generation
    "/OPT:REF",        # Eliminate unreferenced data
    "/OPT:ICF",        # Identical COMDAT folding
    "/MANIFEST:NO",    # No manifest
]

# Define extension
extensions = [
    Extension(
        "xcorr",
        ["xcorr.pyx"],
        include_dirs=[
            np.get_include(),
            FFTW_DIR,  # Contains fftw3.h
        ],
        library_dirs=[FFTW_DIR],  # Contains libfftw3f-3.lib
        libraries=["libfftw3f-3"],  # Links to libfftw3f-3.lib
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", "1"),
            ("CYTHON_TRACE", "0"),
            ("CYTHON_PROFILE", "0"),
        ],
    )
]

# Setup configuration
setup(
    name="xcorr",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'overflowcheck': False,
            'embedsignature': False,
            'cdivision': True,
            'binding': True,
            'linetrace': False,
            'profile': False,
            'infer_types': True,
            'annotation_typing': False,
        },
        quiet=False,
        annotate=False,
    ),
)