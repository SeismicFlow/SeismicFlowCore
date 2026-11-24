# setup_variance.py - Maximum Performance Build Configuration
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Platform-specific optimization flags
if sys.platform == 'win32':
    # Windows MSVC - Maximum optimization
    extra_compile_args = [
        '/O2',              # Optimize for maximum speed
        '/Ot',              # Favor fast code
        '/Ox',              # Full optimization
        '/GL',              # Whole program optimization
        '/openmp',          # OpenMP parallel processing
        '/arch:AVX2',       # AVX2 SIMD instructions
        '/fp:fast',         # Fast floating-point (allows reordering)
        '/Qpar',            # Auto-parallelization hint
        '/favor:AMD64',     # Optimize for 64-bit
        '/wd4244',          # Suppress C4244: conversion warnings (Cython generates these safely)
    ]
    extra_link_args = [
        '/LTCG',            # Link-time code generation
        '/OPT:REF',         # Remove unreferenced functions
        '/OPT:ICF',         # Identical COMDAT folding
    ]
    
elif 'linux' in sys.platform:
    # Linux GCC - Maximum optimization
    extra_compile_args = [
        '-O3',                          # Maximum optimization
        '-march=native',                # Use all CPU features
        '-mtune=native',                # Tune for local CPU
        '-ffast-math',                  # Fast math (aggressive)
        '-fopenmp',                     # OpenMP
        '-ftree-vectorize',             # Auto-vectorization
        '-funroll-loops',               # Loop unrolling
        '-fomit-frame-pointer',         # Extra register
        '-fno-math-errno',              # Faster math
        '-funsafe-math-optimizations',  # Aggressive math opts
    ]
    extra_link_args = ['-fopenmp', '-flto']  # LTO = Link Time Optimization
    
else:
    # macOS Clang
    extra_compile_args = [
        '-O3',
        '-march=native',
        '-ffast-math',
        '-Xpreprocessor', '-fopenmp',  # Clang OpenMP needs special handling
        '-ftree-vectorize',
    ]
    extra_link_args = ['-lomp']  # Link OpenMP library

extensions = [
    Extension(
        "variance",
        ["variance.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
]

setup(
    name="variance",
    version="2.0.0",
    description="Ultra-optimized variance edge detection for seismic data",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,         # Disable None checks for extra speed
            'embedsignature': False,     # Smaller binary
            'overflowcheck': False,      # No overflow checking
        },
        annotate=False,  # Set True to see optimization opportunities
        nthreads=4,      # Parallel Cython compilation
    ),
)