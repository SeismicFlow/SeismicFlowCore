from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

# Enable Cython compiler optimizations
Options.docstrings = False
Options.annotate = False
Options.emit_code_comments = False

# MSVC-specific compiler and linker flags
extra_compile_args = [
    "/O2",              # Optimize for speed (O2 is more stable than Ox)
    "/GL",              # Whole program optimization
    "/GF",              # String pooling
    "/Gy",              # Function-level linking
    "/std:c++17",       # C++17 standard
    "/DNDEBUG",         # Disable debug code
    "/fp:fast",         # Fast floating-point model
    "/arch:AVX2",       # Use AVX2 instructions
    "/MP",              # Multi-processor compilation
    "/EHsc",            # Exception handling model
    "/wd4244",          # Disable conversion warnings
    "/wd4267",          # Disable size_t conversion warnings
    "/openmp",          # OpenMP support for parallel loops
]

extra_link_args = [
    "/LTCG",            # Link-time code generation
    "/OPT:REF",         # Eliminate unreferenced data
    "/OPT:ICF",         # Identical COMDAT folding
]

# Define extension
extensions = [
    Extension(
        "chaos",
        ["chaos.pyx"],
        include_dirs=[
            np.get_include(),
            "C:/eigen-3.4.0",
        ],
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
    name="chaos",
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