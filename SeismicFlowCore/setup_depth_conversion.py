from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = False
Options.annotate = False
Options.emit_code_comments = False

extra_compile_args = [
    "/Ox", "/GL", "/GF", "/Gy", "/DNDEBUG", "/fp:fast",
    "/arch:AVX2", "/MP", "/W0", "/openmp", "/Oi", "/Ot", "/favor:INTEL64",
]

extra_link_args = ["/LTCG", "/OPT:REF", "/OPT:ICF", "/MANIFEST:NO"]

extensions = [
    Extension(
        "depth_conversion",
        ["depth_conversion.pyx"],
        include_dirs=[np.get_include()],
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

setup(
    name="depth_conversion",
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