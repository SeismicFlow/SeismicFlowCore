#!/usr/bin/env python3
"""
build_linux.py -- SeismicFlow Cython/C++ extension build for Linux (no-root CentOS x86_64)

Linux port of the 28 per-module Windows setup_*.py files (MSVC), one extension
module each.  Replaces them with one script; the per-module flags, include
dirs, libraries and Cython directives below are transcribed PER FILE from those
setup files, not flattened into a common template.

Layout expected (everything sits next to this script, alongside the .pyx files):

    <this dir>/
        build_linux.py          <- you are here
        *.pyx                   <- 28 sources
        eigen/Eigen/Dense       <- header-only  (mirrors Windows C:/eigen-3.4.0)
        fftw/include/fftw3.h    <- autotools layout
        fftw/lib/libfftw3f.so*
        python312/bin/python3   <- standalone CPython 3.12 the app actually runs on

Usage:
    ./build_linux.py                     # build everything
    ./build_linux.py --only agc dip
    ./build_linux.py --list              # print the per-module port matrix and exit
    ./build_linux.py --check             # preflight only, compile nothing
    ./build_linux.py --clean             # remove generated .c/.cpp, build dirs, .so
    ./build_linux.py -j 8                # parallelism (default: cpu_count)

Output: <name>.cpython-312-x86_64-linux-gnu.so next to the .pyx files.  The
platform tag differs from the Windows .cp312-win_amd64.pyd tag, so the .so and
.pyd files coexist in the same folder and CPython picks the right one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------

# /arch:AVX2 on Windows is an explicit ISA request, not "whatever this box has".
# "avx2"   -> -mavx2 -mfma       (faithful to the Windows baseline; portable across
#                                 any Haswell-or-newer host, which is what the .pyd
#                                 files already assume)
# "native" -> -march=native -mtune=native  (what setup_variance.py chose in its own
#                                 Linux branch; can emit AVX-512 on a newer build
#                                 host and SIGILL on an older one)
MARCH_POLICY = "avx2"

# /GL + /LTCG -> -flto.  Correct mapping, but LTO across Cython-generated C++ is
# the first thing to turn off if you hit ICEs or bogus link errors on an older
# toolchain.  Setting this False drops -flto from both compile and link.
ENABLE_LTO = True

# The FFTW-linked modules need to find libfftw3f.so.3 at *import* time.  Windows
# resolved this via PATH/DLL search; Linux needs an rpath baked into the .so.
# $ORIGIN keeps it relocatable as long as fftw/ travels with the .so files.
EMBED_RPATH = True

CC = os.environ.get("CC", "gcc")
CXX = os.environ.get("CXX", "g++")

# ---------------------------------------------------------------------------
# Bundled dependency locations -- all relative to this file, never system-wide
# ---------------------------------------------------------------------------

EIGEN_DIR = HERE / "eigen"                 # the folder itself is the include dir
FFTW_DIR = HERE / "fftw"
FFTW_INCLUDE = FFTW_DIR / "include"
FFTW_LIB = FFTW_DIR / "lib"
PY_DIR = HERE / "python312"
PY_BIN = PY_DIR / "bin" / "python3"


def _running_under_bundled_python() -> bool:
    """True if this process is the bundled interpreter. Compared by install prefix, not by
    resolved executable path, so it also works when python312/bin/python3 is a symlink
    (e.g. a venv) and would otherwise resolve to the same file as the system python."""
    return Path(sys.prefix).resolve() == PY_DIR.resolve()

# Windows linked "libfftw3f-3" (the versioned import lib for libfftw3f-3.dll).
# A standard autotools FFTW build on Linux produces libfftw3f.so -> setuptools
# wants the bare stem and adds lib/.so itself.  Do NOT carry the Windows name over.
FFTW_LIBNAME = "fftw3f"

# ---------------------------------------------------------------------------
# MSVC -> GCC/Clang flag translation
#
# Every MSVC flag that appears anywhere in the 28 setup files is listed here.
# A flag mapping to [] is a no-op on GCC (default behaviour or build-system
# concern).  A flag mapping to DROP is one with no faithful equivalent -- it is
# discarded and reported in the "translation notes" section of the summary.
# A flag NOT in this table is a hard error: the script stops and asks rather
# than guessing.
# ---------------------------------------------------------------------------

DROP = object()

COMPILE_MAP: dict[str, object] = {
    # optimisation level
    "/Ox": ["-O3"],                       # max speed
    "/O2": ["-O2"],                       # setup_chaos deliberately stepped down from
                                          # /Ox ("O2 is more stable"); kept
    "/Ob2": ["-finline-functions"],
    "/Oi": [],                            # intrinsics: GCC builtins are on at -O2+
    "/Ot": [],                            # favour speed: implied by -O2/-O3
    # codegen / layout
    "/GL": ["-flto"],                     # whole-program opt, pairs with /LTCG
    "/GF": [],                            # string pooling is default on GCC
    "/Gy": ["-ffunction-sections"],       # pairs with /OPT:REF -> --gc-sections
    "/GS-": ["-fno-stack-protector"],
    # language / defines
    "/std:c++17": ["-std=c++17"],
    "/DNDEBUG": ["-DNDEBUG"],
    "/EHsc": [],                          # C++ exceptions are on by default in g++
    # floating point / ISA
    "/fp:fast": ["-ffast-math"],
    "/arch:AVX2": ["-mavx2", "-mfma"],    # rewritten by MARCH_POLICY below
    "/favor:INTEL64": ["-mtune=intel"],
    # parallelism
    "/openmp": ["-fopenmp"],              # also forced onto the link line
    "/MP": [],                            # per-file parallelism -> our own job pool
    # diagnostics
    "/W0": ["-w"],
    "/wd4244": ["-w"],                     # chaos: MSVC warning numbers have no GCC analogue; chaos gets
    "/wd4267": ["-w"],                     # -w like every other module (Windows /W0), silencing Eigen header noise
}

LINK_MAP: dict[str, object] = {
    "/LTCG": ["-flto"],
    "/OPT:REF": ["-Wl,--gc-sections"],
    "/OPT:ICF": DROP,                      # see translation note
    "/MANIFEST:NO": DROP,                  # Windows-only concept
}

# Reported once, at the end of the run.  Each entry: (flag, modules, explanation)
TRANSLATION_NOTES = {
    "/OPT:ICF": (
        "Identical COMDAT folding. GNU ld has no ICF; gold and lld offer --icf=all. "
        "Dropped -- it is a binary-size optimisation only. Add -fuse-ld=gold -Wl,--icf=all "
        "if you want it back and gold is available."
    ),
    "/wd4244": (
        "MSVC warning number (narrowing conversion). Mapped to -w so chaos builds with the "
        "same (silent) diagnostics as every other module."
    ),
    "/wd4267": (
        "MSVC warning number (size_t truncation). Mapped to -w, same as /wd4244."
    ),
    "/MANIFEST:NO": ("Windows side-by-side manifest suppression. Meaningless on ELF; dropped."),
}

# ---------------------------------------------------------------------------
# Per-module build definitions, transcribed from the individual setup files
# ---------------------------------------------------------------------------

# --- MSVC compile-flag profiles actually observed in the dump --------------
# P_CPP_INTEL : the 14 C++ modules (agc, dominant_freq, ...)
# P_CPP_PLAIN : coherence only -- identical but no /favor:INTEL64
# P_C_INTEL   : the 11 pure-C modules (azimuth, dip, ...) -- no /std:c++17, no /EHsc
# P_CHAOS     : setup_chaos.py only  -- /O2, /wd*, no /W0, no /Oi /Ot /favor
P_CPP_INTEL = ["/Ox", "/GL", "/GF", "/Gy", "/std:c++17", "/DNDEBUG", "/fp:fast",
               "/arch:AVX2", "/MP", "/EHsc", "/W0", "/openmp", "/Oi", "/Ot", "/favor:INTEL64"]
P_CPP_PLAIN = ["/Ox", "/GL", "/GF", "/Gy", "/std:c++17", "/DNDEBUG", "/fp:fast",
               "/arch:AVX2", "/MP", "/EHsc", "/W0", "/openmp", "/Oi", "/Ot"]
P_C_INTEL = ["/Ox", "/GL", "/GF", "/Gy", "/DNDEBUG", "/fp:fast", "/arch:AVX2",
             "/MP", "/W0", "/openmp", "/Oi", "/Ot", "/favor:INTEL64"]
P_CHAOS = ["/O2", "/GL", "/GF", "/Gy", "/std:c++17", "/DNDEBUG", "/fp:fast",
           "/arch:AVX2", "/MP", "/EHsc", "/wd4244", "/wd4267", "/openmp"]

L_STD = ["/LTCG", "/OPT:REF", "/OPT:ICF", "/MANIFEST:NO"]
L_CHAOS = ["/LTCG", "/OPT:REF", "/OPT:ICF"]

# --- Cython directive sets -------------------------------------------------
D_STD = {
    "language_level": "3", "boundscheck": False, "wraparound": False,
    "initializedcheck": False, "nonecheck": False, "overflowcheck": False,
    "embedsignature": False, "cdivision": True, "binding": True,
    "linetrace": False, "profile": False, "infer_types": True,
    "annotation_typing": False,
}
D_VAR = {  # setup_variance.py
    "language_level": 3, "boundscheck": False, "wraparound": False,
    "cdivision": True, "initializedcheck": False, "nonecheck": False,
    "embedsignature": False, "overflowcheck": False,
}

M_STD = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
         ("CYTHON_WITHOUT_ASSERTIONS", "1"),
         ("CYTHON_TRACE", "0"),
         ("CYTHON_PROFILE", "0")]
M_NUMPY_ONLY = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# setup_variance.py already carried a hand-written Linux branch. It is used
# verbatim rather than re-deriving it from the MSVC side -- the author already
# made the call for this module.
LNX_VARIANCE_C = ["-O3", "-march=native", "-mtune=native", "-ffast-math", "-fopenmp",
                  "-ftree-vectorize", "-funroll-loops", "-fomit-frame-pointer",
                  "-fno-math-errno", "-funsafe-math-optimizations"]
LNX_VARIANCE_L = ["-fopenmp", "-flto"]


def M(name, *, lang=None, eigen=False, fftw=False, msvc_c=None, msvc_l=None,
      directives=None, macros=None, linux_c=None, linux_l=None, options=None,
      annotate=False, origin=""):
    # NB: `is not None`, not `or` -- setup_variance.py sets no Cython Options at
    # all, and an empty dict is falsy. Using `or` here would silently give that
    # module the other 27 files' Options.docstrings = False.
    default_options = {"docstrings": False, "annotate": False, "emit_code_comments": False}
    return dict(name=name, source=f"{name}.pyx", lang=lang, eigen=eigen, fftw=fftw,
                msvc_c=msvc_c, msvc_l=msvc_l,
                directives=directives if directives is not None else D_STD,
                macros=macros if macros is not None else M_STD,
                linux_c=linux_c, linux_l=linux_l,
                options=options if options is not None else default_options,
                annotate=annotate, origin=origin)


CPP = dict(lang="c++", msvc_c=P_CPP_INTEL, msvc_l=L_STD)

MODULES = [
    # ---- C++ + FFTW (include fftw3.h, link libfftw3f) ----------------------
    M("agc", **CPP, fftw=True, origin="setup_agc.py"),
    M("app_polarity", **CPP, fftw=True, origin="setup_app_polarity.py"),
    M("cos_phase", **CPP, fftw=True, origin="setup_cos_phase.py"),
    M("envelope", **CPP, fftw=True, origin="setup_envelope.py"),
    M("gsd", **CPP, fftw=True, origin="setup_gsd.py"),
    M("inst_bandwidth", **CPP, fftw=True, origin="setup_inst_bandwidth.py"),
    M("inst_freq", **CPP, fftw=True, origin="setup_inst_freq.py"),
    M("inst_phase", **CPP, fftw=True, origin="setup_inst_phase.py"),
    M("phase_shift", **CPP, fftw=True, origin="setup_phase_shift.py"),
    M("quad_amp", **CPP, fftw=True, origin="setup_quad_amp.py"),
    M("reflection_intensity", **CPP, fftw=True, origin="setup_reflection_intensity.py"),
    M("sweetness", **CPP, fftw=True, origin="setup_sweetness.py"),
    M("xcorr", **CPP, fftw=True, origin="setup_xcorr.py"),

    # ---- C++ + Eigen + FFTW ------------------------------------------------
    M("dominant_freq", **CPP, eigen=True, fftw=True, origin="setup_dominant_freq.py"),

    # ---- C++ + Eigen only (header-only: include_dirs, nothing to link) -----
    M("coherence", lang="c++", eigen=True, msvc_c=P_CPP_PLAIN, msvc_l=L_STD,
      origin="setup_coherence.py"),
    M("chaos", lang="c++", eigen=True, msvc_c=P_CHAOS, msvc_l=L_CHAOS,
      origin="setup_chaos.py"),

    # ---- Pure C, numpy headers only ---------------------------------------
    M("azimuth", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_azimuth.py"),
    M("depth_conversion", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_depth_conversion.py"),
    M("dip", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_dip.py"),
    M("dix_conversion", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_dix_conversion.py"),
    M("dix_inversion", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_dix_inversion.py"),
    M("first_derivative", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_first_derivative.py"),
    M("gradient_magnitude", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_gradient_magnitude.py"),
    M("relative_ai", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_relative_ai.py"),
    M("rms_amp", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_rms_amp.py"),
    M("second_derivative", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_second_derivative.py"),
    M("time_gain", msvc_c=P_C_INTEL, msvc_l=L_STD, origin="setup_time_gain.py"),

    # ---- already had a hand-written Linux branch upstream ------------------
    M("variance", linux_c=LNX_VARIANCE_C, linux_l=LNX_VARIANCE_L, directives=D_VAR,
      macros=M_NUMPY_ONLY, options={}, origin="setup_variance.py (linux branch)"),
]

BY_NAME = {m["name"]: m for m in MODULES}
NEEDS_EIGEN = sorted(m["name"] for m in MODULES if m["eigen"])
NEEDS_FFTW = sorted(m["name"] for m in MODULES if m["fftw"])

BUILD_ROOT = HERE / "build_linux"

# ---------------------------------------------------------------------------
# Flag translation
# ---------------------------------------------------------------------------


class UntranslatableFlag(Exception):
    pass


def _apply_march_policy(flags: list[str]) -> list[str]:
    """Route every ISA-selection flag through MARCH_POLICY so the whole batch agrees."""
    out = []
    for f in flags:
        if f in ("-mavx2", "-mfma", "-march=native", "-mtune=native"):
            continue
        out.append(f)
    if MARCH_POLICY == "native":
        out += ["-march=native", "-mtune=native"]
    else:
        out += ["-mavx2", "-mfma"]
    return out


def translate(msvc_flags: list[str], table: dict, module: str, dropped: dict) -> list[str]:
    out: list[str] = []
    for flag in msvc_flags:
        if flag not in table:
            raise UntranslatableFlag(
                f"{module}: MSVC flag {flag!r} has no entry in the translation table. "
                f"Refusing to guess -- add it to COMPILE_MAP/LINK_MAP or tell me what "
                f"it should become."
            )
        mapped = table[flag]
        if mapped is DROP:
            dropped.setdefault(flag, set()).add(module)
            continue
        out.extend(mapped)
    return out


def compile_args_for(mod: dict, dropped: dict) -> list[str]:
    if mod["linux_c"] is not None:
        args = list(mod["linux_c"])
    else:
        args = translate(mod["msvc_c"], COMPILE_MAP, mod["name"], dropped)
    if not ENABLE_LTO:
        args = [a for a in args if a != "-flto"]
    args = _apply_march_policy(args)
    return _dedupe(args)


def link_args_for(mod: dict, dropped: dict) -> list[str]:
    if mod["linux_l"] is not None:
        args = list(mod["linux_l"])
    else:
        args = translate(mod["msvc_l"], LINK_MAP, mod["name"], dropped)
    cargs = compile_args_for(mod, dropped)
    # /openmp is a compile-only flag on MSVC; on GCC/Clang -fopenmp must appear on
    # the link line too or libgomp never gets pulled in.
    if "-fopenmp" in cargs and "-fopenmp" not in args:
        args.insert(0, "-fopenmp")
    # -ffast-math (/fp:fast) lets GCC vectorise libm calls into libmvec ones such as
    # _ZGVdN4vv_atan2. distutils does not put -lm on the link line, so those come out
    # as undefined symbols and the module fails at *import*, not at build. MSVC had no
    # equivalent problem: its vector math lives in the always-linked CRT. On glibc
    # >= 2.22 libm.so is a linker script that pulls in libmvec via AS_NEEDED.
    if "-ffast-math" in cargs and "-lm" not in args:
        args.append("-lm")
    if not ENABLE_LTO:
        args = [a for a in args if a != "-flto"]
    if mod["fftw"] and EMBED_RPATH:
        args += [f"-Wl,-rpath,$ORIGIN/{FFTW_LIB.relative_to(HERE)}",
                 f"-Wl,-rpath,{FFTW_LIB}"]
    return _dedupe(args)


def _dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def fail(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\n[FAIL] {msg}\n", file=sys.stderr)
    sys.exit(1)


def check_bundled_dirs() -> None:
    """
    Verify eigen/, fftw/ and python312/ before anything else happens -- in
    particular before re-exec, so a missing folder is reported as a missing
    folder rather than as a confusing interpreter error.

    All three are checked in one pass and every problem is reported together;
    finding out about fftw/ only after fixing eigen/ is its own kind of waste.
    """
    problems: list[str] = []

    # -- eigen (header-only; the folder itself is the -I dir) --------------
    if not EIGEN_DIR.is_dir():
        problems.append(
            f"MISSING FOLDER: {EIGEN_DIR}\n"
            f"    Eigen headers, needed by {len(NEEDS_EIGEN)} module(s):\n"
            f"      {', '.join(NEEDS_EIGEN)}\n"
            f"    The folder itself must be the include dir, i.e. "
            f"{EIGEN_DIR / 'Eigen' / 'Dense'} must resolve\n"
            f"    (same layout as the Windows C:/eigen-3.4.0).")
    elif not (EIGEN_DIR / "Eigen" / "Dense").exists():
        problems.append(
            f"BAD LAYOUT: {EIGEN_DIR} exists but {EIGEN_DIR / 'Eigen' / 'Dense'} does not.\n"
            f"    Eigen must be laid out so <Eigen/Dense> resolves with -I{EIGEN_DIR}.\n"
            f"    Needed by: {', '.join(NEEDS_EIGEN)}")

    # -- fftw (autotools layout: include/fftw3.h, lib/libfftw3f.so*) -------
    if not FFTW_DIR.is_dir():
        problems.append(
            f"MISSING FOLDER: {FFTW_DIR}\n"
            f"    FFTW, needed by {len(NEEDS_FFTW)} module(s):\n"
            f"      {', '.join(NEEDS_FFTW)}\n"
            f"    Expected {FFTW_INCLUDE / 'fftw3.h'} and {FFTW_LIB / 'libfftw3f.so'}")
    else:
        if not (FFTW_INCLUDE / "fftw3.h").exists():
            problems.append(
                f"MISSING HEADER: {FFTW_INCLUDE / 'fftw3.h'}\n"
                f"    Needed by: {', '.join(NEEDS_FFTW)}")
        shared = sorted(FFTW_LIB.glob("libfftw3f.so*")) if FFTW_LIB.is_dir() else []
        static = sorted(FFTW_LIB.glob("libfftw3f.a")) if FFTW_LIB.is_dir() else []
        if not shared and not static:
            problems.append(
                f"MISSING LIBRARY: no libfftw3f.so* (or libfftw3f.a) in {FFTW_LIB}\n"
                f"    Note: 'libfftw3f-3' in the Windows setup files is the DLL name. On "
                f"Linux the\n"
                f"    library stem is 'fftw3f' -- this script links -l{FFTW_LIBNAME}.\n"
                f"    Needed by: {', '.join(NEEDS_FFTW)}")

    # -- bundled interpreter ----------------------------------------------
    if not PY_DIR.is_dir():
        problems.append(
            f"MISSING FOLDER: {PY_DIR}\n"
            f"    Standalone CPython 3.12, needed by ALL {len(MODULES)} modules -- it "
            f"defines the\n"
            f"    ABI the .so files are built against. Expected {PY_BIN}.")
    elif not PY_BIN.exists():
        problems.append(
            f"MISSING INTERPRETER: {PY_BIN}\n"
            f"    {PY_DIR} exists but has no bin/python3. Needed by ALL "
            f"{len(MODULES)} modules.")
    elif not os.access(PY_BIN, os.X_OK):
        problems.append(f"NOT EXECUTABLE: {PY_BIN}  (chmod +x it)")

    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s) with the bundled dependencies, "
              f"nothing was compiled:\n", file=sys.stderr)
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n", file=sys.stderr)
        sys.exit(1)


def preflight(targets: list[str]) -> None:
    """Deeper checks, run after re-exec under the bundled interpreter."""
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    print(f"  [ok]   eigen       {EIGEN_DIR}  ({len(NEEDS_EIGEN)} modules)")
    if not sorted(FFTW_LIB.glob("libfftw3f.so*")):
        print(f"  [warn] fftw        only a static libfftw3f.a in {FFTW_LIB}; it must have "
              f"been built with -fPIC to link into a shared module.")
    print(f"  [ok]   fftw        {FFTW_DIR}  ({len(NEEDS_FFTW)} modules)")
    print(f"  [ok]   python312   {PY_BIN}  (all {len(MODULES)} modules)")

    # -- we must be running under the bundled interpreter ------------------
    if not _running_under_bundled_python():
        fail(f"internal: expected to be re-executed under {PY_BIN}, running under "
             f"{sys.executable}")
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if ver != "3.12":
        fail(f"{PY_BIN} reports Python {ver}, expected 3.12.\n"
             f"       The existing Windows binaries are tagged cp312; a mismatch here "
             f"produces .so files the app cannot import.")
    py_h = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if not py_h.exists():
        fail(f"missing {py_h}\n"
             f"       The bundled interpreter has no development headers, so no extension "
             f"can be compiled against it.")
    print(f"  [ok]   interpreter Python {sys.version.split()[0]} "
          f"({sysconfig.get_platform()})")

    # -- 3. build-time Python packages, in the bundled interpreter ---------
    missing = []
    try:
        import Cython  # noqa: F401
        cy = Cython.__version__
    except ImportError:
        cy, _ = None, missing.append("cython")
    try:
        import numpy  # noqa: F401
        npv = numpy.__version__
    except ImportError:
        npv, _ = None, missing.append("numpy")
    try:
        import setuptools  # noqa: F401
        stv = setuptools.__version__
    except ImportError:
        stv, _ = None, missing.append("setuptools")
    if missing:
        fail(f"{PY_BIN} is missing build requirement(s): {', '.join(missing)}\n"
             f"       Install them into the bundled interpreter (not system python):\n"
             f"         {PY_BIN} -m pip install {' '.join(missing)}")
    print(f"  [ok]   toolchain   Cython {cy}, numpy {npv}, setuptools {stv}")

    # -- 4. the C/C++ compilers ------------------------------------------
    for exe, kind in ((CC, "CC"), (CXX, "CXX")):
        if shutil.which(exe) is None:
            fail(f"{kind}={exe!r} not found on PATH.\n"
                 f"       Set {kind} to the compiler you want to use, e.g. "
                 f"{kind}=/opt/rh/devtoolset-11/root/usr/bin/g++")
    needs_cxx17 = sorted(m["name"] for m in MODULES
                         if m["msvc_c"] and "/std:c++17" in m["msvc_c"])
    if needs_cxx17 and not _supports_cxx17():
        fail(f"{CXX} does not accept -std=c++17.\n"
             f"       {len(needs_cxx17)} module(s) require C++17: {', '.join(needs_cxx17)}\n"
             f"       CentOS 7's stock gcc is 4.8.5, which predates C++17. Enable a newer "
             f"toolchain without root, e.g.\n"
             f"         scl enable devtoolset-11 -- {Path(__file__).name}\n"
             f"       or point CC/CXX at a userspace gcc.")
    print(f"  [ok]   compilers   CC={CC}  CXX={CXX}"
          + (f"  (C++17 ok, {len(needs_cxx17)} modules need it)" if needs_cxx17 else ""))

    # -- 5. sources on disk vs. the port matrix ---------------------------
    on_disk = {p.stem for p in HERE.glob("*.pyx")}
    known = set(BY_NAME)

    # An unfamiliar .pyx is a hard stop: it wasn't in the setup_*.py dump, so its
    # include dirs, libraries and flags are unknown and guessing them is exactly
    # what this script is meant not to do.
    unknown = sorted(on_disk - known)
    if unknown:
        fail(f"{len(unknown)} .pyx file(s) have no entry in the port matrix: "
             f"{', '.join(unknown)}\n"
             f"       No setup_*.py was supplied for these, so their include dirs, "
             f"libraries and\n"
             f"       flags are unknown. Refusing to guess -- say what they need and "
             f"they can be added.")

    missing_targets = sorted(set(targets) - on_disk)
    if missing_targets:
        fail(f"{len(missing_targets)} requested module(s) have no .pyx here: "
             f"{', '.join(missing_targets)}")

    missing_other = sorted(known - on_disk - set(targets))
    if missing_other:
        print(f"  [warn] sources     {len(on_disk)}/{len(known)} .pyx present; not building "
              f"and not found: {', '.join(missing_other[:6])}"
              f"{f' (+{len(missing_other) - 6} more)' if len(missing_other) > 6 else ''}")
    else:
        print(f"  [ok]   sources     {len(on_disk)} .pyx files, all accounted for")
    print()


def _supports_cxx17() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "t.cpp"
        src.write_text("#include <cstddef>\nint main(){ if constexpr(true){} return 0; }\n")
        r = subprocess.run([CXX, "-std=c++17", "-fsyntax-only", str(src)],
                           capture_output=True)
        return r.returncode == 0


# ---------------------------------------------------------------------------
# Worker: build exactly one module
# ---------------------------------------------------------------------------

def build_one(name: str) -> int:
    import numpy as np
    from setuptools import Extension, setup
    from Cython.Build import cythonize
    from Cython.Compiler import Options

    mod = BY_NAME[name]
    dropped: dict = {}

    for k, v in mod["options"].items():
        setattr(Options, k, v)

    include_dirs = [np.get_include()]
    library_dirs: list[str] = []
    libraries: list[str] = []
    if mod["fftw"]:
        include_dirs.append(str(FFTW_INCLUDE))
        library_dirs.append(str(FFTW_LIB))
        libraries.append(FFTW_LIBNAME)      # "fftw3f", NOT the Windows "libfftw3f-3"
    if mod["eigen"]:
        include_dirs.append(str(EIGEN_DIR))

    ext = Extension(
        name=name,
        sources=[str(HERE / mod["source"])],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language=mod["lang"],
        extra_compile_args=compile_args_for(mod, dropped),
        extra_link_args=link_args_for(mod, dropped),
        define_macros=mod["macros"],
    )

    build_dir = BUILD_ROOT / name
    build_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(HERE)
    try:
        setup(
            name=name,
            ext_modules=cythonize([ext],
                                  compiler_directives=mod["directives"],
                                  annotate=mod["annotate"],
                                  quiet=True,
                                  nthreads=1),
            script_args=["build_ext", "--inplace", "--build-temp", str(build_dir)],
        )
    finally:
        os.chdir(old_cwd)
    return 0


# ---------------------------------------------------------------------------
# Parent: dispatch, collect, report
# ---------------------------------------------------------------------------

def run_module(name: str) -> tuple[str, bool, str, float]:
    t0 = time.time()
    env = dict(os.environ)
    env["CC"] = CC
    env["CXX"] = CXX
    # Standalone CPython distributions bake the compiler used to build them into
    # sysconfig; override the link driver so we use the toolchain present here.
    env.setdefault("LDSHARED", f"{CC} -shared")
    env.setdefault("LDCXXSHARED", f"{CXX} -shared")
    r = subprocess.run([str(PY_BIN), str(Path(__file__).resolve()), "--worker", name],
                       capture_output=True, text=True, env=env, cwd=str(HERE))
    out = (r.stdout or "") + (r.stderr or "")
    return name, r.returncode == 0, out, time.time() - t0


def describe_matrix() -> None:
    print("=" * 78)
    n_setup = len({m["origin"].split()[0] for m in MODULES})
    print(f"PORT MATRIX -- {len(MODULES)} modules from {n_setup} setup files")
    print("=" * 78)
    print(f"{'module':<22}{'lang':<6}{'deps':<14}{'links':<10}source")
    print("-" * 78)
    for m in sorted(MODULES, key=lambda x: x["name"]):
        deps = []
        if m["eigen"]:
            deps.append("eigen")
        if m["fftw"]:
            deps.append("fftw")
        links = FFTW_LIBNAME if m["fftw"] else "-"
        print(f"{m['name']:<22}{m['lang'] or 'c':<6}{('+'.join(deps) or '-'):<14}"
              f"{links:<10}{m['origin']}")
    print("-" * 78)
    print(f"eigen only  ({len([m for m in MODULES if m['eigen'] and not m['fftw']]):2d}) "
          f"header-only, include_dirs only, nothing to link")
    print(f"fftw        ({len(NEEDS_FFTW):2d}) include + library_dirs + -l{FFTW_LIBNAME}")
    print(f"neither     ({len([m for m in MODULES if not m['eigen'] and not m['fftw']]):2d}) "
          f"numpy headers only")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="SeismicFlow Linux extension build")
    ap.add_argument("--worker", metavar="NAME", help=argparse.SUPPRESS)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="build just these modules")
    ap.add_argument("--list", action="store_true", help="print the port matrix and exit")
    ap.add_argument("--check", action="store_true", help="preflight only, compile nothing")
    ap.add_argument("--clean", action="store_true",
                    help="remove generated .c/.cpp, build dirs and .so, then exit")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    if args.list:
        describe_matrix()
        return 0

    if args.clean:
        n = 0
        for m in MODULES:
            for suffix in (".c", ".cpp", ".html"):
                p = HERE / f"{m['name']}{suffix}"
                if p.exists():
                    p.unlink()
                    n += 1
            for so in HERE.glob(f"{m['name']}.cpython-*.so"):
                so.unlink()
                n += 1
        if BUILD_ROOT.exists():
            shutil.rmtree(BUILD_ROOT)
            n += 1
        print(f"removed {n} generated path(s)")
        return 0

    # -- worker leg: we are already the bundled interpreter ---------------
    if args.worker:
        return build_one(args.worker)

    # -- verify all three bundled folders up front, before anything else ---
    check_bundled_dirs()

    # -- re-exec under the bundled interpreter ----------------------------
    # This has to happen before anything imports Cython/numpy, so that the whole
    # build runs against python312's ABI rather than CentOS's system python.
    if not _running_under_bundled_python():
        print(f"[re-exec] switching from {sys.executable}")
        print(f"[re-exec]           to   {PY_BIN}")
        os.execv(str(PY_BIN), [str(PY_BIN), str(Path(__file__).resolve()), *sys.argv[1:]])

    targets = list(BY_NAME) if not args.only else args.only
    unknown = [t for t in targets if t not in BY_NAME]
    if unknown:
        fail(f"unknown module(s) requested: {', '.join(unknown)}")

    preflight(targets)
    if args.check:
        print("preflight only (--check); nothing compiled.")
        return 0

    # Surface the dropped-flag report up front, computed from the same translation
    # path the build uses -- so it can never drift from what was actually compiled.
    dropped: dict = {}
    for name in targets:
        compile_args_for(BY_NAME[name], dropped)
        link_args_for(BY_NAME[name], dropped)

    print("=" * 78)
    print(f"BUILDING {len(targets)} module(s) with {args.jobs} job(s)")
    print(f"  march policy : {MARCH_POLICY}"
          f"  ({'-mavx2 -mfma' if MARCH_POLICY == 'avx2' else '-march=native -mtune=native'})")
    print(f"  LTO          : {'on (-flto)' if ENABLE_LTO else 'off'}")
    print(f"  openmp       : -fopenmp on compile AND link")
    print("=" * 78)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, ok, out, secs in pool.map(run_module, targets):
            results.append((name, ok, out, secs))
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<24} {secs:6.1f}s")

    failed = [(n, o) for n, ok, o, _ in results if not ok]

    print()
    print("=" * 78)
    print(f"RESULT: {len(results) - len(failed)}/{len(results)} built")
    print("=" * 78)
    for name, ok, _, _ in results:
        if ok:
            so = sorted(HERE.glob(f"{name}.cpython-*.so"))
            if so:
                print(f"  {so[0].name}")

    if dropped:
        print()
        print("TRANSLATION NOTES -- MSVC flags with no faithful GCC equivalent, dropped:")
        for flag, mods in sorted(dropped.items()):
            note = TRANSLATION_NOTES.get(flag, "no note recorded")
            print(f"\n  {flag}  ({', '.join(sorted(mods))})")
            print(textwrap.fill(note, width=74,
                                initial_indent="      ", subsequent_indent="      "))

    if failed:
        print()
        print("=" * 78)
        print(f"{len(failed)} FAILURE(S) -- last 40 lines each")
        print("=" * 78)
        for name, out in failed:
            print(f"\n----- {name} " + "-" * (66 - len(name)))
            print("\n".join(out.strip().splitlines()[-40:]))
        return 1

    if EMBED_RPATH and any(BY_NAME[t]["fftw"] for t in targets):
        print()
        print("NOTE: the FFTW-linked modules carry -Wl,-rpath,$ORIGIN/fftw/lib. If you move")
        print("      the .so files away from this directory, ship fftw/lib alongside them or")
        print("      set LD_LIBRARY_PATH so the loader can locate 'libfftw3f.so.3'.")
        print("      Verify with:  readelf -d <module>.so | head")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UntranslatableFlag as e:
        fail(str(e))
    except KeyboardInterrupt:
        sys.exit(130)
