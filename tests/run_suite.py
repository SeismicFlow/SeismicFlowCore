#!/usr/bin/env python3
"""
run_suite.py -- minimal cross-platform test suite for SeismicFlowCore.

One script, identical on Linux and Windows.  For each of the 28 public
functions it checks:

  1. import        the compiled extension (.so / .pyd) loads on its own
  2. run           the function executes on a tiny synthetic cube without error
  3. sane          output is not empty / all-NaN / all-zero
  4. repeatable    calling it twice on the same input gives bit-identical output,
                   and the input array is not modified
  5. thread-inv.   output with OMP_NUM_THREADS=1 vs =4 agrees within TOLERANCE
  6. parity        (optional) output agrees with a reference .npz produced on
                   another platform, within TOLERANCE

Every function is run in its OWN subprocess, so a crash (segfault) in one module
is recorded as a failure for that module instead of killing the whole run.

Typical use
-----------
  # Linux: produce the reference outputs
  python run_suite.py run --pkg-dir /path/to/SeismicFlowCore \
         --save-reference linux_reference_outputs.npz

  # Windows: check parity against the Linux reference
  python run_suite.py run --pkg-dir C:\\path\\to\\SeismicFlowCore \
         --compare linux_reference_outputs.npz

Tolerance (fixed in advance, same for every module)
---------------------------------------------------
  |a - b| <= ATOL_FRAC * max|b|  +  RTOL * |b|      for every element
  with RTOL = 1e-4, ATOL_FRAC = 1e-5   (roughly 100-1000x float32 epsilon).
  Angle-valued outputs (inst_phase, azimuth; degrees, period 360) are compared
  with the wrap-around difference, so 359.9999 vs 0.0001 counts as 0.0002.
  NaN / +-Inf positions must match exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Synthetic test data -- deliberately tiny:  (time, inline, crossline)
# ----------------------------------------------------------------------------
SEED = 20260101
NT, NI, NX = 128, 16, 16          # 32,768 samples, 128 KiB as float32
DT = 0.004                        # 4 ms sampling

RTOL = 1e-4
ATOL_FRAC = 1e-5


def make_inputs() -> dict[str, np.ndarray]:
    """Seeded synthetic cubes.  Shipped as test_inputs.npz so every platform
    reads byte-identical input instead of re-deriving it from an RNG."""
    rng = np.random.default_rng(SEED)

    # Ricker wavelet, 30 Hz peak, 4 ms sampling
    f0 = 30.0
    tw = np.arange(-32, 33) * DT
    a = (np.pi * f0 * tw) ** 2
    ricker = (1.0 - 2.0 * a) * np.exp(-a)

    # Laterally continuous reflectivity, dipping across the survey, plus
    # 30% trace-independent reflectivity so coherence/variance are not trivial.
    pad = 64
    base = rng.standard_normal(NT + 2 * pad)
    refl = np.empty((NT, NI, NX))
    for i in range(NI):
        for j in range(NX):
            s = int(round(0.4 * i + 0.25 * j))
            refl[:, i, j] = base[pad + s: pad + s + NT]
    refl += 0.3 * rng.standard_normal((NT, NI, NX))

    seis = np.apply_along_axis(lambda tr: np.convolve(tr, ricker, mode="same"), 0, refl)
    seis += 0.01 * seis.std() * rng.standard_normal(seis.shape)
    seis *= 100.0 / seis.std()

    # second cube for cross-correlation: same data delayed 3 samples + noise
    seis2 = np.roll(seis, 3, axis=0) + 0.1 * seis.std() * rng.standard_normal(seis.shape)

    # smooth, strictly increasing interval velocity (m/s) with lateral variation
    k = np.arange(NT)[:, None, None]
    ii = np.arange(NI)[None, :, None]
    jj = np.arange(NX)[None, None, :]
    vel = (1500.0 + 10.0 * k + 40.0 * np.sin(2 * np.pi * ii / NI)
           + 25.0 * np.cos(2 * np.pi * jj / NX)) * np.ones((NT, NI, NX))

    # stacking (RMS) velocity corresponding to `vel` -- valid input for Dix inversion
    vrms = np.sqrt(np.cumsum(vel ** 2, axis=0) / (np.arange(NT)[:, None, None] + 1.0))

    return {n: np.ascontiguousarray(v, dtype=np.float32)
            for n, v in dict(seis=seis, seis2=seis2, vel=vel, vrms=vrms).items()}


def inputs_sha256(inp: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(inp):
        h.update(k.encode())
        h.update(np.ascontiguousarray(inp[k]).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Test cases: one per public function in SeismicFlowCore/__init__.py
#   name  = public alias,  mod = extension module,  fn = function inside it
#   args  = input cubes (positional), kw = keyword parameters actually used
#   angle = {output_index: period} for angle-valued outputs (degrees)
# ----------------------------------------------------------------------------
CASES = [
    dict(name="agc", mod="agc", fn="trace_agc", args=["seis"], kw=dict(window_size=9)),
    dict(name="app_polarity", mod="app_polarity", fn="apparent_polarity", args=["seis"],
         kw=dict(threshold=1e-6, use_masking=True, window_size=33)),
    dict(name="azimuth", mod="azimuth", fn="local_structural_azimuth", args=["seis"], kw={},
         angle={0: 360.0}),
    dict(name="chaos", mod="chaos", fn="chaos", args=["seis"],
         kw=dict(sigma_t=1.5, sigma_il=1.5, sigma_xl=1.5, truncate=4.0)),
    dict(name="coherence", mod="coherence", fn="gst_coherence", args=["seis"],
         kw=dict(window=(5, 5, 5), sigma=1.0)),
    dict(name="cos_phase", mod="cos_phase", fn="cosine_of_phase", args=["seis"], kw={}),
    dict(name="depth_conversion", mod="depth_conversion", fn="time_to_depth_conversion",
         args=["vel", DT, "seis"], kw={}),
    dict(name="dip", mod="dip", fn="local_structural_dip", args=["seis"], kw={}),
    dict(name="dix_conversion", mod="dix_conversion", fn="interval_to_stacking_velocity",
         args=["vel", DT], kw={}),
    dict(name="dix_inversion", mod="dix_inversion", fn="stacking_to_interval_velocity",
         args=["vrms", DT], kw={}),
    dict(name="dominant_freq", mod="dominant_freq", fn="dominant_frequency",
         args=["seis", DT], kw=dict(threshold=1e-6, use_masking=True, window_size=33, waif_window=5)),
    dict(name="envelope", mod="envelope", fn="envelope", args=["seis"], kw={}),
    dict(name="first_derivative", mod="first_derivative", fn="first_derivative", args=["seis"], kw={}),
    dict(name="gradient_magnitude", mod="gradient_magnitude", fn="gradient_magnitude", args=["seis"], kw={}),
    dict(name="gsd", mod="gsd", fn="generalized_spectral_decomposition", args=["seis"],
         kw=dict(frequency=30.0, num_cycles=2, phase_deg=90.0)),
    dict(name="inst_bandwidth", mod="inst_bandwidth", fn="instantaneous_bandwidth",
         args=["seis", DT], kw=dict(threshold=1e-6, use_masking=True, window_size=33)),
    dict(name="inst_freq", mod="inst_freq", fn="instantaneous_frequency",
         args=["seis", DT], kw=dict(threshold=1e-6, use_masking=True, window_size=33, waif_window=5)),
    dict(name="inst_phase", mod="inst_phase", fn="instantaneous_phase", args=["seis"],
         kw=dict(threshold=1e-6, use_masking=True, window_size=33), angle={0: 360.0}),
    dict(name="phase_shift", mod="phase_shift", fn="phase_shift", args=["seis", 45.0], kw={}),
    dict(name="quad_amp", mod="quad_amp", fn="quadrature_amplitude", args=["seis"], kw={}),
    dict(name="reflection_intensity", mod="reflection_intensity", fn="reflection_intensity",
         args=["seis"], kw={}),
    dict(name="relative_ai", mod="relative_ai", fn="relative_acoustic_impedance",
         args=["seis", 10.0, 4.0], kw={}),
    dict(name="rms_amp", mod="rms_amp", fn="rms_amplitude", args=["seis"], kw=dict(window_size=9)),
    dict(name="second_derivative", mod="second_derivative", fn="second_derivative", args=["seis"], kw={}),
    dict(name="sweetness", mod="sweetness", fn="sweetness", args=["seis", DT], kw={}),
    dict(name="time_gain", mod="time_gain", fn="time_gain", args=["seis"],
         kw=dict(gain_exponent=1.5, sample_rate=DT)),
    dict(name="variance", mod="variance", fn="variance_edge_cython", args=["seis"],
         kw=dict(inline_range=3, crossline_range=3, vertical_smoothing=15)),
    dict(name="xcorr", mod="xcorr", fn="cross_correlation", args=["seis", "seis2"],
         kw=dict(max_lag=32, sample_rate=4.0)),
]
BY_NAME = {c["name"]: c for c in CASES}


# ----------------------------------------------------------------------------
# Worker: runs ONE module in ONE process (so crashes are contained)
# ----------------------------------------------------------------------------
def _flatten(ret) -> list[np.ndarray]:
    items = ret if isinstance(ret, (tuple, list)) else [ret]
    return [np.asarray(x) for x in items]


def _stats(a: np.ndarray) -> dict:
    fin = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
    f = a[fin].astype(np.float64)
    return dict(
        shape=list(a.shape), dtype=str(a.dtype),
        n_nan=int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0,
        n_inf=int(np.isinf(a).sum()) if a.dtype.kind == "f" else 0,
        min=float(f.min()) if f.size else None,
        max=float(f.max()) if f.size else None,
        mean=float(f.mean()) if f.size else None,
        frac_zero=float((a == 0).mean()),
        sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
    )


def worker_main(a: argparse.Namespace) -> int:
    case = BY_NAME[a.worker]
    res: dict = dict(name=case["name"], import_ok=False, run_ok=False)
    out_arrays: list[np.ndarray] = []
    try:
        if a.dll_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(a.dll_dir)
        sys.path.insert(0, str(Path(a.pkg_dir).resolve()))
        t0 = time.perf_counter()
        mod = importlib.import_module(case["mod"])
        fn = getattr(mod, case["fn"])
        res["import_s"] = time.perf_counter() - t0
        res["import_ok"] = True
        res["module_file"] = getattr(mod, "__file__", "?")
    except BaseException as e:                       # noqa: BLE001
        res["error"] = f"import: {type(e).__name__}: {e}"
        print("RESULT:" + json.dumps(res))
        return 0

    try:
        npz = np.load(a.inputs)
        cubes = {k: np.array(npz[k]) for k in npz.files}
        args = [cubes[x] if isinstance(x, str) else x for x in case["args"]]
        before = [hashlib.sha256(x.tobytes()).hexdigest() for x in args if isinstance(x, np.ndarray)]

        t0 = time.perf_counter()
        r1 = _flatten(fn(*args, **case["kw"]))
        res["call_s"] = time.perf_counter() - t0
        r2 = _flatten(fn(*args, **case["kw"]))
        after = [hashlib.sha256(x.tobytes()).hexdigest() for x in args if isinstance(x, np.ndarray)]

        out_arrays = r1
        res["run_ok"] = True
        res["n_outputs"] = len(r1)
        res["outputs"] = [_stats(x) for x in r1]
        res["repeatable"] = (len(r1) == len(r2) and
                             all(x.tobytes() == y.tobytes() for x, y in zip(r1, r2)))
        res["input_unmodified"] = (before == after)
        first = r1[0]
        fin = np.isfinite(first)
        res["sane"] = bool(fin.any() and np.any(first[fin] != 0))
    except BaseException as e:                       # noqa: BLE001
        res["error"] = f"run: {type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()

    if out_arrays:
        np.savez(a.out, **{f"out{i}": x for i, x in enumerate(out_arrays)})
    print("RESULT:" + json.dumps(res))
    return 0


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------
def compare_arrays(a: np.ndarray, b: np.ndarray, period: float | None = None) -> dict:
    """a = candidate, b = reference."""
    r = dict(shape_equal=a.shape == b.shape, dtype_equal=a.dtype == b.dtype,
             bit_identical=bool(a.shape == b.shape and a.tobytes() == b.tobytes()))
    if not r["shape_equal"]:
        r.update(ok=False, max_norm_err=None, nonfinite_match=False, n_violations=None)
        return r
    af, bf = a.astype(np.float64), b.astype(np.float64)
    r["nonfinite_match"] = bool(np.array_equal(np.isnan(af), np.isnan(bf)) and
                                np.array_equal(af[np.isinf(af)], bf[np.isinf(bf)]) and
                                np.array_equal(np.isinf(af), np.isinf(bf)))
    fin = np.isfinite(af) & np.isfinite(bf)
    if not fin.any():
        r.update(ok=bool(r["nonfinite_match"]), max_abs_err=0.0, max_norm_err=0.0, n_violations=0)
        return r
    d = af[fin] - bf[fin]
    if period:
        d = (d + period / 2.0) % period - period / 2.0
    scale = max(float(np.abs(bf[fin]).max()), 1e-30)
    tol = ATOL_FRAC * scale + RTOL * np.abs(bf[fin])
    viol = np.abs(d) > tol
    r.update(max_abs_err=float(np.abs(d).max()), max_norm_err=float(np.abs(d).max() / scale),
             n_violations=int(viol.sum()), n_compared=int(fin.sum()),
             ok=bool(r["nonfinite_match"] and not viol.any() and r["dtype_equal"]))
    return r


def compare_case(case: dict, got: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict:
    per = []
    ok = True
    for i in range(len(got)):
        ka = f"out{i}"
        if ka not in ref or ka not in got:
            per.append(dict(ok=False, note="missing output"))
            ok = False
            continue
        c = compare_arrays(got[ka], ref[ka], case.get("angle", {}).get(i))
        per.append(c)
        ok &= c["ok"]
    if len(ref) != len(got):
        ok = False
    return dict(ok=ok, per_output=per,
                max_norm_err=max((p.get("max_norm_err") or 0.0) for p in per) if per else None,
                bit_identical=all(p.get("bit_identical") for p in per))


# ----------------------------------------------------------------------------
# Parent
# ----------------------------------------------------------------------------
def run_worker(case_name: str, threads: int, a: argparse.Namespace, tmpdir: Path):
    out = tmpdir / f"{case_name}.t{threads}.npz"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--worker", case_name,
           "--pkg-dir", a.pkg_dir, "--inputs", a.inputs, "--out", str(out)]
    if a.dll_dir:
        cmd += ["--dll-dir", a.dll_dir]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return dict(name=case_name, import_ok=None, run_ok=False, error=f"TIMEOUT >{a.timeout}s"), {}
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("RESULT:")), None)
    if line is None:
        return dict(name=case_name, import_ok=None, run_ok=False,
                    error=f"process died (return code {p.returncode}); "
                          f"stderr tail: {p.stderr.strip()[-300:]}"), {}
    res = json.loads(line[len("RESULT:"):])
    res["wall_s"] = time.perf_counter() - t0
    outs = {}
    if out.exists():
        z = np.load(out)
        outs = {k: np.array(z[k]) for k in z.files}
    return res, outs


def package_import_check(a: argparse.Namespace) -> dict:
    """`import SeismicFlowCore` -- all 28 aliases must resolve to callables."""
    pkg = Path(a.pkg_dir).resolve()
    names = [c["name"] for c in CASES]
    code = (
        "import sys, json, os\n"
        f"d = {a.dll_dir!r}\n"
        "if d and hasattr(os, 'add_dll_directory'): os.add_dll_directory(d)\n"
        f"sys.path.insert(0, {str(pkg.parent)!r})\n"
        f"import importlib; m = importlib.import_module({pkg.name!r})\n"
        f"names = {names!r}\n"
        "missing = [n for n in names if not callable(getattr(m, n, None))]\n"
        "print('PKG:' + json.dumps(dict(ok=not missing, missing=missing)))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ), timeout=a.timeout)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PKG:")), None)
    if line is None:
        return dict(ok=False, missing=names, error=(p.stderr.strip()[-400:] or f"rc={p.returncode}"))
    return json.loads(line[4:])


def env_info() -> dict:
    info = dict(platform=platform.platform(), machine=platform.machine(),
                python=sys.version.split()[0], numpy=np.__version__,
                cpu_count=os.cpu_count(), seed=SEED, cube_shape=[NT, NI, NX],
                rtol=RTOL, atol_frac=ATOL_FRAC)
    if sys.platform.startswith("linux"):
        for key, cmd in (("gcc", ["gcc", "--version"]), ("glibc", ["ldd", "--version"])):
            try:
                info[key] = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
            except Exception:                         # noqa: BLE001
                info[key] = "n/a"
    return info


def parent_main(a: argparse.Namespace) -> int:
    inputs_path = Path(a.inputs)
    if not inputs_path.exists():
        np.savez(inputs_path, **make_inputs())
        print(f"[inputs] generated {inputs_path}")
    inp = {k: np.array(v) for k, v in np.load(inputs_path).items()}
    in_hash = inputs_sha256(inp)

    ref = None
    if a.compare:
        z = np.load(a.compare, allow_pickle=False)
        ref_hash = str(z["__input_sha256__"])
        if ref_hash != in_hash:
            print("ERROR: input cubes differ from those used for the reference "
                  f"({in_hash[:12]} vs {ref_hash[:12]}). Use the shipped test_inputs.npz.")
            return 2
        ref = {k: np.array(z[k]) for k in z.files if not k.startswith("__")}

    names = a.only or [c["name"] for c in CASES]
    threads = a.threads
    tmpdir = Path(a.workdir) if a.workdir else HERE / "_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    print(f"pkg-dir : {a.pkg_dir}")
    print(f"threads : {threads}   cube: {NT}x{NI}x{NX} float32   inputs sha256: {in_hash[:16]}...")
    print(f"platform: {platform.platform()}  python {sys.version.split()[0]}  numpy {np.__version__}\n")

    rows, details, save_ref = [], {}, {}
    for n in names:
        case = BY_NAME[n]
        per_thread = {}
        for t in threads:
            res, outs = run_worker(n, t, a, tmpdir)
            per_thread[t] = (res, outs)
        r0, o0 = per_thread[threads[0]]
        row = dict(module=n, import_ok=r0.get("import_ok"), run_ok=r0.get("run_ok"),
                   sane=r0.get("sane"), repeatable=r0.get("repeatable"),
                   input_unmodified=r0.get("input_unmodified"),
                   error=r0.get("error", ""))

        # --- thread invariance: every other thread count vs the first ----------
        thr_ok, thr_err, thr_bits = None, None, None
        if r0.get("run_ok") and len(threads) > 1:
            thr_ok, thr_err, thr_bits = True, 0.0, True
            for t in threads[1:]:
                rt, ot = per_thread[t]
                if not rt.get("run_ok"):
                    thr_ok = False
                    continue
                c = compare_case(case, ot, o0)
                thr_ok &= c["ok"]
                thr_bits &= c["bit_identical"]
                thr_err = max(thr_err, c["max_norm_err"] or 0.0)
        row.update(thread_inv_ok=thr_ok, thread_inv_max_norm_err=thr_err, thread_inv_bitwise=thr_bits)

        # --- parity against reference (other platform) -------------------------
        par_ok, par_err, par_bits = None, None, None
        if ref is not None:
            rmod = {k.split("__", 1)[1]: v for k, v in ref.items() if k.startswith(n + "__")}
            if not r0.get("run_ok"):
                par_ok = False
            else:
                par_ok, par_err, par_bits = True, 0.0, True
                for t in threads:
                    rt, ot = per_thread[t]
                    if not rt.get("run_ok"):
                        par_ok = False
                        continue
                    c = compare_case(case, ot, rmod)
                    par_ok &= c["ok"]
                    par_bits &= c["bit_identical"]
                    par_err = max(par_err, c["max_norm_err"] or 0.0)
                    details.setdefault(n, {})[f"parity_t{t}"] = c
        row.update(parity_ok=par_ok, parity_max_norm_err=par_err, parity_bitwise=par_bits)

        if r0.get("run_ok"):
            for k, v in o0.items():
                save_ref[f"{n}__{k}"] = v
        details.setdefault(n, {})["worker"] = {str(t): per_thread[t][0] for t in threads}
        rows.append(row)
        flag = lambda v: {True: "ok", False: "FAIL", None: "-"}[v]  # noqa: E731
        print(f"  {n:<22} import:{flag(row['import_ok']):<5} run:{flag(row['run_ok']):<5} "
              f"sane:{flag(row['sane']):<5} repeat:{flag(row['repeatable']):<5} "
              f"thr-inv:{flag(row['thread_inv_ok']):<5}"
              + (f" parity:{flag(row['parity_ok']):<5} (max norm err {row['parity_max_norm_err']:.2e})"
                 if ref is not None and row["parity_max_norm_err"] is not None else ""))
        if row["error"]:
            print(f"      -> {row['error'][:200]}")

    print("\n[package] import SeismicFlowCore ...", end=" ")
    pkg = package_import_check(a)
    print("ok (28/28 public names callable)" if pkg["ok"] else f"FAIL {pkg}")

    # ------------------------------ summary -----------------------------------
    n = len(rows)
    cnt = lambda key: sum(1 for r in rows if r.get(key) is True)  # noqa: E731
    summary = dict(modules=n, import_ok=cnt("import_ok"), run_ok=cnt("run_ok"), sane=cnt("sane"),
                   repeatable=cnt("repeatable"), input_unmodified=cnt("input_unmodified"),
                   thread_inv_ok=cnt("thread_inv_ok"), thread_inv_bitwise=cnt("thread_inv_bitwise"),
                   package_import_ok=bool(pkg["ok"]))
    if ref is not None:
        summary.update(parity_ok=cnt("parity_ok"), parity_bitwise=cnt("parity_bitwise"))
    print("\nSUMMARY  (" + ", ".join(f"{k}={v}" + (f"/{n}" if isinstance(v, int) and not isinstance(v, bool) and k != 'modules' else "")
                                      for k, v in summary.items()) + ")")

    if a.save_reference:
        np.savez(a.save_reference, __input_sha256__=np.array(in_hash), **save_ref)
        print(f"[reference] wrote {a.save_reference}  ({len(save_ref)} arrays)")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(env=env_info(), summary=summary, package=pkg,
                                                rows=rows, details=details), indent=1, default=str))
        print(f"[json] {a.json}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {a.csv}")

    bad = [r["module"] for r in rows
           if not (r["import_ok"] and r["run_ok"] and r["sane"] and r["repeatable"]
                   and r["thread_inv_ok"] is not False and r["parity_ok"] is not False)]
    if bad or not pkg["ok"]:
        print("\nNEEDS ATTENTION:", ", ".join(bad) or "(package import)")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SeismicFlowCore cross-platform test suite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-inputs", help="(re)generate test_inputs.npz")
    g.add_argument("--out", default=str(HERE / "test_inputs.npz"))

    r = sub.add_parser("run", help="run the suite")
    r.add_argument("--pkg-dir", required=True, help="the SeismicFlowCore folder holding the .so/.pyd files")
    r.add_argument("--inputs", default=str(HERE / "test_inputs.npz"))
    r.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    r.add_argument("--only", nargs="+")
    r.add_argument("--save-reference", metavar="NPZ")
    r.add_argument("--compare", metavar="NPZ", help="reference outputs from another platform")
    r.add_argument("--json"), r.add_argument("--csv")
    r.add_argument("--dll-dir", help="Windows: folder with libfftw3f-3.dll if not on PATH")
    r.add_argument("--timeout", type=int, default=300)
    r.add_argument("--workdir")
    r.add_argument("--worker", help=argparse.SUPPRESS)
    r.add_argument("--out", help=argparse.SUPPRESS)

    a = ap.parse_args()
    if a.cmd == "gen-inputs":
        np.savez(a.out, **make_inputs())
        print(f"wrote {a.out}")
        return 0
    if a.worker:
        return worker_main(a)
    return parent_main(a)


if __name__ == "__main__":
    sys.exit(main())
