#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, time, json, shlex, platform, subprocess, threading, pathlib, hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import psutil

##Try to import optional libs for plots and GPU; degrade gracefully if missing##
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

##Stable timestamp string##
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

##Quick file sha1 for the main scripts to fingerprint versions##
def file_sha1(path: str) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

##Collect system and env info for reproducibility##
def gather_env_fingerprint(extra_files: List[str]) -> Dict[str, Any]:
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "env_vars_of_interest": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
    }

    ##Try to capture numpy/scipy versions and BLAS if available##
    try:
        import numpy as _np
        info["numpy_version"] = _np.__version__
        try:
            import numpy.__config__ as _npconf
            info["numpy_build_config"] = {}
            for k in dir(_npconf):
                if k.endswith("_info"):
                    info["numpy_build_config"][k] = getattr(_npconf, k)
        except Exception:
            pass
    except Exception:
        info["numpy_version"] = None

    try:
        import scipy as _sp
        info["scipy_version"] = _sp.__version__
    except Exception:
        info["scipy_version"] = None

    ##GPU info if available##
    if _NVML_OK:
        try:
            pynvml.nvmlInit()
            ndev = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(ndev):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({
                    "index": i,
                    "name": name,
                    "total_mem_GB": round(mem.total / (1024**3), 2)
                })
            info["gpus"] = gpus
        except Exception:
            info["gpus"] = None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    else:
        info["gpus"] = None

    ##Script fingerprints if user provided paths##
    fps = {}
    for p in extra_files:
        if p and os.path.isfile(p):
            fps[p] = file_sha1(p)
    info["file_hashes_sha1"] = fps
    return info

##Monitor resource usage of a process tree: peak RSS, CPU%; optional GPU polling##
class ResourceMonitor(threading.Thread):
    def __init__(self, pid: int, poll_sec: float = 0.1, collect_gpu: bool = False):
        super().__init__(daemon=True)
        self.pid = pid
        self.poll_sec = poll_sec
        self.collect_gpu = collect_gpu and _NVML_OK
        self._stop_evt = threading.Event()   ##rename to avoid clash with Thread._stop()##
        self.peak_rss = 0
        self.cpu_percent_max = 0.0
        self.samples = []
        self.gpu_samples = []

    def run(self):
        try:
            proc = psutil.Process(self.pid)
        except psutil.Error:
            return

        if self.collect_gpu:
            try:
                pynvml.nvmlInit()
            except Exception:
                self.collect_gpu = False

        try:
            while not self._stop_evt.is_set():
                try:
                    rss_total = 0
                    cpu_total = 0.0
                    procs = [proc] + proc.children(recursive=True)
                    for p in procs:
                        try:
                            with p.oneshot():
                                mi = p.memory_info()
                                rss_total += mi.rss
                                cpu_total += p.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                            pass

                    self.peak_rss = max(self.peak_rss, rss_total)
                    self.cpu_percent_max = max(self.cpu_percent_max, cpu_total)
                    self.samples.append({"t": time.time(), "rss": rss_total, "cpu": cpu_total})

                    if self.collect_gpu:
                        try:
                            ngpu = pynvml.nvmlDeviceGetCount()
                            for i in range(ngpu):
                                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                                self.gpu_samples.append({
                                    "t": time.time(),
                                    "gpu_index": i,
                                    "gpu_util": util.gpu,
                                    "mem_used_MB": int(mem.used / (1024**2))
                                })
                        except Exception:
                            pass

                    time.sleep(self.poll_sec)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    break
        finally:
            if self.collect_gpu:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def stop(self):
        self._stop_evt.set()


##Execute one command with monitoring##
def run_once(cmd: str, timeout: Optional[int], log_dir: pathlib.Path, collect_gpu: bool) -> Dict[str, Any]:
    run_id = _ts()
    stdout_path = log_dir / f"stdout_{run_id}.log"
    stderr_path = log_dir / f"stderr_{run_id}.log"
    meta: Dict[str, Any] = {
        "cmd": cmd,
        "start_time": datetime.now().isoformat(),
        "timeout_sec": timeout,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        start_wall = time.perf_counter()
        p = subprocess.Popen(
            cmd if os.name == "nt" else shlex.split(cmd),
            stdout=out, stderr=err, shell=(os.name == "nt"),
            env=os.environ.copy()
        )

        mon = ResourceMonitor(p.pid, poll_sec=0.1, collect_gpu=collect_gpu)
        mon.start()
        timed_out = False
        try:
            ret = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                p.kill()
            except Exception:
                pass
            ret = -9
        finally:
            mon.stop()
            mon.join(timeout=2.0)

        end_wall = time.perf_counter()

    try:
        proc = psutil.Process(p.pid)
        cpu_times = proc.cpu_times()
        cpu_used = (cpu_times.user + cpu_times.system)
    except Exception:
        cpu_used = None

    meta.update({
        "end_time": datetime.now().isoformat(),
        "return_code": ret,
        "timed_out": timed_out,
        "wall_time_sec": round(end_wall - start_wall, 6),
        "cpu_time_sec": round(cpu_used, 6) if cpu_used is not None else None,
        "peak_rss_mb": round(mon.peak_rss / (1024**2), 2),
        "max_cpu_percent_sum": round(mon.cpu_percent_max, 2),
    })

    if mon.gpu_samples:
        try:
            by_gpu = {}
            for s in mon.gpu_samples:
                i = s["gpu_index"]
                by_gpu.setdefault(i, {"util": [], "mem": []})
                by_gpu[i]["util"].append(s["gpu_util"])
                by_gpu[i]["mem"].append(s["mem_used_MB"])
            meta["gpu_summary"] = {
                int(i): {
                    "util_max_percent": max(v["util"]) if v["util"] else None,
                    "mem_max_mb": max(v["mem"]) if v["mem"] else None
                } for i, v in by_gpu.items()
            }
        except Exception:
            meta["gpu_summary"] = None
    else:
        meta["gpu_summary"] = None

    return meta

##Main orchestrator##
def main():
    ap = argparse.ArgumentParser(description="Benchmark harness for reproducible script performance analysis.")
    ap.add_argument("--cmd", action="append", help="Command to run (e.g., \"python main.py --n 1000\"). Can be given multiple times.")
    ap.add_argument("--config", type=str, help="Path to JSON config with a 'commands' array.")
    ap.add_argument("--runs", type=int, default=5, help="Number of measured runs per command (default 5).")
    ap.add_argument("--warmups", type=int, default=1, help="Warmup runs per command (default 1).")
    ap.add_argument("--timeout", type=int, default=None, help="Per-run timeout in seconds.")
    ap.add_argument("--outdir", type=str, default="outputs/sessions", help="Output directory for results.")
    ap.add_argument("--gpu", action="store_true", help="Attempt to collect GPU utilization (requires NVIDIA + pynvml).")
    ap.add_argument("--poll", type=float, default=0.1, help="Sampling period for monitors (sec).")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag/label for this benchmark session.")
    ap.add_argument("--fingerprint", action="append", help="Paths to important .py files to hash for versioning. Can repeat.")
    ap.add_argument("--setenv", action="append", help="Environment variables to set for child runs, e.g. OMP_NUM_THREADS=1. Can repeat.")
    args = ap.parse_args()

    commands: List[str] = []
    if args.cmd:
        commands.extend(args.cmd)
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict) or "commands" not in cfg or not isinstance(cfg["commands"], list):
            print("Config must be JSON with a list field 'commands'", file=sys.stderr)
            sys.exit(2)
        commands.extend(cfg["commands"])

    if not commands:
        print("Provide at least one --cmd or a --config with commands.", file=sys.stderr)
        sys.exit(2)

    if args.setenv:
        for kv in args.setenv:
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k.strip()] = v.strip()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / f"logs_{_ts()}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    extra_files = args.fingerprint if args.fingerprint else []
    env_info = gather_env_fingerprint(extra_files=extra_files)
    env_info["session_tag"] = args.tag

    with open(outdir / "environment_fingerprint.json", "w") as f:
        json.dump(env_info, f, indent=2)

    all_results: List[Dict[str, Any]] = []
    session_id = _ts()

    for cmd in commands:
        print(f"\n=== Benchmarking: {cmd} ===")
        for w in range(args.warmups):
            _ = run_once(cmd, timeout=args.timeout, log_dir=logs_dir, collect_gpu=args.gpu)
            print(f"  Warmup {w+1}/{args.warmups} done.")

        for r in range(args.runs):
            res = run_once(cmd, timeout=args.timeout, log_dir=logs_dir, collect_gpu=args.gpu)
            res["command"] = cmd
            res["run_index"] = r
            res["session_id"] = session_id
            res["tag"] = args.tag
            all_results.append(res)
            print(f"  Run {r+1}/{args.runs}: wall={res['wall_time_sec']:.4f}s, peakRSS={res['peak_rss_mb']} MB, rc={res['return_code']}{' (TO)' if res['timed_out'] else ''}")

    json_path = outdir / f"results_{session_id}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if pd is not None:
        df = pd.DataFrame(all_results)
        csv_path = outdir / f"results_{session_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        if plt is not None:
            try:
                fig1 = plt.figure()
                labels = []
                data = []
                for c in sorted(set(df["command"])):
                    labels.append(c)
                    data.append(df[df["command"] == c]["wall_time_sec"].values)
                plt.boxplot(data, showfliers=True)
                plt.xticks(range(1, len(labels) + 1), [f"cmd{idx+1}" for idx in range(len(labels))], rotation=0)
                plt.ylabel("Wall time (s)")
                plt.title("Runtime distribution per command")
                fig1.tight_layout()
                fig1_path = outdir / f"runtime_boxplot_{session_id}.png"
                fig1.savefig(fig1_path, dpi=150)
                plt.close(fig1)

                fig2 = plt.figure()
                means = df.groupby("command")["peak_rss_mb"].mean()
                x = list(range(len(means)))
                plt.bar(x, means.values)
                plt.xticks(x, [f"cmd{idx+1}" for idx in range(len(means))])
                plt.ylabel("Peak RSS (MB)")
                plt.title("Average peak memory by command")
                fig2.tight_layout()
                fig2_path = outdir / f"peakmem_bar_{session_id}.png"
                fig2.savefig(fig2_path, dpi=150)
                plt.close(fig2)

                print(f"Saved plots:\n  {fig1_path}\n  {fig2_path}")
            except Exception as e:
                print(f"Plotting skipped: {e}")
    else:
        print("\nInstall pandas/matplotlib to get CSV and plots.")

    print(f"\nRaw JSON saved to: {json_path}")
    print(f"Logs dir: {logs_dir}")

if __name__ == "__main__":
    main()
