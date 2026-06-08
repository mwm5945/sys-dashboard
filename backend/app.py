import glob
import time
import threading
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import psutil

try:
    from py3nvml import py3nvml_nvmlInitHandleError
    from py3nvml.py3nvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetFanSpeed,
        NVML_TEMPERATURE_GPU,
    )
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

app = FastAPI(title="Sys Dashboard")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

history_lock = threading.Lock()
history: deque = deque(maxlen=60)


def _get_cpu_temperatures():
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                low = name.lower()
                if any(kw in low for kw in ("core", "cpu")) or any(kw in name for kw in ("Tctl", "Tdie")):
                    return entries[0].current
            return next(iter(temps.values()))[0].current
    except Exception:
        pass

    for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            val = int(Path(zone_path).read_text().strip()) / 1000.0
            type_file = Path(zone_path).parent / "type"
            if type_file.exists():
                t = type_file.read_text().strip().lower()
                if any(kw in t for kw in ("x86", "coretemp")):
                    return val
        except Exception:
            continue

    for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            return int(Path(zone_path).read_text().strip()) / 1000.0
        except Exception:
            continue

    return None


def _get_fans():
    """Return list of dict {name, rpm} for all system fans."""
    result = []
    try:
        fans = psutil.sensors_fans()
        if fans:
            for name, entries in fans.items():
                for entry in entries:
                    rpm = entry.current or None
                    label = entry.label or name
                    result.append({"name": label, "rpm": rpm})
    except Exception:
        pass
    return result


def _get_nvidia_gpus():
    if not HAS_NVML:
        return []
    try:
        handle, ok = py3nvml_nvmlInitHandleError()
        if not ok:
            return []
        count = nvmlDeviceGetCount(handle)
    except Exception:
        return []

    devices = []
    for i in range(count):
        try:
            dev = nvmlDeviceGetHandleByIndex(handle, i)
            name_raw = nvmlDeviceGetName(dev)
            name = name_raw.strip() if hasattr(name_raw, "strip") else str(name_raw)
            util = nvmlDeviceGetUtilizationRates(dev)
            mem_info = nvmlDeviceGetMemoryInfo(dev)

            try:
                temp = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            try:
                power_w = nvmlDeviceGetPowerUsage(dev) / 1e9
            except Exception:
                power_w = None

            try:
                fan = nvmlDeviceGetFanSpeed(dev)
            except Exception:
                fan = None

            mem_pct = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total else None

            devices.append({
                "name": name,
                "index": i,
                "utilization": util.gpu,
                "memory_used": mem_info.used,
                "memory_total": mem_info.total,
                "memory_percent": mem_pct,
                "temperature": temp,
                "power": round(power_w, 2) if power_w else None,
                "fan_speed": fan,
            })
        except Exception:
            continue

    return devices


def _get_other_gpus():
    results = []
    try:
        for card in sorted(glob.glob("/sys/class/drm/card*/device")):
            p = Path(card)
            name_file = p / "name"
            try:
                name = name_file.read_text().strip()
            except Exception:
                name = "unknown-gpu"
            results.append({
                "name": name,
                "index": -1,
                "utilization": None,
                "memory_used": None,
                "memory_total": None,
                "memory_percent": None,
                "temperature": None,
                "power": None,
                "fan_speed": None,
            })
    except Exception:
        pass
    return results


def collect_gpu_metrics():
    nv = _get_nvidia_gpus()
    return nv if nv else _get_other_gpus()


_prev_net_counters = {}


def _collect_network():
    global _prev_net_counters
    net_counters = psutil.net_io_counters(pernic=True)
    now = time.monotonic()
    active_nics = []

    for k, v in net_counters.items():
        if k.startswith("lo"):
            continue
        if not (v.bytes_sent + v.bytes_recv) > 0:
            continue

        rates = {"bytes_sent_per_sec": 0, "bytes_recv_per_sec": 0}
        prev = _prev_net_counters.get(k)
        if prev is not None and hasattr(prev, "timestamp"):
            dt = now - prev["timestamp"]
            if dt > 0:
                rates["bytes_sent_per_sec"] = (v.bytes_sent - prev["bytes_sent"]) / dt
                rates["bytes_recv_per_sec"] = (v.bytes_recv - prev["bytes_recv"]) / dt

        _prev_net_counters[k] = {
            "timestamp": now,
            "bytes_sent": v.bytes_sent,
            "bytes_recv": v.bytes_recv,
        }

        active_nics.append({
            "name": k,
            "bytes_sent": v.bytes_sent,
            "bytes_recv": v.bytes_recv,
            "packets_sent": v.packets_sent,
            "packets_recv": v.packets_recv,
            **rates,
        })

    return active_nics


def collect_once():
    ts = datetime.now(timezone.utc).isoformat()

    cpu_freq = psutil.cpu_freq()
    freq_data = {
        "current": round(cpu_freq.current, 2) if cpu_freq and cpu_freq.current else None,
        "min": round(cpu_freq.min, 2) if cpu_freq and cpu_freq.min else None,
        "max": round(cpu_freq.max, 2) if cpu_freq and cpu_freq.max else None,
    }

    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    return {
        "timestamp": ts,
        "cpu": {
            "percent": psutil.cpu_percent(interval=0),
            "per_core": psutil.cpu_percent(interval=0, percpu=True),
            "freq": freq_data,
            "count_logical": psutil.cpu_count(logical=True) or 0,
            "count_physical": psutil.cpu_count(logical=False) or 0,
            "temperature": _get_cpu_temperatures(),
        },
        "memory": {
            "total": vm.total,
            "used": vm.used,
            "available": vm.available,
            "percent": vm.percent,
            "swap_total": sm.total,
            "swap_used": sm.used,
            "swap_percent": sm.percent,
        },
        "gpus": collect_gpu_metrics(),
        "network": _collect_network(),
        "fans": _get_fans(),
    }


def _background_collector():
    psutil.cpu_percent(interval=0)
    while True:
        time.sleep(1)
        data = collect_once()
        with history_lock:
            history.append(data)


threading.Thread(target=_background_collector, daemon=True).start()


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        status_code=404,
        media_type="text/html",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )


@app.get("/api/metrics")
def api_metrics():
    return collect_once()


@app.get("/api/history")
def api_history():
    with history_lock:
        snapshots = list(history)

    if not snapshots:
        return {
            "timestamps": [],
            "cpu_percent": [],
            "mem_percent": [],
            "net_bytes_sent": [],
            "net_bytes_recv": [],
            "gpu_utilization": {},
        }

    timestamps = [s["timestamp"] for s in snapshots]
    cpu_percents = [s["cpu"]["percent"] for s in snapshots]
    mem_percents = [s["memory"]["percent"] for s in snapshots]
    net_sent = [sum(nic["bytes_sent_per_sec"] for nic in s["network"]) for s in snapshots]
    net_recv = [sum(nic["bytes_recv_per_sec"] for nic in s["network"]) for s in snapshots]

    gpu_utils: dict[int, list[float | None]] = {}
    for s in snapshots:
        for gpu in s.get("gpus", []):
            idx = gpu.get("index", 0)
            gpu_utils.setdefault(idx, []).append(gpu.get("utilization"))

    return {
        "timestamps": timestamps,
        "cpu_percent": cpu_percents,
        "mem_percent": mem_percents,
        "net_bytes_sent": net_sent,
        "net_bytes_recv": net_recv,
        "gpu_utilization": {str(k): v for k, v in gpu_utils.items()},
    }
