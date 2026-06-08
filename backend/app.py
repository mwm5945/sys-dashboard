import glob
import re
import time
import threading
import json
import subprocess
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import psutil

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
    """Query nvidia-smi for GPU metrics via JSON output."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu="
             "index,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,fan.speed,"
             "clocks.current.graphics,clocks.current.memory",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []

    devices = []
    _mb = 1024 * 1024
    for i, line in enumerate(out.stdout.strip().split("\n")):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue

        def _num(s, fallback=None):
            try:
                return float(s)
            except (ValueError, TypeError):
                return fallback

        devices.append({
            "name": parts[1] if len(parts) > 1 else f"GPU {i}",
            "index": i,
            "utilization": _num(parts[2], 0),
            "memory_used": int(_num(parts[3], 0)) * _mb,
            "memory_total": int(_num(parts[4], 0)) * _mb,
            "memory_percent": round(float(_num(parts[3], 0)) / max(float(_num(parts[4], 1)), 1) * 100, 1),
            "temperature": _num(parts[5]),
            "power": round(_num(parts[6], 0), 2),
            "fan_speed": _num(parts[7]),
        })

    return devices


def _read_intel_gt_freqs(card_dir):
    """Read Intel GT frequency data from sysfs/debugfs."""
    device_path = card_dir / "device"

    result = {
        "gt_cur_freq_mhz": None,
        "gt_min_freq_mhz": None,
        "gt_max_freq_mhz": None,
        "frequency_percent": None,
    }

    # Try i915 driver GT frequency sysfs files (available on modern kernels)
    freq_paths = [
        device_path / "intel_guc" / "freq_table",
        device_path / "gt_cur_freq_mhz",
        device_path / "hwmon" / "hwmon*" / "in0_input",  # sometimes used as freq proxy
    ]

    for fp in freq_paths:
        matched = Path(fp).glob("*") if "*" in str(fp) else [fp]
        for p in matched:
            try:
                if p.exists() and p.is_file():
                    val = int(p.read_text().strip())
                    if "min" in p.name.lower():
                        result["gt_min_freq_mhz"] = val
                    elif "max" in p.name.lower():
                        result["gt_max_freq_mhz"] = val
                    else:
                        result["gt_cur_freq_mhz"] = val
            except Exception:
                continue

    # Try hwmon for frequency info
    hwmon_base = device_path / "hwmon"
    if hwmon_base.exists():
        for hwmon_dir in sorted(hwmon_base.glob("*")):
            if not hwmon_dir.is_dir():
                continue
            # Read all temp inputs for freq data on some platforms
            for tp in sorted(hwmon_dir.glob("temp*_input")):
                try:
                    val = int(tp.read_text().strip())
                    if result["gt_cur_freq_mhz"] is None and val > 50:
                        result["gt_cur_freq_mhz"] = val
                except Exception:
                    continue

    # Try debugfs for i915 engine stats (best quality, may need root)
    try:
        card_name = card_dir.name  # e.g., "card0"
        card_num = card_name.replace("card", "")
        debugfs_base = Path(f"/sys/kernel/debug/dri/{card_num}")

        # Read engine info for utilization
        engine_file = debugfs_base / "i915_engine_info"
        if engine_file.exists():
            content = engine_file.read_text()
            lines = content.strip().split("\n")
            total_ns = 0
            busy_ns = 0
            in_stats = False

            for line in lines:
                if "[engine stats]:" in line or "Engine time" in line:
                    in_stats = True
                    continue
                if in_stats and line.strip():
                    # Parse "Render/3D: xxxxx.yyy busy zzzz.aaa free bbb.bbb idle"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            if "." in part and i + 1 < len(parts):
                                total_ns += float(part)
                                busy_ns += float(parts[i + 1])
                        except (ValueError, IndexError):
                            continue

            if total_ns > 0 and busy_ns >= 0:
                utilization = round((busy_ns / max(total_ns, 1)) * 100, 1)
                result["frequency_percent"] = min(utilization, 100)
    except Exception:
        pass

    # Calculate percentage from freq range if available
    if (result["gt_cur_freq_mhz"] is not None and result["gt_max_freq_mhz"] is not None
            and result["gt_max_freq_mhz"] > 0):
        pct = round(result["gt_cur_freq_mhz"] / result["gt_max_freq_mhz"] * 100, 1)
        if result["frequency_percent"] is None or pct > result["frequency_percent"]:
            result["frequency_percent"] = pct

    return result


def _read_intel_vram(card_dir):
    """Read Intel GPU memory usage from sysfs."""
    device_path = card_dir / "device"
    hwmon_base = device_path / "hwmon"

    vram_used_mb = None
    vram_total_mb = None

    # Try to read VRAM info via debugfs
    try:
        card_name = card_dir.name
        card_num = card_name.replace("card", "")
        debugfs_base = Path(f"/sys/kernel/debug/dri/{card_num}")

        gm_info = debugfs_base / "i915_gem_info"
        if gm_info.exists():
            content = gm_info.read_text()
            # Parse "total: xxxxx, used: yyyy, reserved: zzzz" line
            for line in content.strip().split("\n"):
                low = line.lower()
                if "total:" in low and "used:" in low:
                    # Format: " total: 8388608, used: 12345, reserved: 678"
                    parts = {}
                    for token in re.split(r'[,\s]+', line):
                        token = token.strip()
                        if ":" in token:
                            k, v = token.split(":", 1)
                            try:
                                parts[k.strip().lower()] = int(v.strip())
                            except ValueError:
                                pass

                    # Values are in pages (4KB each) or bytes depending on kernel version
                    if "total" in parts and "used" in parts:
                        total_val = parts["total"]
                        used_val = parts["used"]
                        # Most kernels report in 4K pages
                        # But some newer kernels report in bytes
                        if total_val > 100_000_000:
                            # If total > 100M, likely already in pages (typical ~2M+ for 8GB)
                            vram_total_mb = round(total_val * 4 / 1024, 1)
                            vram_used_mb = round(used_val * 4 / 1024, 1)
                        elif total_val > 10_000:
                            # Likely in pages
                            vram_total_mb = round(total_val * 4 / 1024, 1)
                            vram_used_mb = round(used_val * 4 / 1024, 1)

    except Exception:
        pass

    return {
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
    }


def _get_intel_gpus():
    """Query Intel iGPU metrics from i915 driver sysfs interfaces."""
    results = []

    if not Path("/sys/class/drm").exists():
        return results

    card_entries = sorted(Path("/sys/class/drm").glob("card*"))
    card_entries = [c for c in card_entries if re.match(r'^card\d+$', c.name)]

    for card_dir in card_entries:
        device_path = card_dir / "device"
        p = Path(device_path)

        # Verify this is an Intel GPU (vendor 0x8086)
        try:
            vid = (p / "vendor").read_text().strip()
            if vid != "0x8086":
                continue
        except Exception:
            continue

        # Read GPU name
        name = None
        name_file = p / "name"
        try:
            name = name_file.read_text().strip()
        except Exception:
            pass

        if not name:
            try:
                drv_link = p / "driver"
                drv_name = drv_link.resolve().name if drv_link.exists() else ""
                device_id = (p / "device").read_text().strip()
                name = f"Intel Device {device_id}"
            except Exception:
                name = "Intel iGPU"

        # Read temperature via hwmon
        temp = None
        hwmon_base = p / "hwmon"
        if hwmon_base.exists():
            for hwmon_dir in sorted(hwmon_base.glob("*")):
                if not hwmon_dir.is_dir():
                    continue
                for tp in sorted(hwmon_dir.glob("temp*_input")):
                    try:
                        val = int(tp.read_text().strip())
                        # Some hwmon report in millidegrees C, others directly
                        temp_c = val / 1000.0 if val > 100 else val
                        if 20 <= temp_c <= 120:
                            temp = round(temp_c, 1)
                            break
                    except Exception:
                        continue
                if temp is not None:
                    break

            # Fallback: check direct temp files at hwmon base level
            if temp is None:
                for tp in sorted(hwmon_base.glob("temp*_input")):
                    try:
                        val = int(tp.read_text().strip())
                        temp_c = val / 1000.0 if val > 100 else val
                        if 20 <= temp_c <= 120:
                            temp = round(temp_c, 1)
                            break
                    except Exception:
                        continue

        # Read power via hwmon (usually pow1_input on Intel platforms)
        power_watts = None
        if hwmon_base.exists():
            for hwmon_dir in sorted(hwmon_base.glob("*")):
                if not hwmon_dir.is_dir():
                    continue
                for pwf in sorted(hwmon_dir.glob("power*_input")):
                    try:
                        val = int(pwf.read_text().strip()) / 1000.0  # milliwatts -> watts
                        power_watts = round(val, 2)
                        break
                    except Exception:
                        continue
                if power_watts is not None:
                    break

        # Fallback power at direct level
        if power_watts is None and hwmon_base.exists():
            for pwf in sorted(hwmon_base.glob("power*_input")):
                try:
                    val = int(pwf.read_text().strip()) / 1000.0
                    power_watts = round(val, 2)
                    break
                except Exception:
                    continue

        # Read GT frequencies and utilization
        freq_data = _read_intel_gt_freqs(card_dir)
        gpu_utilization = freq_data.get("frequency_percent")

        # Read VRAM usage
        vram_data = _read_intel_vram(card_dir)
        vram_used_bytes = None
        vram_total_bytes = None
        vram_pct = None

        if vram_data["vram_used_mb"] is not None and vram_data["vram_total_mb"] is not None:
            vram_used_bytes = vram_data["vram_used_mb"] * 1024 * 1024
            vram_total_bytes = vram_data["vram_total_mb"] * 1024 * 1024
            if vram_total_bytes > 0:
                vram_pct = round(vram_used_bytes / vram_total_bytes * 100, 1)

        # If no utilization from sysfs, try parsing via intel_gpu_top command
        if gpu_utilization is None:
            try:
                out = subprocess.run(
                    ["intel_gpu_top", "-d", "0", "-T", "1", "-J", "--json"],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0:
                    jdata = json.loads(out.stdout)
                    gpu_utilization = round(jdata.get("gpu_utilization.busy%", 0), 1)
                    # Also try to get frequency from intel_gpu_top output
                    freq_keys = [k for k in jdata if "freq" in k.lower() and "cur" in k.lower()]
                    for fk in freq_keys:
                        if isinstance(jdata[fk], (int, float)):
                            freq_data["gt_cur_freq_mhz"] = round(jdata[fk], 1)
                    max_keys = [k for k in jdata if "freq" in k.lower() and "max" in k.lower()]
                    for mk in max_keys:
                        if isinstance(jdata[mk], (int, float)):
                            freq_data["gt_max_freq_mhz"] = round(jdata[mk], 1)
            except Exception:
                pass

        # Read RC6 residency (idle percentage - lower means GPU is busier)
        rc6_residency = None
        try:
            rc6_file = p / "intel_gt_pm_interval"
            if not rc6_file.exists():
                card_num = card_dir.name.replace("card", "")
                rc6_file = Path(f"/sys/kernel/debug/dri/{card_num}/i915_pm_info")

            if rc6_file.exists():
                content = rc6_file.read_text()
                for line in content.strip().split("\n"):
                    if "RC6" in line.upper() or "sleep" in line.lower():
                        nums = [float(x) for x in re.findall(r'[\d.]+', line)]
                        if nums:
                            rc6_residency = round(min(nums[0], 100), 1)
                        break
        except Exception:
            pass

        card_num = int(card_dir.name.replace("card", ""))

        results.append({
            "name": name,
            "index": card_num + 100,  # Use high index to avoid collision with Nvidia GPU indices
            "vendor": "Intel",
            "utilization": gpu_utilization,
            "memory_used": int(vram_used_bytes) if vram_used_bytes else None,
            "memory_total": int(vram_total_bytes) if vram_total_bytes else None,
            "memory_percent": vram_pct,
            "gt_cur_freq_mhz": freq_data.get("gt_cur_freq_mhz"),
            "gt_min_freq_mhz": freq_data.get("gt_min_freq_mhz"),
            "gt_max_freq_mhz": freq_data.get("gt_max_freq_mhz"),
            "rc6_residency": rc6_residency,
            "temperature": temp,
            "power": power_watts,
        })

    return results

def _get_other_gpus():
    vendor_map = {
        "0x8086": "Intel", "0x1002": "AMD", "0x10de": "NVIDIA",
    }
    results = []
    try:
        card_entries = sorted(Path("/sys/class/drm").glob("card*"))
        card_entries = [c for c in card_entries if re.match(r'^card\d+$', c.name)]

        for card_dir in card_entries:
            device_path = card_dir / "device"
            p = Path(device_path)

            try:
                vid = (p / "vendor").read_text().strip()
            except Exception:
                continue

            # Skip NVIDIA cards - already covered by nvidia-smi
            if vid == "0x10de":
                continue

            # Skip Intel cards - already covered by _get_intel_gpus
            if vid == "0x8086":
                continue

            # Read name
            name = None
            name_file = p / "name"
            try:
                name = name_file.read_text().strip()
            except Exception:
                pass

            if not name:
                try:
                    vendor = vendor_map.get(vid, "Unknown")
                    drv_link = p / "driver"
                    drv_name = drv_link.resolve().name if drv_link.exists() else ""
                    name = f"{vendor} {drv_name}"
                except Exception:
                    continue

            temp = None
            hwmon_base = card_dir.parent / "hwmon"
            if not hwmon_base.exists():
                hwmon_base = Path(card_dir).parent / "hwmon"
            if hwmon_base.exists():
                for tz in glob.glob(str(hwmon_base) + "/thermal_zone*/temp*"):
                    try:
                        temp = round(int(Path(tz).read_text().strip()) / 1000, 1)
                        break
                    except Exception:
                        continue

            if temp is None and hwmon_base.exists():
                for tp in glob.glob(str(hwmon_base) + "/temp*_input"):
                    try:
                        temp = round(int(Path(tp).read_text().strip()) / 1000, 1)
                        break
                    except Exception:
                        continue

            fan_speed = None
            if hwmon_base.exists():
                for fp in glob.glob(str(hwmon_base) + "/fan_*_input"):
                    try:
                        fan_speed = int(Path(fp).read_text().strip())
                        break
                    except Exception:
                        continue

            results.append({
                "name": name,
                "index": -1,
                "utilization": None,
                "memory_used": None,
                "memory_total": None,
                "memory_percent": None,
                "temperature": temp,
                "power": None,
                "fan_speed": fan_speed,
            })
    except Exception:
        pass
    return results


def collect_gpu_metrics():
    nv = _get_nvidia_gpus()
    intel = _get_intel_gpus()
    other = _get_other_gpus()

    seen_names = {g["name"] for g in nv} | {g["name"] for g in intel}
    result = list(nv) + list(intel)

    offset = len(result)
    for i, g in enumerate(other):
        if g["name"] not in seen_names:
            g["index"] = offset + i
            result.append(g)

    return result


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
        if prev is not None and isinstance(prev, dict):
            dt = now - prev["timestamp"]
            if dt > 0:
                rates["bytes_sent_per_sec"] = round((v.bytes_sent - prev["bytes_sent"]) / dt, 2)
                rates["bytes_recv_per_sec"] = round((v.bytes_recv - prev["bytes_recv"]) / dt, 2)

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

    gpu_hist: dict[int, dict] = {}
    for s in snapshots:
        for gpu in s.get("gpus", []):
            idx = gpu.get("index", 0)
            if idx not in gpu_hist:
                gpu_hist[idx] = {
                    "utilization": [],
                    "temperature": [],
                    "memory_percent": [],
                    "gt_cur_freq_mhz": [],
                    "gt_min_freq_mhz": [],
                    "gt_max_freq_mhz": [],
                    "rc6_residency": [],
                }
            gpu_hist[idx]["utilization"].append(gpu.get("utilization"))
            gpu_hist[idx]["temperature"].append(gpu.get("temperature"))
            gpu_hist[idx]["memory_percent"].append(gpu.get("memory_percent"))
            gpu_hist[idx]["gt_cur_freq_mhz"].append(gpu.get("gt_cur_freq_mhz"))
            gpu_hist[idx]["gt_min_freq_mhz"].append(gpu.get("gt_min_freq_mhz"))
            gpu_hist[idx]["gt_max_freq_mhz"].append(gpu.get("gt_max_freq_mhz"))
            gpu_hist[idx]["rc6_residency"].append(gpu.get("rc6_residency"))

    gpu_series = {}
    for k, v in gpu_hist.items():
        gpu_series[str(k)] = v

    return {
        "timestamps": timestamps,
        "cpu_percent": cpu_percents,
        "mem_percent": mem_percents,
        "net_bytes_sent": net_sent,
        "net_bytes_recv": net_recv,
        "gpu_history": gpu_series,
    }
