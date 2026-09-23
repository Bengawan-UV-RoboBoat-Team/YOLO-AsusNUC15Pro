"""OpenVINO device discovery for CPU / integrated GPU / NPU targets."""

from __future__ import annotations

# Maps the short device key used throughout this project to the device
# string Ultralytics expects when selecting an OpenVINO inference target.
TARGET_DEVICES: dict[str, str] = {
    "CPU": "intel:cpu",
    "GPU": "intel:gpu",
    "NPU": "intel:npu",
}


def probe_openvino_devices() -> dict[str, str]:
    """Return {device_key: full_device_name} for every OpenVINO device that
    is physically available on this machine. A device key absent from the
    result means it wasn't detected (e.g. NPU driver not installed).
    """
    from openvino import Core

    core = Core()
    available: dict[str, str] = {}
    for device_key in TARGET_DEVICES:
        if device_key not in core.available_devices:
            continue
        try:
            full_name = core.get_property(device_key, "FULL_DEVICE_NAME")
        except Exception:
            full_name = device_key
        available[device_key] = full_name
    return available


def resolve_run_devices(requested: list[str] | None = None) -> list[str]:
    """Intersect the devices the user asked to benchmark with what's
    actually detected, printing a clear skip reason for anything missing.
    Returns the device keys (e.g. ["CPU", "NPU"]) to actually run.
    """
    wanted = [d.upper() for d in requested] if requested else list(TARGET_DEVICES)
    detected = probe_openvino_devices()

    resolved: list[str] = []
    for device_key in wanted:
        if device_key not in TARGET_DEVICES:
            print(f"[devices] ignoring unknown device '{device_key}' (expected one of {list(TARGET_DEVICES)})")
            continue
        if device_key in detected:
            print(f"[devices] using {device_key}: {detected[device_key]}")
            resolved.append(device_key)
        else:
            print(f"[devices] skipping {device_key}: not detected (driver missing or hardware absent)")
    return resolved
