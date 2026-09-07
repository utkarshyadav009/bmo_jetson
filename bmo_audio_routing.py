#!/usr/bin/env python3
"""BMO Audio Routing & Endpoint Auto-Configuration Manager.

Handles dynamic detection and configuration between:
1. Bluetooth Headsets (e.g., Sony WH-1000XM4 HFP/A2DP)
2. USB Speaker & Microphone Arrays (e.g., Seeed ReSpeaker 4 Mic Array)
3. System default audio endpoints
"""

import sys
import subprocess
import sounddevice as sd
import numpy as np
from typing import Tuple, Optional, Dict, Any


def get_pactl_endpoints() -> Tuple[list, list]:
    """Return lists of available PulseAudio sink and source names."""
    try:
        sink_lines = subprocess.getoutput("pactl list short sinks").splitlines()
        source_lines = subprocess.getoutput("pactl list short sources").splitlines()
        sinks = [l.split()[1] for l in sink_lines if len(l.split()) >= 2]
        sources = [l.split()[1] for l in source_lines if len(l.split()) >= 2]
        return sinks, sources
    except Exception as e:
        print(f"[!] Warning: Failed to query pactl endpoints: {e}", file=sys.stderr)
        return [], []


def detect_and_configure_audio(
    mode: str = "auto",
    force_headphone: Optional[bool] = None,
    desired_volume_pct: int = 80
) -> Dict[str, Any]:
    """Detects active audio devices and configures PulseAudio default endpoints.

    Args:
        mode: 'auto', 'bluetooth', 'usb', or 'default'.
        force_headphone: If True/False, overrides auto-detected is_headphone state.
        desired_volume_pct: Volume percentage to set on the active sink (default 80%).

    Returns:
        dict with keys:
          'in_dev': sounddevice input device index (PulseAudio)
          'out_dev': sounddevice output device index (PulseAudio)
          'is_headphone': bool (True for headphones -> enable duplex barge-in)
          'desc': str (Human-readable description of endpoint)
          'sink': str (PulseAudio sink name)
          'source': str (PulseAudio source name)
          'mode': str ('bluetooth', 'usb', 'system_default')
    """
    if mode in ("bluetooth", "auto"):
        # If bluetooth card is in a2dp_sink mode, automatically switch to handsfree_head_unit (duplex mic + speaker)
        try:
            card_lines = subprocess.getoutput("pactl list short cards").splitlines()
            for cl in card_lines:
                parts = cl.split()
                if len(parts) >= 2 and "bluez_card" in parts[1]:
                    subprocess.run(["pactl", "set-card-profile", parts[1], "handsfree_head_unit"], capture_output=True, timeout=2)
                    import time
                    time.sleep(0.3)  # Give PulseAudio 300ms to register endpoints
                    break
        except Exception:
            pass

    sinks, sources = get_pactl_endpoints()

    # Filter Bluetooth endpoints
    bt_sinks = [s for s in sinks if "bluez_sink" in s]
    bt_sources = [s for s in sources if "bluez_source" in s]

    # Filter USB endpoints (excluding monitor sources)
    usb_sinks = [s for s in sinks if any(k in s.lower() for k in ("respeaker", "seeed", "usb-", "usb_", "uac"))]
    usb_sources = [
        s for s in sources
        if any(k in s.lower() for k in ("respeaker", "seeed", "usb-", "usb_", "uac"))
        and not s.endswith(".monitor")
    ]

    selected_sink = None
    selected_source = None
    endpoint_type = "system_default"
    endpoint_desc = ""
    is_headphone = False

    # Resolution logic based on requested mode
    if mode == "bluetooth" or (mode == "auto" and bt_sinks and bt_sources):
        if bt_sinks and bt_sources:
            selected_sink = bt_sinks[0]
            selected_source = bt_sources[0]
            endpoint_type = "bluetooth"
            is_headphone = True
            mac_part = selected_sink.split(".")[1] if "." in selected_sink else selected_sink
            endpoint_desc = f"Bluetooth Headset ({mac_part})"
        else:
            print("[!] Warning: Bluetooth mode requested but bluez sink/source not found.", file=sys.stderr)

    if (mode == "usb" and selected_sink is None) or (mode == "auto" and selected_sink is None and (usb_sinks or usb_sources)):
        if usb_sinks:
            selected_sink = usb_sinks[0]
        if usb_sources:
            selected_source = usb_sources[0]
        endpoint_type = "usb"
        is_headphone = False
        endpoint_desc = f"USB Speaker/Mic ({selected_sink or selected_source})"

    if selected_sink is None and selected_source is None:
        # Fallback to existing PulseAudio default
        curr_sink = subprocess.getoutput("pactl info | grep 'Default Sink:'").replace("Default Sink:", "").strip()
        curr_source = subprocess.getoutput("pactl info | grep 'Default Source:'").replace("Default Source:", "").strip()
        selected_sink = curr_sink if curr_sink else (sinks[0] if sinks else None)
        selected_source = curr_source if curr_source else (sources[0] if sources else None)
        endpoint_type = "default"
        if selected_sink and any(k in selected_sink.lower() for k in ("bluez", "headphone", "headset")):
            is_headphone = True
            endpoint_desc = f"Default Bluetooth/Headphone ({selected_sink})"
        else:
            is_headphone = False
            endpoint_desc = f"System Default ({selected_sink or 'Default'})"

    if force_headphone is not None:
        is_headphone = force_headphone

    # Apply default sink/source to PulseAudio
    if selected_sink:
        subprocess.run(["pactl", "set-default-sink", selected_sink], check=False, stderr=subprocess.DEVNULL)
        if desired_volume_pct:
            subprocess.run(["pactl", "set-sink-volume", selected_sink, f"{desired_volume_pct}%"], check=False, stderr=subprocess.DEVNULL)
    if selected_source:
        subprocess.run(["pactl", "set-default-source", selected_source], check=False, stderr=subprocess.DEVNULL)
        subprocess.run(["pactl", "set-source-volume", selected_source, "100%"], check=False, stderr=subprocess.DEVNULL)

    # Locate PulseAudio device index in sounddevice
    devs = sd.query_devices()
    pulse_idx = None
    for i, d in enumerate(devs):
        if d["name"].lower() == "pulse":
            pulse_idx = i
            break
    if pulse_idx is None:
        for i, d in enumerate(devs):
            if "pulse" in d["name"].lower():
                pulse_idx = i
                break
    if pulse_idx is None:
        for i, d in enumerate(devs):
            if "default" in d["name"].lower():
                pulse_idx = i
                break
    if pulse_idx is None:
        pulse_idx = 0

    return {
        "in_dev": pulse_idx,
        "out_dev": pulse_idx,
        "is_headphone": is_headphone,
        "desc": endpoint_desc,
        "sink": selected_sink,
        "source": selected_source,
        "type": endpoint_type,
    }


def self_test_audio(duration: float = 1.0) -> bool:
    """Performs quick acoustic playback tone and microphone capture test."""
    config = detect_and_configure_audio()
    print("-------------------------------------------------------")
    print(f"[*] Testing Audio Endpoint: {config['desc']}")
    print(f"[*] Sink:    {config['sink']}")
    print(f"[*] Source:  {config['source']}")
    print(f"[*] Headset: {'Yes (Barge-in enabled)' if config['is_headphone'] else 'No (Desk speaker)'}")
    print("-------------------------------------------------------")

    dev_idx = config["out_dev"]

    # 1. Test Tone Output
    print("[1/2] Playing test tone (440Hz beep for 0.4s)...", flush=True)
    t = np.linspace(0, 0.4, int(24000 * 0.4), endpoint=False, dtype=np.float32)
    tone = 0.15 * np.sin(2 * np.pi * 440 * t)
    try:
        sd.play(tone, samplerate=24000, device=dev_idx)
        sd.wait()
        print("      Tone playback completed successfully.")
    except Exception as e:
        print(f"      [!] Playback failed: {e}")
        return False

    # 2. Test Microphone Capture
    print(f"[2/2] Recording {duration:.1f}s microphone audio...", flush=True)
    try:
        in_data = sd.rec(int(16000 * duration), samplerate=16000, channels=1, device=dev_idx, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(in_data ** 2)))
        peak = float(np.max(np.abs(in_data)))
        print(f"      Microphone capture successful! RMS: {rms:.5f}, Peak: {peak:.5f}")
        return True
    except Exception as e:
        print(f"      [!] Capture failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BMO Audio Routing Utility")
    parser.add_argument("--test", action="store_true", help="Run acoustic test tone & microphone capture")
    parser.add_argument("--mode", choices=["auto", "bluetooth", "usb", "default"], default="auto", help="Endpoint selection mode")
    parser.add_argument("--headphone", action="store_true", default=None, help="Force headphone mode")
    parser.add_argument("--speaker", dest="headphone", action="store_false", help="Force speaker mode")
    args = parser.parse_args()

    cfg = detect_and_configure_audio(mode=args.mode, force_headphone=args.headphone)
    print("BMO Active Audio Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.test:
        print()
        self_test_audio()
