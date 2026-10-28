#!/usr/bin/env python3
import os
import sys

print("=== Starting DevOps Bot Component Tests ===")

# Test 1: Import core dependencies and NVML
try:
    import discord
    import requests
    import pynvml
    print("[PASS] Core Python dependencies & pynvml imported successfully.")
except ImportError as e:
    print(f"[FAIL] Missing dependency import: {e}")
    sys.exit(1)

# Import functions from bot.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import bot
    print("[PASS] Successfully imported bot.py module.")
except Exception as e:
    print(f"[FAIL] Could not import bot.py: {e}")
    sys.exit(1)

# Test 2: Test Docker Socket API & Demuxer
print("\n--- Testing Docker Socket API ---")
try:
    containers = bot.get_docker_containers(all_containers=True)
    print(f"[PASS] get_docker_containers() returned:\n{containers[:300]}...")
except Exception as e:
    print(f"[FAIL] Docker containers check failed: {e}")

try:
    stats = bot.get_docker_stats()
    print(f"[PASS] get_docker_stats() returned:\n{stats[:300]}...")
except Exception as e:
    print(f"[FAIL] Docker stats check failed: {e}")

# Test 3: Test System & GPU Telemetry
print("\n--- Testing System & GPU Telemetry ---")
try:
    sys_stats = bot.get_lxc_and_gpu_stats()
    print(f"[PASS] get_lxc_and_gpu_stats() returned:\n{sys_stats}")
except Exception as e:
    print(f"[FAIL] System/GPU stats check failed: {e}")

# Test 4: Test Kubernetes / Binary Checks
print("\n--- Testing Kubernetes & CLI Checks ---")
try:
    k8s_res = bot.get_k8s_resources("pods", namespace="all")
    print(f"[PASS] get_k8s_resources() executed. Output length: {len(k8s_res)} chars")
except Exception as e:
    print(f"[FAIL] Kubernetes check failed: {e}")

print("\n=== All Local Tests Completed ===")
