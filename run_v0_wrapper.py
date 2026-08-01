#!/usr/bin/env python3
"""Wrapper to run V0 with proper output capture."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medqa_usmle.variants import v0_direct
from medqa_usmle.configs import load_config

config = load_config()
print(f"[{__import__('datetime').datetime.now()}] Config: model={config['model']['name']}, num_predict={config['model'].get('num_predict')}")

results = v0_direct.run_v0(split="test")

print(f"[{__import__('datetime').datetime.now()}] V0 complete: {len(results)} questions")
