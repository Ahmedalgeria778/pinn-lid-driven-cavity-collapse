# -*- coding: utf-8 -*-
"""Compatibility shim: delegate to the full figure regeneration."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "regen_all_figures.py")]))