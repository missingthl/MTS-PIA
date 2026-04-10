
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_phase13c4_step1_activation import run_worker, CONFIG

print("Launching Probe Run for Seed 0 BandGate (1 Epoch)...")
CONFIG['epochs'] = 1
# Force 1 epoch
try:
    run_worker(0, "bandgate", True)
except SystemExit:
    pass # Expected exit from runner usually? Runner returns dict, run_worker might exit if exception.
except Exception as e:
    print(f"Probe Run Exception: {e}")
    # We don't care about training success, only probe dump.
