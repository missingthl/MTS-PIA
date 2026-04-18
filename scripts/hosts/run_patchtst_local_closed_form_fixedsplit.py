#!/usr/bin/env python
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from scripts.hosts.run_tsl_local_closed_form_fixedsplit import main


if __name__ == "__main__":
    main(default_host_backbone="patchtst")
