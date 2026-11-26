"""Shared pytest configuration.

Hypothesis profiles:
- ``default``: at least 100 examples per property (design requirement), no deadline so
  slow CI machines do not produce flaky timing failures.
- ``ci``: same, derandomized so every run explores the same examples.

Select a profile with ``HYPOTHESIS_PROFILE=ci``.
"""

from __future__ import annotations

import os

from hypothesis import settings

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=100, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
