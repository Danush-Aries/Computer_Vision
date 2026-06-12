"""
Pytest configuration for the Gojo Hand Tracking test suite.

This conftest is loaded before any test module. It sets MPLBACKEND=Agg and
patches the mediapipe font-manager path so tests run headlessly in CI and on
machines where the system font cache may be in an unexpected state.
"""

import os
import sys
import types
import unittest.mock as mock

# -----------------------------------------------------------------------
# Force headless matplotlib backend before anything imports matplotlib.
# mediapipe's drawing_utils imports matplotlib at module level, which tries
# to initialise the GUI font manager — this crashes on some macOS machines
# with a KeyError: '_items' in the font cache.  Setting the backend to Agg
# before the first import prevents the crash.
# -----------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")

# -----------------------------------------------------------------------
# Monkey-patch matplotlib.font_manager._get_macos_fonts so that even if
# the font cache JSON is malformed the import succeeds.
# -----------------------------------------------------------------------

def _patch_matplotlib_font_manager():
    try:
        import matplotlib.font_manager as fm
        original = fm._get_macos_fonts

        def _safe_get_macos_fonts():
            try:
                return original()
            except (KeyError, TypeError):
                return []

        fm._get_macos_fonts = _safe_get_macos_fonts
    except Exception:
        pass


_patch_matplotlib_font_manager()
