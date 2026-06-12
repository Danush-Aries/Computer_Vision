"""
Root conftest.py — runs before any test collection.

Workaround for a broken macOS font cache (KeyError: '_items') that causes
matplotlib.font_manager to crash at import time, which in turn prevents
mediapipe (which imports matplotlib.pyplot at module level) from being
imported during test collection.

The fix: intercept the import of matplotlib.font_manager via a custom
meta-path finder, execute the real module, then silently replace the
buggy _get_macos_fonts function with a safe version before the broken
call site executes.
"""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types


# Tell matplotlib to use the non-interactive Agg backend so no display
# or font rendering is needed.
os.environ["MPLBACKEND"] = "Agg"


class _FontManagerPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """
    Intercepts the import of matplotlib.font_manager and patches
    _get_macos_fonts to tolerate a broken system font cache.
    """

    _target = "matplotlib.font_manager"

    def find_module(self, fullname, path=None):  # legacy finder API
        return self if fullname == self._target else None

    def find_spec(self, fullname, path, target=None):
        if fullname != self._target:
            return None
        # Remove ourselves temporarily to avoid infinite recursion
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        spec.loader = self
        return spec

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        # Remove ourselves so the real loader can run without hitting us again
        sys.meta_path.remove(self)
        try:
            loader = importlib.util.find_spec(module.__name__).loader
            loader.exec_module(module)
        finally:
            pass  # don't re-add; we only need to patch once

        # Patch _get_macos_fonts so a missing '_items' key is handled
        _orig = getattr(module, "_get_macos_fonts", None)
        if _orig is not None:
            def _safe_get_macos_fonts():
                try:
                    return _orig()
                except (KeyError, TypeError, Exception):
                    return []
            module._get_macos_fonts = _safe_get_macos_fonts

        # Re-run FontManager initialisation with the safe function in place
        # (it may have already run if the module was cached; if so, skip)
        if not hasattr(module, "fontManager"):
            try:
                module.fontManager = module._load_fontmanager()
            except Exception:
                pass


# Register patcher as the first meta-path finder so it runs before the
# normal finders when matplotlib.font_manager is first imported.
sys.meta_path.insert(0, _FontManagerPatcher())
