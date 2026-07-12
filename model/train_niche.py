"""Train wrapper — sets NICHE env var so we don't need to edit model_train.py."""
import os, sys, runpy

os.environ["TRAIN_NICHE"] = sys.argv[1] if len(sys.argv) > 1 else "fitness"
# Patch the NICHE constant at import time via monkeypatching the module globals
import builtins
_real_compile = builtins.compile

def _patched_compile(source, filename, mode, *args, **kwargs):
    if isinstance(source, str) and 'NICHE      = "fitness"' in source:
        source = source.replace(
            'NICHE      = "fitness"',
            f'NICHE      = "{os.environ["TRAIN_NICHE"]}"'
        )
    return _real_compile(source, filename, mode, *args, **kwargs)

builtins.compile = _patched_compile

runpy.run_path("model_train.py", run_name="__main__")
