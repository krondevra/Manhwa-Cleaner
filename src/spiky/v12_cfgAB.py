import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replicate_pipeline_v12 as _m
_m.FIX_A = True; _m.FIX_B = True; _m.FIX_E = False
_m.FIX_S = False
def clean_page(rgb, steps="QS"):
    return _m.clean_page(rgb, steps=steps)
