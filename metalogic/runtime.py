from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG  = yaml.safe_load(open(ROOT / "configs" / "runtime.local.yaml"))

PRJ  = Path(CFG["project_root"])
INP  = PRJ / CFG["inputs_dir"]
OUT  = PRJ / CFG["outputs_dir"]
BUCKET = CFG.get("gcs_bucket", "")
USE_VISION = bool(CFG.get("use_vision", False))

def summary():
    print(f"Project root: {PRJ}")
    print(f"Inputs:       {INP}")
    print(f"Outputs:      {OUT}")
    print(f"Bucket:       {BUCKET}")
    print(f"USE_VISION:   {USE_VISION}")
