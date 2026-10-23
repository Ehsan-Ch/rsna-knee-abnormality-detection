"""Rebuild clean Kaggle notebooks without mutating recovered snapshots."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "v4": "Ehsan_RSNA_V4_Optimized.ipynb",
    "v5": "Ehsan_RSNA_V5_Three_Offset_Experimental.ipynb",
}

def normalize(nb, version):
    nb = copy.deepcopy(nb)
    for cell in nb["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    for key in ("papermill", "ehsan_experiment", "ehsan_validation"):
        nb["metadata"].pop(key, None)
    nb["metadata"]["kaggle"].update(isGpuEnabled=True, isInternetEnabled=False,
                                    accelerator="nvidiaTeslaT4")
    nb["metadata"]["ehsan_project"] = {
        "version": version, "packaged_on": "2026-10-23",
        "gpu_run_during_packaging": False,
        "historical_public_score": 0.958 if version == "v4" else None,
        "status": "historically_scored" if version == "v4" else "experimental_unscored",
        "validation": "See evidence/local_validation.json and docs/METHODS.md",
    }
    return nb

def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=ROOT) as tmp:
        assert Path(tmp).resolve().is_relative_to(ROOT.resolve())
        for version, name in NAMES.items():
            work = Path(tmp) / version
            shutil.copytree(ROOT / "experiments" / version, work,
                            ignore=shutil.ignore_patterns("__pycache__"))
            builder = "build.py" if version == "v4" else "build_v5.py"
            subprocess.run([sys.executable, str(work / builder)], cwd=work, check=True)
            built = json.loads((work / name).read_text(encoding="utf-8-sig"))
            archived = json.loads((ROOT / "experiments" / version / name).read_text(encoding="utf-8-sig"))
            assert [c["source"] for c in built["cells"] if c["cell_type"] == "code"] == [
                c["source"] for c in archived["cells"] if c["cell_type"] == "code"
            ], f"{version} builder changed executable notebook code"
            # Preserve the delivered explanation; only execution metadata is normalized.
            clean = normalize(archived, version)
            (ROOT / "notebooks" / name).write_text(
                json.dumps(clean, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"Built notebooks/{name}; executable cells match recovered snapshot")

if __name__ == "__main__":
    main()
