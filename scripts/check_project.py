"""Run existing tests and verify archive and notebook integrity. No GPU inference."""
import ast
import base64
import hashlib
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    results = []
    for version, test, count in (("v4", "test_revision.py", 7), ("v5", "test_v5.py", 8)):
        run = subprocess.run([sys.executable, test, "-v"], cwd=ROOT / "experiments" / version,
                             capture_output=True, text=True)
        print(run.stdout, end="")
        print(run.stderr, end="")
        assert run.returncode == 0, f"{version} tests failed"
        assert f"Ran {count} tests" in run.stderr
        results.append({"version": version, "tests_passed": count, "scope": "local_synthetic_or_mocked"})
    manifest = json.loads((ROOT / "evidence/import_manifest.json").read_text())
    for item in manifest["files"]:
        content = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"], item["path"]
    cells_checked = sources_checked = 0
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["metadata"]["kaggle"]["isGpuEnabled"]
        assert not nb["metadata"]["kaggle"]["isInternetEnabled"]
        assert "papermill" not in nb["metadata"]
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and not cell["outputs"]
            source = "".join(cell["source"])
            if source.startswith("%%writefile"):
                source = source.split("\n", 1)[1]
            ast.parse(source, filename=f"{path.name}:cell{i}")
            cells_checked += 1
            match = re.search(r"_b64.b64decode\('([^']+)'\)", source)
            if match:
                for name, code in json.loads(zlib.decompress(base64.b64decode(match[1]))).items():
                    ast.parse(code, filename=name)
                    sources_checked += 1
    assert cells_checked > 0 and sources_checked > 0
    patterns = {
        "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
        "github_fine_grained_token": r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "aws_access_id": r"\bAKIA[A-Z0-9]{16}\b",
    }
    scanned = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in (".git", "__pycache__") for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in (".py", ".ipynb", ".json", ".md", ".txt", ".cjs"):
            continue
        text = path.read_text(encoding="utf-8-sig")
        for label, pattern in patterns.items():
            assert not re.search(pattern, text), f"Potential {label} in {path.relative_to(ROOT)}"
        scanned += 1
    report = {
        "date": "2026-10-23", "status": "passed",
        "python": sys.version.split()[0], "tests": results,
        "tests_passed": sum(x["tests_passed"] for x in results),
        "imported_hashes_verified": len(manifest["files"]),
        "entry_notebook_code_cells_parsed": cells_checked,
        "embedded_sources_parsed": sources_checked,
        "text_files_credential_scanned": scanned,
        "credential_scan_scope": "Common token and private-key signatures; not an exhaustive security audit",
        "gpu_inference_run": False, "v5_official_score": None
    }
    (ROOT / "evidence/local_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
