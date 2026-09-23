"""Package explicit public deliverables only; never include .env or raw data."""
from pathlib import Path
import hashlib
import json
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
files = ['out/nodes_roles.csv', 'out/clusters.csv', 'out/top_nodes.csv',
         'out/run_metadata.json', 'README.md', 'README.en.md',
         'docs/scheme.svg', 'docs/scheme.md', 'docs/demo.md',
         'docs/cluster-hypotheses.md', 'docs/release-check.md', 'requirements.lock.txt']
for name in files:
    if not (root / name).is_file():
        raise SystemExit('Missing: ' + name)
manifest = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'working_tree_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
            'sha256': {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
path = root / 'out/submission.zip'
with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(root/name, name)
    archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
print(path)
