"""Minimal explicit source configuration for isolated build/staging fixtures."""
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]

def seed_sources(root):
    original=json.loads((ROOT/'config/sources.json').read_text())
    stable=dict(original['sources']['upstream-stable'])
    stable['profiles']=list(json.loads((root/'config/profiles.json').read_text()))
    (root/'config/sources.json').write_text(json.dumps({'formatVersion':1,'defaultSource':'upstream-stable','sources':{'upstream-stable':stable}}))
    plan=root/stable['patchSeries'];plan.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/stable['patchSeries'],plan)
    for item in json.loads(plan.read_text())['patches']:
        dest=root/item['file'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/item['file'],dest)
