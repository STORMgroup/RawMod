"""Run one featurization step here, or submit it to a scheduler.

Every script in this directory builds one shell command per dataset and hands it to
`launch()`. With `--local`, with `RAWMOD_LOCAL=1`, or whenever `sbatch` is not on the
PATH, the command runs in this process and the datasets are done one after another.
Otherwise it is submitted, and the resources come from the site configuration
(`RAWMOD_SBATCH_CPU` and `RAWMOD_SBATCH_GPU` in the environment or `paths.env`), so no
partition or node name is written into these scripts.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

_PATHS_LOADED: dict[str, str] | None = None


def _site(name: str) -> str:
    """Value of a setting from the environment or the site file, '' if unset."""
    global _PATHS_LOADED
    v = os.environ.get(name)
    if v:
        return v
    if _PATHS_LOADED is None:
        _PATHS_LOADED = {}
        p = Path(os.environ.get('RAWMOD_PATHS_FILE',
                                Path.home() / '.config' / 'rawmod' / 'paths.env'))
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, val = line.split('=', 1)
                    _PATHS_LOADED[re.sub(r'^export\s+', '', k.strip()).strip()] = \
                        os.path.expandvars(val.strip().strip('"\''))
    v = _PATHS_LOADED.get(name, '')
    return '' if v.startswith('$') else v


def use_local(flag: bool = False) -> bool:
    """True when the work should run here instead of being submitted."""
    return bool(flag) or os.environ.get('RAWMOD_LOCAL') == '1' or shutil.which('sbatch') is None


def launch(wrap, *, name, log, mem=None, cpus=8, time=None, dependency=None,
           dependency_type='afterok', kind='cpu', dry=False, local=False):
    """Run or submit one step. Returns the job id, or '' when it ran here."""
    local = use_local(local)
    log = str(log)
    if local:
        log = log.replace('%j', 'local').replace('%x', name)
        if dry:
            print(f'[dry-run] {name}: {wrap}\n')
            return ''
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        print(f'  {name}: running here, log {log}', flush=True)
        r = subprocess.run(['bash', '-o', 'pipefail', '-c',
                            f'{wrap} 2>&1 | tee {shlex.quote(log)}'])
        if r.returncode:
            raise SystemExit(f'{name} failed with status {r.returncode}; see {log}')
        return ''
    cmd = ['sbatch', '--parsable', f'--job-name={name}', f'--output={log}', f'--error={log}',
           '--ntasks=1', f'--cpus-per-task={cpus}']
    cmd += shlex.split(_site('RAWMOD_SBATCH_GPU' if kind == 'gpu' else 'RAWMOD_SBATCH_CPU'))
    if mem:
        cmd.append(f'--mem={mem}')
    if time:
        cmd.append(f'--time={time}')
    if dependency:
        cmd.append(f'--dependency={dependency_type}:{dependency}')
    cmd += ['--wrap', wrap]
    if dry:
        print(f"[dry-run] {' '.join(shlex.quote(c) for c in cmd)}\n")
        return ''
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'sbatch failed for {name}: {r.stderr.strip()}')
    return r.stdout.strip()
