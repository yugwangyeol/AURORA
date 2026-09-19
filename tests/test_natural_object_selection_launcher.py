import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('parallel', [False,True])
def test_launch_order_and_gpu_isolation(tmp_path,parallel):
    script=tmp_path/'fake_python'
    script.write_text('''#!/usr/bin/env python3
import json,os,sys
a=sys.argv
stage=a[a.index('--stage')+1]
condition=a[a.index('--condition')+1] if '--condition' in a else None
print('EVENT:'+json.dumps({'stage':stage,'condition':condition,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                         'pythonpath':os.environ['PYTHONPATH']}),flush=True)
''')
    script.chmod(0o755)
    for name in ('deor','selector'):
        folder=tmp_path/name;folder.mkdir();(folder/'config.json').write_text('{}')
    env=dict(os.environ,DEOR_PYTHON=str(script),SELECTOR_PYTHON=str(script),GPU='0',
             SELECTOR_GPUS='0,1' if parallel else '0',MODEL_PATH=str(tmp_path/'deor'),
             SELECTOR_MODEL=str(tmp_path/'selector'),OUTPUT_DIR=str(tmp_path/'results'),
             SELECTOR_SITE_PACKAGES=str(tmp_path/'selector_packages'))
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['/bin/bash',str(root/'scripts/run_pgot_natural_object_selection.sh')],
                          env=env,text=True,capture_output=True,check=True)
    events=[json.loads(line[6:]) for line in result.stdout.splitlines() if line.startswith('EVENT:')]
    assert [r['stage'] for r in events]==['prepare','extract','select','select','score']
    selections={r['condition']:r['gpu'] for r in events if r['stage']=='select'}
    assert selections=={'category':'0','description':'1' if parallel else '0'}
    assert all(r['gpu']=='0' for r in events if r['stage']!='select')
    assert all(r['pythonpath'].startswith(str(tmp_path/'selector_packages')+':')
               for r in events if r['stage']=='select')
    assert all(str(tmp_path/'selector_packages') not in r['pythonpath']
               for r in events if r['stage']!='select')
    assert (tmp_path/'results/logs/select_category.log').is_file()
    assert (tmp_path/'results/logs/select_description.log').is_file()


def test_parallel_selector_failure_propagates_without_scoring(tmp_path):
    script=tmp_path/'fake_python'
    script.write_text('''#!/usr/bin/env python3
import sys
a=sys.argv
stage=a[a.index('--stage')+1]
print('STAGE:'+stage,flush=True)
if '--condition' in a and a[a.index('--condition')+1]=='category': sys.exit(7)
''')
    script.chmod(0o755)
    for name in ('deor','selector'):
        folder=tmp_path/name;folder.mkdir();(folder/'config.json').write_text('{}')
    env=dict(os.environ,DEOR_PYTHON=str(script),SELECTOR_PYTHON=str(script),GPU='0',SELECTOR_GPUS='0,1',
             MODEL_PATH=str(tmp_path/'deor'),SELECTOR_MODEL=str(tmp_path/'selector'),OUTPUT_DIR=str(tmp_path/'results'))
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['/bin/bash',str(root/'scripts/run_pgot_natural_object_selection.sh')],
                          env=env,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'STAGE:score' not in result.stdout
    assert 'Selector failed' in result.stderr
