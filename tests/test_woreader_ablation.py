"""Exercise the actual shell chain without launching GPUs or loading weights."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pgot.eval.verify_woreader_config import verify


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'scripts/run_pgot_oneshot_memory8_register32_noprior_woreader.sh'
CONFIG = dict(
    pgot_e8_owner_weight=1.0, pgot_e8_reader_supervision_mode='none',
    pgot_e8_reader_object_weight=0.0, pgot_e8_reader_background_weight=0.0,
    pgot_e11_object_memories_per_owner=8, pgot_e11_register_memories_per_owner=32,
    pgot_n_register=4, pgot_one_shot_reader_enable=True,
    pgot_one_shot_readout_mode='memory_content', pgot_one_shot_writer_softmax_axis='patch',
    pgot_one_shot_writer_owner_prior=False, pgot_lm_loss_weight=1.0,
    pgot_recon_loss_weight=1.5, hidden_size=1536,
)


@pytest.mark.parametrize('change', [{}, {'pgot_e8_owner_weight': 0.0},
    {'pgot_e8_reader_supervision_mode': 'writer'}, {'pgot_e8_reader_object_weight': 0.5},
    {'pgot_one_shot_direct_rae_query': True}, {'pgot_one_shot_memory_contrastive_enable': True},
    {'hidden_size': 3584}, {'pgot_one_shot_memory_value_source': 'dino'}])
def test_checkpoint_audit(tmp_path, change):
    (tmp_path / 'config.json').write_text(json.dumps(dict(CONFIG, **change)))
    if change:
        with pytest.raises(ValueError):
            verify(tmp_path)
    else:
        verify(tmp_path)


def launcher_env(tmp_path):
    fake = tmp_path / 'python'
    fake.write_text('''#!'''+sys.executable+'''
import json, os, pathlib, sys
a = sys.argv[1:]
def get(k): return a[a.index(k)+1]
train = 'torch.distributed.run' in a
mode = 'train' if train else get('--caption_mode')
print('EVENT:'+json.dumps(dict(mode=mode,gpu=os.environ['CUDA_VISIBLE_DEVICES'],args=a)),flush=True)
if os.environ.get('FAIL_MODE') == mode: sys.exit(9)
if train:
    c=json.loads(os.environ['TEST_CONFIG'])
    for k in c:
        flag='--'+k
        if flag not in a: continue
        v=get(flag)
        try: c[k]=json.loads(v.lower())
        except ValueError: c[k]=v
    p=pathlib.Path(get('--output_dir'))/('checkpoint-'+get('--max_steps'))
    p.mkdir(parents=True)
    (p/'config.json').write_text(json.dumps(c))
''')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=str(fake), AUDIT_PYTHON=sys.executable,
        OUTPUT_DIR=str(tmp_path / 'checkpoint'), EVAL_ROOT=str(tmp_path / 'eval'),
        CUDA_VISIBLE_DEVICES='2,3', TEST_CONFIG=json.dumps(CONFIG),
        E8_READER_SUPERVISION_MODE='writer', E8_READER_OBJECT_WEIGHT='0.5',
        E8_READER_BACKGROUND_WEIGHT='0.25')
    for k in ('RESUME_FROM_CHECKPOINT', 'FSDP_CONFIG', 'FAIL_MODE', 'MODEL_PATH'):
        env.pop(k, None)
    return env


def invoke(env, script=RUN):
    p = subprocess.run(['/bin/bash', str(script)], env=env, text=True, capture_output=True)
    events = [json.loads(line[6:]) for line in p.stdout.splitlines() if line.startswith('EVENT:')]
    return p, events


def test_actual_launcher_train_ar_without_tf(tmp_path):
    env = launcher_env(tmp_path)
    p, events = invoke(env)
    assert p.returncode == 0, p.stdout+p.stderr
    assert [e['mode'] for e in events] == ['train', 'autoregressive']
    assert [e['gpu'] for e in events] == ['2,3', '3']
    assert not (tmp_path/'eval/tf.log').exists()
    assert not (tmp_path/'eval/tf').exists()
    args = events[0]['args']
    def value(k): return args[args.index('--'+k)+1]
    assert value('pgot_e8_reader_supervision_mode') == 'none'
    assert value('pgot_e8_reader_object_weight') == '0.0'
    assert value('pgot_e8_reader_background_weight') == '0.0'
    assert value('pgot_e8_owner_weight') == '1.0'
    assert value('pgot_one_shot_direct_rae_query') == 'False'
    assert value('pgot_e11_object_memories_per_owner') == '8'
    assert value('pgot_e11_register_memories_per_owner') == '32'
    assert value('pgot_n_register') == '4'
    assert value('pgot_one_shot_memory_contrastive_enable') == 'False'
    assert value('model_name_or_path').endswith('pgot_e11_capacity_dit16/checkpoint-10000')
    for e, folder in zip(events[1:], ['ar']):
        a=e['args']
        assert a[a.index('--model_path')+1] == str(tmp_path/'checkpoint/checkpoint-5000')
        assert a[a.index('--output_dir')+1] == str(tmp_path/'eval'/folder)
        assert '--compute_rfid' in a and '--compute_kid' in a
    log=(tmp_path/'eval/train.log').read_bytes()
    rerun, extra=invoke(env)
    assert rerun.returncode == 2 and not extra
    assert (tmp_path/'eval/train.log').read_bytes() == log


@pytest.mark.parametrize('mode,expected', [('train',['train']),
    ('autoregressive',['train','autoregressive'])])
def test_errors_stop_downstream_stages(tmp_path, mode, expected):
    env=launcher_env(tmp_path)
    env['FAIL_MODE']=mode
    p, events=invoke(env)
    assert p.returncode != 0
    assert [e['mode'] for e in events] == expected


@pytest.mark.parametrize('gpus', ['2', '2,2', '0,1,2,3'])
def test_invalid_gpu_list_fails_before_launch(tmp_path, gpus):
    env=launcher_env(tmp_path)
    env['CUDA_VISIBLE_DEVICES']=gpus
    p, events=invoke(env)
    assert p.returncode == 2 and not events


def test_existing_baseline_defaults_unchanged(tmp_path):
    env=launcher_env(tmp_path)
    for k in ('E8_READER_SUPERVISION_MODE','E8_READER_OBJECT_WEIGHT','E8_READER_BACKGROUND_WEIGHT'):
        env.pop(k, None)
    p=subprocess.run(['/bin/bash', str(ROOT/'scripts/train_pgot_oneshot_memory8_register32_noprior.sh')],
                     env=env, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout+p.stderr
    a=json.loads(next(s[6:] for s in p.stdout.splitlines() if s.startswith('EVENT:')))['args']
    assert a[a.index('--pgot_e8_reader_supervision_mode')+1] == 'writer'
    assert a[a.index('--pgot_e8_reader_object_weight')+1] == '0.5'
    assert a[a.index('--pgot_e8_reader_background_weight')+1] == '0.25'


def test_eval_refuses_wrong_checkpoint_before_loading_gpu(tmp_path):
    env = launcher_env(tmp_path)
    model = tmp_path / 'wrong-checkpoint'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(CONFIG, pgot_e8_owner_weight=0.0)))
    env['MODEL_PATH'] = str(model)
    p = subprocess.run(['/bin/bash', str(ROOT / 'scripts/eval_pgot_oneshot_memory8_register32_noprior_woreader.sh')],
                       env=env, text=True, capture_output=True)
    assert p.returncode != 0
    assert 'Not the original 1.5B' in p.stderr
    assert 'EVENT:' not in p.stdout


WOLM_RUN = ROOT / 'scripts/run_pgot_oneshot_memory8_register32_noprior_wolm.sh'
WOLM_CONFIG = dict(CONFIG, pgot_lm_loss_weight=0.0, pgot_e8_reader_supervision_mode='writer',
                   pgot_e8_reader_object_weight=0.5, pgot_e8_reader_background_weight=0.25)


def wolm_env(tmp_path):
    env = launcher_env(tmp_path)
    env.update(CUDA_VISIBLE_DEVICES='0,1', TEST_CONFIG=json.dumps(WOLM_CONFIG),
               PGOT_LM_LOSS_WEIGHT='1.0', E8_READER_SUPERVISION_MODE='none',
               E8_READER_OBJECT_WEIGHT='0.0', E8_READER_BACKGROUND_WEIGHT='0.0')
    return env


@pytest.mark.parametrize('change', [{}, {'pgot_lm_loss_weight': 1.0},
    {'pgot_e8_reader_supervision_mode': 'none'}, {'pgot_e8_reader_object_weight': 0.0},
    {'pgot_e8_reader_background_weight': 0.0}, {'pgot_e8_owner_weight': 0.0},
    {'pgot_recon_loss_weight': 0.0}])
def test_wolm_checkpoint_audit(tmp_path, change):
    (tmp_path / 'config.json').write_text(json.dumps(dict(WOLM_CONFIG, **change)))
    if change:
        with pytest.raises(ValueError):
            verify(tmp_path, 'wolm')
    else:
        verify(tmp_path, 'wolm')


def test_wolm_launcher_removes_only_lm_loss_and_skips_tf(tmp_path):
    env = wolm_env(tmp_path)
    p, events = invoke(env, WOLM_RUN)
    assert p.returncode == 0, p.stdout + p.stderr
    assert [e['mode'] for e in events] == ['train', 'autoregressive']
    assert [e['gpu'] for e in events] == ['0,1', '1']
    assert not (tmp_path / 'eval/tf.log').exists()
    a = events[0]['args']
    values = {
        'pgot_lm_loss_weight': '0.0', 'pgot_recon_loss_weight': '1.5',
        'pgot_e8_owner_weight': '1.0', 'pgot_e8_reader_supervision_mode': 'writer',
        'pgot_e8_reader_object_weight': '0.5', 'pgot_e8_reader_background_weight': '0.25',
        'pgot_e11_object_memories_per_owner': '8', 'pgot_e11_register_memories_per_owner': '32',
        'pgot_n_register': '4', 'pgot_one_shot_direct_rae_query': 'False',
        'pgot_one_shot_rae_query_adapter_enable': 'False',
        'pgot_one_shot_memory_contrastive_enable': 'False',
        'pgot_one_shot_memory_value_source': 'siglip',
    }
    for key, expected in values.items():
        assert a[a.index('--' + key) + 1] == expected
    assert '--master_port=29674' in a
    assert a[a.index('--model_name_or_path') + 1].endswith('pgot_e11_capacity_dit16/checkpoint-10000')
    a = events[1]['args']
    assert a[a.index('--model_path') + 1] == str(tmp_path / 'checkpoint/checkpoint-5000')
    assert a[a.index('--output_dir') + 1] == str(tmp_path / 'eval/ar')
    assert '--compute_rfid' in a and '--compute_kid' in a
    log = (tmp_path / 'eval/train.log').read_bytes()
    rerun, events = invoke(env, WOLM_RUN)
    assert rerun.returncode == 2 and not events
    assert (tmp_path / 'eval/train.log').read_bytes() == log


@pytest.mark.parametrize('mode,expected', [('train', ['train']),
    ('autoregressive', ['train', 'autoregressive'])])
def test_wolm_failures_stop_pipeline(tmp_path, mode, expected):
    env = wolm_env(tmp_path)
    env['FAIL_MODE'] = mode
    p, events = invoke(env, WOLM_RUN)
    assert p.returncode != 0
    assert [e['mode'] for e in events] == expected


@pytest.mark.parametrize('gpus', ['0', '0,0', '0,1,2,3'])
def test_wolm_rejects_invalid_gpu_list(tmp_path, gpus):
    env = wolm_env(tmp_path)
    env['CUDA_VISIBLE_DEVICES'] = gpus
    p, events = invoke(env, WOLM_RUN)
    assert p.returncode == 2 and not events
