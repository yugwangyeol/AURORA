"""Fail fast if a completed training checkpoint isn't the requested ablation."""
import argparse
import json
from pathlib import Path


def verify(folder, ablation='woreader', model_size='1.5b'):
    if ablation not in ('woreader', 'wolm'):
        raise ValueError(f'Unknown loss ablation: {ablation}')
    if model_size not in ('1.5b', '7b'):
        raise ValueError(f'Unknown model size: {model_size}')
    cfg = json.loads((Path(folder) / 'config.json').read_text())
    expected = dict(
        pgot_e8_owner_weight=1.0, pgot_e8_reader_supervision_mode='none',
        pgot_e8_reader_object_weight=0.0, pgot_e8_reader_background_weight=0.0,
        pgot_e11_object_memories_per_owner=8, pgot_e11_register_memories_per_owner=32,
        pgot_n_register=4, pgot_one_shot_reader_enable=True,
        pgot_one_shot_readout_mode='memory_content', pgot_one_shot_writer_softmax_axis='patch',
        pgot_one_shot_writer_owner_prior=False, pgot_lm_loss_weight=1.0,
        pgot_recon_loss_weight=1.5,
    )
    if model_size == '1.5b':
        expected.update(hidden_size=1536)
    else:
        expected.update(hidden_size=3584, diffusion_model_hidden_size=4096,
                        diffusion_model_z_channels=4096)
    if ablation == 'wolm':
        expected.update(pgot_lm_loss_weight=0.0, pgot_e8_reader_supervision_mode='writer',
                        pgot_e8_reader_object_weight=0.5, pgot_e8_reader_background_weight=0.25)
    errors = {k: dict(expected=v, actual=cfg.get(k))
              for k, v in expected.items() if cfg.get(k) != v}
    for k in ('pgot_one_shot_direct_rae_query','pgot_one_shot_rae_query_adapter_enable',
              'pgot_one_shot_memory_contrastive_enable'):
        if cfg.get(k, False):
            errors[k] = dict(expected=False, actual=cfg[k])
    if cfg.get('pgot_one_shot_memory_value_source', 'siglip') != 'siglip':
        errors['value_source'] = cfg.get('pgot_one_shot_memory_value_source')
    if errors:
        label = '1.5B' if model_size == '1.5b' else '7B'
        raise ValueError(f'Not the original {label} {ablation} loss experiment: {errors}')
    label = '1.5B' if model_size == '1.5b' else '7B'
    print(f'PASS checkpoint config: {ablation}; Writer GT=1; '
          f'LM={expected["pgot_lm_loss_weight"]}; '
          f'Reader mode={expected["pgot_e8_reader_supervision_mode"]}, '
          f'object={expected["pgot_e8_reader_object_weight"]}, '
          f'background={expected["pgot_e8_reader_background_weight"]}; '
          f'original {label} O8/R32. {folder}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--ablation', choices=('woreader', 'wolm'), default='woreader')
    p.add_argument('--model_size', choices=('1.5b', '7b'), default='1.5b')
    args = p.parse_args()
    verify(args.model_path, args.ablation, args.model_size)


if __name__ == '__main__':
    main()
