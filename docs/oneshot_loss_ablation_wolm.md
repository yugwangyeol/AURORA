# Original 1.5B: w/o LM loss

An independent ablation of the original Object8 / Register32 configuration,
not a continuation of the w/o Reader-loss model. Training uses the same E11
checkpoint initializer as the baseline and a separate output directory.

Only the LM objective weight changes from 1.0 to 0.0. Writer GT grounding stays
at 1.0, Reader–Writer object/background supervision stays at 0.5/0.25, and
reconstruction stays at 1.5. MLLM RAE queries, SigLIP values and the no-owner-prior
memory route stay enabled as in the baseline. No direct RAE query, adapter,
contrastive objective or DINO memory is used.

Training still supplies caption tokens under teacher forcing: removing the LM
loss does not remove text inputs or freeze the MLLM. Other objectives can still
update trainable MLLM parameters. AR evaluation generates descriptions without
GT captions, so weaker description generation is a possible outcome of this
ablation, not grounds for changing its evaluation protocol.

```bash
cd /home/jovyan/PGOT
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_pgot_oneshot_memory8_register32_noprior_wolm.sh
```

Runs 5000 training steps on GPUs 0,1, then AR evaluation on GPU 1. TF evaluation
is excluded. Outputs use `checkpoints/pgot_oneshot_memory8_register32_noprior_wolm`
and `outputs/eval_pgot_oneshot_memory8_register32_noprior_wolm/ar`. Existing
training outputs are refused; errors halt subsequent stages. The trained
checkpoint configuration is audited before evaluation loads a model.

For a completed checkpoint, run AR without retraining:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_pgot_oneshot_memory8_register32_noprior_wolm.sh
```
