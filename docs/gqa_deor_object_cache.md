# GQA DeOR object-representation cache

This pipeline prepares the balanced GQA train/val/testdev split and extracts
frozen DeOR representations once per unique image. It is the first three stages
of the DeOR-to-7B understanding experiment.

Each valid generated object stores exactly:

- one 1536-D semantic token;
- eight 1536-D visual-memory tokens;
- the generated category/description and thing/stuff flag;
- patch ownership probabilities and a compact 32x32 ownership label map.

The four background semantic registers (each has 32 visual memories) and the 24
padded memory positions per object are excluded. Floating tensors are stored in
FP16; the frozen extractor itself runs in FP32. Questions and answers live in a
separate JSONL and are never read by the extractor.

```bash
cd /home/jovyan/PGOT
bash scripts/prepare_gqa_deor_cache.sh

# One resumable shard. Use different GPUs and SHARD_INDEX values for parallelism.
GPU=0 SPLIT=train NUM_SHARDS=4 SHARD_INDEX=0 \
  bash scripts/cache_gqa_deor_objects.sh

# Resume all four train shards, then cache and verify val/testdev as well.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_gqa_deor_cache_4gpu.sh
```

Prepared manifests are under `/home/jovyan/data/gqa/prepared_balanced/manifests`.
Feature bundles are under `/home/jovyan/data/gqa/deor_object_cache_m8_r32_bs8`.
The default is FP32 extraction with batch size 8 per GPU. Batch size is part of
cache provenance, so caches produced with different batching are never mixed.
