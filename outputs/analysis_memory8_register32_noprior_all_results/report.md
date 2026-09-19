# M8/R32 no-prior 전체 평가 결과

기준 checkpoint: `checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
Object당 visual memory 8개, background semantic register 4개, register당 visual memory 32개(배경 memory 총 128개).
이 보고서는 기존 결과 파일을 모아 정리했으며 평가를 새로 실행하지 않았다.
TF/AR은 reconstruction 평가이며 compositional generation은 별도의 slot mixing 평가다.
Compositional generation은 TF caption, mixing pool 4, random register, 10 diffusion steps, seed 1234, metric resolution 224×224 설정이다. CODA와 세부 mixing rule의 동일성은 보장되지 않는다.
8개 샘플 editing/probe 점수는 변화량 및 재caption 진단이며, 편집 성공률이나 대규모 성능 추정으로 해석하면 안 된다.

## eval_pgot_oneshot_memory8_register32_noprior/tf

원본: [outputs/eval_pgot_oneshot_memory8_register32_noprior/tf/summary.json](/home/jovyan/PGOT/outputs/eval_pgot_oneshot_memory8_register32_noprior/tf/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| protocol_version | 1.5 |
| ckpt | /home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| checkpoint_lora_detected | True |
| readout | ovt_owner |
| num_samples | 4720 |
| fARI | 0.5951635906343059 |
| mBO | 0.542687402238543 |
| mIoU | 0.5870809325691849 |
| active_count_mean | 5.152966101694915 |
| dataset | pix2cap |
| class_metrics_enabled | True |
| mBO_c | 0.5577962612430585 |
| mIoU_c | 0.6443524830895713 |
| mBO_c_num_samples | 4708 |
| mIoU_c_num_samples | 4720 |
| class_gt_granularity | category |
| class_gt_source | coco_thing_category_cache |
| gt_source | coco_instance |
| gt_granularity | instance |
| mBO_variant | mBO^i |
| mIoU_variant | mIoU^i |
| mBO_i | 0.542687402238543 |
| mIoU_i | 0.5870809325691849 |
| image_preprocess_mode | coda_center_crop |
| coda_crop_size | 512 |
| metric_resolution | 512 |
| image_patch_grid | 32 |
| image_patch_tokens | 1024 |
| caption_mode | teacher_forced |
| teacher_forced_caption | True |
| llm_attention_readout_requested | auto |
| register_hard_gt_mask_training | False |
| register_eval_route | unrestricted |
| register_hard_mask_used_in_eval | False |
| register_hard_gt_mask_used_in_eval | False |
| segmentation_forward | unrestricted_discovery |
| reconstruction_forward | unrestricted |
| recon_source | diffusion |
| latent_splice | none |
| e6_coda_direct_bottleneck | False |
| e7_causal_ownership_bottleneck | False |
| e8_visual_memory_bottleneck | True |
| one_shot_reader_enabled | True |
| one_shot_readout_mode | memory_content |
| one_shot_detach_owner_routing | True |
| one_shot_direct_rae_query | False |
| mllm_rae_query_tokens | 256 |
| e8_update_mode | one_shot_memory |
| e10_raw_value_enabled | True |
| e11_dual_m4_enabled | False |
| e12_centroid_reader_enabled | False |
| e11_memories_per_owner | 32 |
| e11_object_memories_per_owner | 8 |
| e11_register_memories_per_owner | 32 |
| e11_query_separation_enabled | False |
| e8_reader_num_layers | 1 |
| object_semantic_owners_per_object | 1 |
| background_semantic_registers | 4 |
| background_visual_memories | 128 |
| visual_memory_value_source | one final semantic+memory-ID patch attention over frozen raw SigLIP values |
| one_shot_memory_enabled | True |
| memory_reader_key_mode | content |
| memory_writer_softmax_axis | patch |
| memory_writer_owner_prior | False |
| memory_reader_owner_routing | soft |
| decoder_condition | semantic owner routing then attention over stored visual memories; no raw-patch Reader access |
| rae_standard_slot_attention_blocked | True |
| e8_clean_refinement | False |
| e8_memory_injection_enabled | False |
| e8_reader_object_weight | 0.5 |
| e8_reader_background_weight | 0.25 |
| e8_causal_training_enabled | False |
| dit_direct_memory_enabled | False |
| dit_soft_routing_enabled | False |
| dit_soft_routing_scale | 1.0 |
| null_bg_tokens | 0 |
| background_owner | aggregate probability over registers |
| dit_ovt_cross_attn_disabled | False |
| coda_overlap_excluded | True |
| owner_source | final_semantic_owner_readout |
| ovt_layers | final |
| owner_temperature | 1.0 |
| owner_weight | 1.0 |
| reader_supervision_mode | writer |
| reader_object_weight | 0.5 |
| reader_background_weight | 0.25 |
| void_tokens | 0 |
| rFID | 13.556286811828613 |
| KID_mean | 0.0012448430061340332 |
| KID_std | 0.0001601700350875035 |
| KID_x1000 | 1.2448430061340332 |
| KID_std_x1000 | 0.1601700350875035 |
| KID_subsets | 100 |
| KID_subset_size | 1000 |
| recon_psnr | 14.048700949903262 |
| recon_ssim | 0.3419444447549028 |
| recon_mse | 0.04376463114932156 |
| recon_mae | 0.15173745351662007 |

객체 수별 segmentation (아래 수치는 %):

| 객체 수 | 샘플 수 | fARI ↑ | mBOᶦ ↑ | mIoUᶦ ↑ |
| --- | --- | --- | --- | --- |
| 1-3 | 2375 | 49.3650 | 66.7504 | 72.6758 |
| 4-6 | 1044 | 71.8236 | 49.8984 | 54.0640 |
| 7-10 | 711 | 69.8098 | 38.4834 | 41.1619 |
| 11+ | 590 | 66.1978 | 31.0347 | 31.8443 |

## eval_pgot_oneshot_memory8_register32_noprior/ar

원본: [outputs/eval_pgot_oneshot_memory8_register32_noprior/ar/summary.json](/home/jovyan/PGOT/outputs/eval_pgot_oneshot_memory8_register32_noprior/ar/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| protocol_version | 1.5 |
| ckpt | /home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| checkpoint_lora_detected | True |
| readout | ovt_owner |
| num_samples | 4720 |
| fARI | 0.5886754923315094 |
| mBO | 0.5353347126963741 |
| mIoU | 0.5801781340289103 |
| active_count_mean | 5.578813559322034 |
| dataset | pix2cap |
| class_metrics_enabled | True |
| mBO_c | 0.5498759192400271 |
| mIoU_c | 0.6374646803001099 |
| mBO_c_num_samples | 4708 |
| mIoU_c_num_samples | 4720 |
| class_gt_granularity | category |
| class_gt_source | coco_thing_category_cache |
| gt_source | coco_instance |
| gt_granularity | instance |
| mBO_variant | mBO^i |
| mIoU_variant | mIoU^i |
| mBO_i | 0.5353347126963741 |
| mIoU_i | 0.5801781340289103 |
| image_preprocess_mode | coda_center_crop |
| coda_crop_size | 512 |
| metric_resolution | 512 |
| image_patch_grid | 32 |
| image_patch_tokens | 1024 |
| caption_mode | autoregressive |
| teacher_forced_caption | False |
| ar_max_new_tokens | 512 |
| ar_schema_seed_token | <thing> |
| ar_schema_seed_uses_gt | False |
| ar_scene_end_rate | 0.9209745762711864 |
| ar_format_valid_rate | 0.9209745762711864 |
| ar_generated_object_count_mean | 5.578813559322034 |
| ar_gt_object_count_mean | 5.152966101694915 |
| ar_object_count_mae | 1.4508474576271186 |
| llm_attention_readout_requested | auto |
| register_hard_gt_mask_training | False |
| register_eval_route | unrestricted |
| register_hard_mask_used_in_eval | False |
| register_hard_gt_mask_used_in_eval | False |
| segmentation_forward | unrestricted_discovery |
| reconstruction_forward | unrestricted |
| recon_source | diffusion |
| latent_splice | none |
| e6_coda_direct_bottleneck | False |
| e7_causal_ownership_bottleneck | False |
| e8_visual_memory_bottleneck | True |
| one_shot_reader_enabled | True |
| one_shot_readout_mode | memory_content |
| one_shot_detach_owner_routing | True |
| one_shot_direct_rae_query | False |
| mllm_rae_query_tokens | 256 |
| e8_update_mode | one_shot_memory |
| e10_raw_value_enabled | True |
| e11_dual_m4_enabled | False |
| e12_centroid_reader_enabled | False |
| e11_memories_per_owner | 32 |
| e11_object_memories_per_owner | 8 |
| e11_register_memories_per_owner | 32 |
| e11_query_separation_enabled | False |
| e8_reader_num_layers | 1 |
| object_semantic_owners_per_object | 1 |
| background_semantic_registers | 4 |
| background_visual_memories | 128 |
| visual_memory_value_source | one final semantic+memory-ID patch attention over frozen raw SigLIP values |
| one_shot_memory_enabled | True |
| memory_reader_key_mode | content |
| memory_writer_softmax_axis | patch |
| memory_writer_owner_prior | False |
| memory_reader_owner_routing | soft |
| decoder_condition | semantic owner routing then attention over stored visual memories; no raw-patch Reader access |
| rae_standard_slot_attention_blocked | True |
| e8_clean_refinement | False |
| e8_memory_injection_enabled | False |
| e8_reader_object_weight | 0.5 |
| e8_reader_background_weight | 0.25 |
| e8_causal_training_enabled | False |
| dit_direct_memory_enabled | False |
| dit_soft_routing_enabled | False |
| dit_soft_routing_scale | 1.0 |
| null_bg_tokens | 0 |
| background_owner | aggregate probability over registers |
| dit_ovt_cross_attn_disabled | False |
| coda_overlap_excluded | True |
| owner_source | final_semantic_owner_readout |
| ovt_layers | final |
| owner_temperature | 1.0 |
| owner_weight | 1.0 |
| reader_supervision_mode | writer |
| reader_object_weight | 0.5 |
| reader_background_weight | 0.25 |
| void_tokens | 0 |
| rFID | 13.428787231445312 |
| KID_mean | 0.0012073444668203592 |
| KID_std | 0.0001343636540696025 |
| KID_x1000 | 1.2073444668203592 |
| KID_std_x1000 | 0.1343636540696025 |
| KID_subsets | 100 |
| KID_subset_size | 1000 |
| recon_psnr | 14.047912295187935 |
| recon_ssim | 0.34229080723677524 |
| recon_mse | 0.04373888683153151 |
| recon_mae | 0.15159717057081792 |

객체 수별 segmentation (아래 수치는 %):

| 객체 수 | 샘플 수 | fARI ↑ | mBOᶦ ↑ | mIoUᶦ ↑ |
| --- | --- | --- | --- | --- |
| 1-3 | 2375 | 48.8882 | 66.6042 | 72.5831 |
| 4-6 | 1044 | 71.4846 | 49.0483 | 53.3413 |
| 7-10 | 711 | 69.3986 | 37.3615 | 39.9941 |
| 11+ | 590 | 64.0223 | 28.6092 | 29.3814 |

## compgen_pgot_oneshot_memory8_register32_noprior

원본: [outputs/compgen_pgot_oneshot_memory8_register32_noprior/summary.json](/home/jovyan/PGOT/outputs/compgen_pgot_oneshot_memory8_register32_noprior/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| task | compositional_generation |
| protocol | CODA Table 4 (arXiv:2601.01224): 'configurations are created by randomly mixing slots within a batch'; FID and KID x1e3 of composed images vs real images. CODA publishes no mixing script: register handling, object count and sampling rule below are PGOT choices. |
| mixing_rule | each target keeps its own object count; every object slot (semantic slot + its visual memories) is drawn without replacement from the other images' valid objects in the same batch; registers follow register_source |
| model_path | checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| one_shot_readout_mode | memory_content |
| object_memories_per_owner | 8 |
| register_memories_per_owner | 32 |
| mix_mode | random_slots |
| register_source | random |
| batch_size_mixing_pool | 4 |
| seed | 1234 |
| caption_mode | teacher_forced |
| diffusion_inference_steps | 10 |
| guidance_scale | 1.0 |
| real_reference | the same val images (CODA-512 center crop), one real per generated image |
| n_real | 4720 |
| n_generated | 4720 |
| FID | 21.039810180664062 |
| KID_x1e3_mean | 5.575435236096382 |
| KID_x1e3_std | 0.7513578748330474 |
| kid_subsets | 100 |
| kid_subset_size | 1000 |
| raw_rae_query_batch_std | 0.0 |
| peak_gpu_memory_GiB | 22.362115383148193 |
| elapsed_sec | 8114.82998251915 |
| compositions_file | outputs/compgen_pgot_oneshot_memory8_register32_noprior/compositions.jsonl |

### identity_check

| 항목 | 값 |
| --- | --- |
| passed | True |
| max_relative_error | 0.0 |
| tolerance | 0.001 |

### mix_stats

| 항목 | 값 |
| --- | --- |
| mean_objects_per_image | 5.152966101694915 |
| foreign_object_slot_fraction | 1.0 |
| images_sampled_with_replacement | 437 |
| images_left_unmixed | 0 |
| mixed_condition_mean_abs_delta | 0.8352666567947905 |

## segeval_pgot_oneshot_memory8_register32_noprior/external_voc_ar

원본: [outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_voc_ar/summary.json](/home/jovyan/PGOT/outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_voc_ar/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| protocol_version | 1.5 |
| ckpt | checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| checkpoint_lora_detected | True |
| readout | ovt_owner |
| num_samples | 1449 |
| fARI | 0.4402920189937942 |
| mBO | 0.7927968849448204 |
| mIoU | 0.79786278252934 |
| active_count_mean | 3.432022084195997 |
| dataset | voc |
| sMBO | 0.7674612972191498 |
| sMIOU | 0.7960619790090043 |
| class_metrics_enabled | False |
| gt_source | dataset_mask |
| gt_granularity | instance |
| mBO_variant | mBO^i |
| mIoU_variant | mIoU^i |
| mBO_i | 0.7927968849448204 |
| mIoU_i | 0.79786278252934 |
| image_preprocess_mode | default |
| coda_crop_size | 512 |
| metric_resolution | 512 |
| image_patch_grid | 32 |
| image_patch_tokens | 1024 |
| caption_mode | autoregressive |
| teacher_forced_caption | False |
| ar_max_new_tokens | 512 |
| ar_schema_seed_token | <thing> |
| ar_schema_seed_uses_gt | False |
| ar_scene_end_rate | 0.9779158040027606 |
| ar_format_valid_rate | 0.9779158040027606 |
| ar_generated_object_count_mean | 3.432022084195997 |
| ar_gt_object_count_mean | 2.216011042097999 |
| ar_object_count_mae | 1.3333333333333333 |
| llm_attention_readout_requested | auto |
| register_hard_gt_mask_training | False |
| register_eval_route | unrestricted |
| register_hard_mask_used_in_eval | False |
| register_hard_gt_mask_used_in_eval | False |
| segmentation_forward | unrestricted_discovery |
| reconstruction_forward | unrestricted |
| recon_source | diffusion |
| latent_splice | none |
| e6_coda_direct_bottleneck | False |
| e7_causal_ownership_bottleneck | False |
| e8_visual_memory_bottleneck | True |
| one_shot_reader_enabled | True |
| one_shot_readout_mode | memory_content |
| one_shot_detach_owner_routing | True |
| one_shot_direct_rae_query | False |
| mllm_rae_query_tokens | 256 |
| e8_update_mode | one_shot_memory |
| e10_raw_value_enabled | True |
| e11_dual_m4_enabled | False |
| e12_centroid_reader_enabled | False |
| e11_memories_per_owner | 32 |
| e11_object_memories_per_owner | 8 |
| e11_register_memories_per_owner | 32 |
| e11_query_separation_enabled | False |
| e8_reader_num_layers | 1 |
| object_semantic_owners_per_object | 1 |
| background_semantic_registers | 4 |
| background_visual_memories | 128 |
| visual_memory_value_source | one final semantic+memory-ID patch attention over frozen raw SigLIP values |
| one_shot_memory_enabled | True |
| memory_reader_key_mode | content |
| memory_writer_softmax_axis | patch |
| memory_writer_owner_prior | False |
| memory_reader_owner_routing | soft |
| decoder_condition | semantic owner routing then attention over stored visual memories; no raw-patch Reader access |
| rae_standard_slot_attention_blocked | True |
| e8_clean_refinement | False |
| e8_memory_injection_enabled | False |
| e8_reader_object_weight | 0.5 |
| e8_reader_background_weight | 0.25 |
| e8_causal_training_enabled | False |
| dit_direct_memory_enabled | False |
| dit_soft_routing_enabled | False |
| dit_soft_routing_scale | 1.0 |
| null_bg_tokens | 0 |
| background_owner | aggregate probability over registers |
| dit_ovt_cross_attn_disabled | False |
| coda_overlap_excluded | False |
| owner_source | final_semantic_owner_readout |
| ovt_layers | final |
| owner_temperature | 1.0 |
| owner_weight | 1.0 |
| reader_supervision_mode | writer |
| reader_object_weight | 0.5 |
| reader_background_weight | 0.25 |
| void_tokens | 0 |

객체 수별 segmentation (아래 수치는 %):

| 객체 수 | 샘플 수 | fARI ↑ | mBOᶦ ↑ | mIoUᶦ ↑ |
| --- | --- | --- | --- | --- |
| 1-3 | 1209 | 37.1081 | 82.9851 | 83.3766 |
| 4-6 | 180 | 79.8208 | 63.5347 | 64.8777 |
| 7-10 | 54 | 76.7160 | 53.8143 | 54.2993 |
| 11+ | 6 | 70.6991 | 34.1669 | 32.9844 |

## segeval_pgot_oneshot_memory8_register32_noprior/external_movi-c_ar

원본: [outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_movi-c_ar/summary.json](/home/jovyan/PGOT/outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_movi-c_ar/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| protocol_version | 1.5 |
| ckpt | checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| checkpoint_lora_detected | True |
| readout | ovt_owner |
| num_samples | 6000 |
| fARI | 0.553202086872637 |
| mBO | 0.23834341315132446 |
| mIoU | 0.31650555165298283 |
| active_count_mean | 1.6785 |
| dataset | movi-c |
| class_metrics_enabled | False |
| gt_source | dataset_mask |
| gt_granularity | instance |
| mBO_variant | mBO^i |
| mIoU_variant | mIoU^i |
| mBO_i | 0.23834341315132446 |
| mIoU_i | 0.31650555165298283 |
| image_preprocess_mode | default |
| coda_crop_size | 512 |
| metric_resolution | 512 |
| image_patch_grid | 32 |
| image_patch_tokens | 1024 |
| caption_mode | autoregressive |
| teacher_forced_caption | False |
| ar_max_new_tokens | 512 |
| ar_schema_seed_token | <thing> |
| ar_schema_seed_uses_gt | False |
| ar_scene_end_rate | 0.9996666666666667 |
| ar_format_valid_rate | 0.9996666666666667 |
| ar_generated_object_count_mean | 1.6785 |
| ar_gt_object_count_mean | 5.1435 |
| ar_object_count_mae | 3.5343333333333335 |
| llm_attention_readout_requested | auto |
| register_hard_gt_mask_training | False |
| register_eval_route | unrestricted |
| register_hard_mask_used_in_eval | False |
| register_hard_gt_mask_used_in_eval | False |
| segmentation_forward | unrestricted_discovery |
| reconstruction_forward | unrestricted |
| recon_source | diffusion |
| latent_splice | none |
| e6_coda_direct_bottleneck | False |
| e7_causal_ownership_bottleneck | False |
| e8_visual_memory_bottleneck | True |
| one_shot_reader_enabled | True |
| one_shot_readout_mode | memory_content |
| one_shot_detach_owner_routing | True |
| one_shot_direct_rae_query | False |
| mllm_rae_query_tokens | 256 |
| e8_update_mode | one_shot_memory |
| e10_raw_value_enabled | True |
| e11_dual_m4_enabled | False |
| e12_centroid_reader_enabled | False |
| e11_memories_per_owner | 32 |
| e11_object_memories_per_owner | 8 |
| e11_register_memories_per_owner | 32 |
| e11_query_separation_enabled | False |
| e8_reader_num_layers | 1 |
| object_semantic_owners_per_object | 1 |
| background_semantic_registers | 4 |
| background_visual_memories | 128 |
| visual_memory_value_source | one final semantic+memory-ID patch attention over frozen raw SigLIP values |
| one_shot_memory_enabled | True |
| memory_reader_key_mode | content |
| memory_writer_softmax_axis | patch |
| memory_writer_owner_prior | False |
| memory_reader_owner_routing | soft |
| decoder_condition | semantic owner routing then attention over stored visual memories; no raw-patch Reader access |
| rae_standard_slot_attention_blocked | True |
| e8_clean_refinement | False |
| e8_memory_injection_enabled | False |
| e8_reader_object_weight | 0.5 |
| e8_reader_background_weight | 0.25 |
| e8_causal_training_enabled | False |
| dit_direct_memory_enabled | False |
| dit_soft_routing_enabled | False |
| dit_soft_routing_scale | 1.0 |
| null_bg_tokens | 0 |
| background_owner | aggregate probability over registers |
| dit_ovt_cross_attn_disabled | False |
| coda_overlap_excluded | False |
| owner_source | final_semantic_owner_readout |
| ovt_layers | final |
| owner_temperature | 1.0 |
| owner_weight | 1.0 |
| reader_supervision_mode | writer |
| reader_object_weight | 0.5 |
| reader_background_weight | 0.25 |
| void_tokens | 0 |

객체 수별 segmentation (아래 수치는 %):

| 객체 수 | 샘플 수 | fARI ↑ | mBOᶦ ↑ | mIoUᶦ ↑ |
| --- | --- | --- | --- | --- |
| 1-3 | 1666 | 65.3012 | 32.1014 | 44.9214 |
| 4-6 | 2681 | 57.0501 | 22.8845 | 30.1071 |
| 7-10 | 1653 | 42.4550 | 17.1329 | 20.7786 |

## segeval_pgot_oneshot_memory8_register32_noprior/external_movi-e_ar

원본: [outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_movi-e_ar/summary.json](/home/jovyan/PGOT/outputs/segeval_pgot_oneshot_memory8_register32_noprior/external_movi-e_ar/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| protocol_version | 1.5 |
| ckpt | checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| checkpoint_lora_detected | True |
| readout | ovt_owner |
| num_samples | 6000 |
| fARI | 0.3335162629741316 |
| mBO | 0.1339181456111061 |
| mIoU | 0.16084227184578775 |
| active_count_mean | 2.4436666666666667 |
| dataset | movi-e |
| class_metrics_enabled | False |
| gt_source | dataset_mask |
| gt_granularity | instance |
| mBO_variant | mBO^i |
| mIoU_variant | mIoU^i |
| mBO_i | 0.1339181456111061 |
| mIoU_i | 0.16084227184578775 |
| image_preprocess_mode | default |
| coda_crop_size | 512 |
| metric_resolution | 512 |
| image_patch_grid | 32 |
| image_patch_tokens | 1024 |
| caption_mode | autoregressive |
| teacher_forced_caption | False |
| ar_max_new_tokens | 512 |
| ar_schema_seed_token | <thing> |
| ar_schema_seed_uses_gt | False |
| ar_scene_end_rate | 0.9983333333333333 |
| ar_format_valid_rate | 0.9981666666666666 |
| ar_generated_object_count_mean | 2.4436666666666667 |
| ar_gt_object_count_mean | 12.258166666666666 |
| ar_object_count_mae | 9.815166666666666 |
| llm_attention_readout_requested | auto |
| register_hard_gt_mask_training | False |
| register_eval_route | unrestricted |
| register_hard_mask_used_in_eval | False |
| register_hard_gt_mask_used_in_eval | False |
| segmentation_forward | unrestricted_discovery |
| reconstruction_forward | unrestricted |
| recon_source | diffusion |
| latent_splice | none |
| e6_coda_direct_bottleneck | False |
| e7_causal_ownership_bottleneck | False |
| e8_visual_memory_bottleneck | True |
| one_shot_reader_enabled | True |
| one_shot_readout_mode | memory_content |
| one_shot_detach_owner_routing | True |
| one_shot_direct_rae_query | False |
| mllm_rae_query_tokens | 256 |
| e8_update_mode | one_shot_memory |
| e10_raw_value_enabled | True |
| e11_dual_m4_enabled | False |
| e12_centroid_reader_enabled | False |
| e11_memories_per_owner | 32 |
| e11_object_memories_per_owner | 8 |
| e11_register_memories_per_owner | 32 |
| e11_query_separation_enabled | False |
| e8_reader_num_layers | 1 |
| object_semantic_owners_per_object | 1 |
| background_semantic_registers | 4 |
| background_visual_memories | 128 |
| visual_memory_value_source | one final semantic+memory-ID patch attention over frozen raw SigLIP values |
| one_shot_memory_enabled | True |
| memory_reader_key_mode | content |
| memory_writer_softmax_axis | patch |
| memory_writer_owner_prior | False |
| memory_reader_owner_routing | soft |
| decoder_condition | semantic owner routing then attention over stored visual memories; no raw-patch Reader access |
| rae_standard_slot_attention_blocked | True |
| e8_clean_refinement | False |
| e8_memory_injection_enabled | False |
| e8_reader_object_weight | 0.5 |
| e8_reader_background_weight | 0.25 |
| e8_causal_training_enabled | False |
| dit_direct_memory_enabled | False |
| dit_soft_routing_enabled | False |
| dit_soft_routing_scale | 1.0 |
| null_bg_tokens | 0 |
| background_owner | aggregate probability over registers |
| dit_ovt_cross_attn_disabled | False |
| coda_overlap_excluded | False |
| owner_source | final_semantic_owner_readout |
| ovt_layers | final |
| owner_temperature | 1.0 |
| owner_weight | 1.0 |
| reader_supervision_mode | writer |
| reader_object_weight | 0.5 |
| reader_background_weight | 0.25 |
| void_tokens | 0 |

객체 수별 segmentation (아래 수치는 %):

| 객체 수 | 샘플 수 | fARI ↑ | mBOᶦ ↑ | mIoUᶦ ↑ |
| --- | --- | --- | --- | --- |
| 1-3 | 3 | 7.4580 | 16.2651 | 15.1597 |
| 4-6 | 182 | 54.5126 | 17.8603 | 23.3484 |
| 7-10 | 1487 | 41.2522 | 15.0843 | 19.2890 |
| 11+ | 4328 | 29.7653 | 12.6204 | 14.6783 |

## probe_editing_memory8_register32_noprior

원본: [outputs/probe_editing_memory8_register32_noprior/summary.json](/home/jovyan/PGOT/outputs/probe_editing_memory8_register32_noprior/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| model_path | /home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| num_images | 8 |
| object_memories_per_owner | 8 |
| register_memories_per_owner | 32 |
| identity_check_max_relative_error | 7.07166645952384e-06 |
| diffusion_inference_steps | 10 |
| seed | 1234 |
| elapsed_sec | 173.5 |

### aggregate

| 조건 | inside_mean_abs | outside_mean_abs | global_mean_abs | energy_inside_fraction | mask_area_fraction |
| --- | --- | --- | --- | --- | --- |
| recon_seed2 | 0.1359 | 0.1162 | 0.1192 | 0.2539 | 0.1863 |
| remove | 0.1248 | 0.0342 | 0.0467 | 0.5298 | 0.1863 |
| replace | 0.1818 | 0.0879 | 0.1028 | 0.3544 | 0.1863 |
| text_color | 0.0197 | 0.0164 | 0.0162 | 0.3339 | 0.1863 |
| text_category | 0.0701 | 0.0419 | 0.0453 | 0.3088 | 0.1863 |
| insert_text | 0.0202 | 0.0196 | 0.0192 | 0.1667 | 0.1863 |

개별 샘플 기록은 위 원본 JSON의 records 항목 참조.

## probe_removal_instruction_memory8_register32_noprior

원본: [outputs/probe_removal_instruction_memory8_register32_noprior/summary.json](/home/jovyan/PGOT/outputs/probe_removal_instruction_memory8_register32_noprior/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| model_path | /home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| num_images | 8 |
| writer_replication_max_relative_error | 0.0 |
| elapsed_sec | 510.6 |

### removal_aggregate

| 조건 | siglip_cos_to_recon_inside | siglip_cos_to_recon_outside | recaption_mentions_object |
| --- | --- | --- | --- |
| remove_set | 0.5931 | 0.8087 | 0.875 |
| remove_caption | 0.674 | 0.7874 | 0.875 |
| remove_masked_pred | 0.3932 | 0.7058 | 0.5 |
| remove_masked_pred_dil | 0.3255 | 0.6485 | 0.5 |
| remove_masked_gt | 0.4234 | 0.7381 | 0.75 |
| remove_prior | 0.4631 | 0.7025 | 0.75 |
| recon_prior | 0.8163 | 0.7838 | 0.875 |
| recon | None | None | 0.875 |

### instruction_aggregate

| 조건 | mentions_object | object_count | format_valid | siglip_cos_to_default_render_inside | siglip_cos_to_default_render_outside |
| --- | --- | --- | --- | --- | --- |
| default | 0.875 | 6.375 | 0.875 | None | None |
| instr_do_not_include | 1.0 | 5.375 | 0.875 | 0.7983 | 0.7998 |
| instr_except | 1.0 | 5.875 | 0.875 | 0.8207 | 0.8144 |

개별 샘플 기록은 위 원본 JSON의 records 항목 참조.

## probe_two_image_composition_memory8_register32_noprior

원본: [outputs/probe_two_image_composition_memory8_register32_noprior/summary.json](/home/jovyan/PGOT/outputs/probe_two_image_composition_memory8_register32_noprior/summary.json)

| 항목 | 원본 값 |
| --- | --- |
| model_path | /home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000 |
| num_pairs | 8 |
| elapsed_sec | 169.7 |

### aggregate

| 조건 | donor_mask_area_fraction | donor_location_iou | energy_in_donor_mask | energy_in_A_objects | A_objects_area_fraction |
| --- | --- | --- | --- | --- | --- |
| add | 0.181 | 0.2392 | 0.4283 | 0.3922 | 0.293 |
| object_on_A_bg | 0.181 | 0.2365 | 0.39 | 0.4875 | 0.293 |

개별 샘플 기록은 위 원본 JSON의 records 항목 참조.

## 별도 LLM controller 편집 계획 평가

원본: [render_log.json](/home/jovyan/PGOT/outputs/controller_edit_memory8_register32_noprior/render_masked_pred_dil/render_log.json)

`correct`는 background와 remove/add owner ID 계획이 expected plan과 일치하는지 평가한 값이다. PGOT 렌더링 이미지의 편집 성공률은 아니다.

| 작업 | 정확한 계획 | 전체 | 계획 정확도 |
| --- | --- | --- | --- |
| add | 6 | 6 | 100.00% |
| remove | 6 | 6 | 100.00% |
| replace | 6 | 6 | 100.00% |

