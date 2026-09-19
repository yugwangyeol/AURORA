"""Fixed manual donor transfers through the native PGOT Reader."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from pgot.eval.ar_binding_router import normalize_category, parse_ar_slots
from pgot.eval.class_mask_pairs import ar_pairs
from pgot.eval.figure_model_editing import edit_stats, panel_grid, tensor_image
from pgot.eval.natural_object_selection import crop, digest
from pgot.eval.object_representation_transfer import CONDITIONS, replace_description, transfer_objects
from pgot.eval.run_natural_object_selection import checkpoint_signature, file_digest, jsonl, write_json

VERSION = "native-object-representation-transfer-v1"


def config(args):
    return dict(version=VERSION, checkpoint=str(Path(args.model_path).resolve()),
                checkpoint_signature=checkpoint_signature(args.model_path),
                pairs_sha256=file_digest(args.pairs), annotations_sha256=file_digest(args.annotations),
                captions_sha256=file_digest(args.captions), caption_summary_sha256=file_digest(args.caption_summary),
                dtype="fp32", seed=args.seed, steps=args.diffusion_inference_steps,
                guidance_scale=args.guidance_scale, eval_size=512, max_objects=50,
                max_caption_tokens=args.max_caption_tokens, conditions=list(CONDITIONS),
                gt_inference_input=False, external_selector=False, pixel_paste=False,
                reader_prior=False, writer_rewrite=False, spatial_warp=False,
                selection="Fixed manual curated pairs; several sources previously GT-assisted shortlisted; not benchmark",
                placement="Native learned representations; no guarantee of target-location preservation")


def prepare(args):
    root = Path(args.output_dir)
    conf = config(args)
    fingerprint = digest(conf)
    path = root/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Existing experiment differs; choose a fresh output directory")
        print("Identical transfer manifest reused.", flush=True)
        return
    summary = json.loads(Path(args.caption_summary).read_text())
    if Path(summary["ckpt"]).resolve() != Path(args.model_path).resolve() or summary.get("caption_mode") != "autoregressive":
        raise ValueError("Require image-only AR captions from the selected checkpoint")
    pairs = jsonl(args.pairs)
    if not pairs or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Empty or duplicate pair IDs")
    captions = {int(r["image_id"]): r for r in jsonl(args.captions)}
    # Only paths are read from this annotation manifest, never masks/GT captions.
    images = {int(r["image_id"]): r["image_path"] for r in jsonl(args.annotations)}
    entries = {}
    for pair in pairs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pair["pair_id"]):
            raise ValueError("Unsafe pair ID")
        if pair["target_image_id"] == pair["donor_image_id"]:
            raise ValueError("Donor must be another image")
        for role in ("target", "donor"):
            iid = int(pair[f"{role}_image_id"])
            record = captions[iid]
            text, count = record["text"], int(record["ovt_count"])
            slots = parse_ar_slots(text, count)
            index = int(pair[f"{role}_slot"])
            if not 0 <= index < count or not slots[index]["parsed"] or not slots[index]["is_thing"]:
                raise ValueError(f"Invalid {role} owner for {pair['pair_id']}")
            expected_category=pair.get(f"{role}_category",pair.get("category"))
            if expected_category is None or slots[index]["normalized_category"] != normalize_category(expected_category):
                raise ValueError("Manual pair category does not agree with GENERATED category")
            if not Path(images[iid]).is_file():
                raise ValueError("Missing source image")
            entries[str(iid)] = dict(image_id=iid, image_path=images[iid], caption=text,
                                    caption_sha256=digest(text), ovt_count=count, slots=slots)
            pair[f"{role}_description"] = slots[index]["description"]
            pair[f"{role}_category"] = slots[index]["category"]
            # Native mask readout compacts thing labels; do not use OVT+1 blindly.
            candidate = next(p for p in ar_pairs(text,count) if p["slot_index"] == index)
            pair[f"{role}_prediction_label"] = candidate["prediction_label"]
        cross_category=normalize_category(pair["target_category"])!=normalize_category(pair["donor_category"])
        pair.setdefault("category",f"{pair['target_category']} -> {pair['donor_category']}")
        pair["cross_category"]=cross_category
        pair["text_only_caption"] = replace_description(entries[str(pair["target_image_id"])]["caption"],
                pair["target_slot"], pair["donor_description"],pair["donor_category"] if cross_category else None)
    write_json(path, dict(config=conf, fingerprint=fingerprint, entries=entries, pairs=pairs))
    print(f"Prepared {len(pairs)} fixed manual transfers / {len(entries)} images.", flush=True)


def manifest(args):
    root = Path(args.output_dir)
    value = json.loads((root/"manifest.json").read_text())
    if value["config"] != config(args):
        raise ValueError("Checkpoint/source/inference configuration differs from manifest")
    return root, value


def outline(image,mask):
    from scipy.ndimage import binary_erosion
    pixels=np.asarray(image).copy()
    edge=mask & ~binary_erosion(mask,iterations=2)
    pixels[edge]=[40,230,80]
    return Image.fromarray(pixels)


def refresh_figure(folder,pair):
    a=Image.open(folder/"source_A.png").convert("RGB")
    b=Image.open(folder/"donor_B.png").convert("RGB")
    target=np.asarray(Image.open(folder/"target_prediction_mask.png"),bool)
    donor=np.asarray(Image.open(folder/"donor_prediction_mask.png"),bool)
    names=["reconstruction","text_only","visual_only","bundle_swap"]
    panels=[outline(a,target),outline(b,donor)]+[Image.open(folder/f"{c}_native.png").convert("RGB") for c in names]
    titles=["A: target","B: donor","A reconstruction","Donor text only","Visual memory only","Full owner bundle"]
    grid=panel_grid(panels,titles,folder/"figure.png",size=256)
    labelled=Image.new("RGB",(grid.width,grid.height+48),"white");labelled.paste(grid,(0,48))
    ImageDraw.Draw(labelled).text((8,8),pair["instruction"],fill="black")
    ImageDraw.Draw(labelled).text((8,27),"Manual IDs; native Reader; donor placement is not controlled.",fill="black")
    labelled.save(folder/"figure.png")


def export(root, m):
    records, figures = [], []
    for pair in m["pairs"]:
        folder = root/"pairs"/pair["pair_id"]
        if not (folder/"metadata.json").exists():
            continue
        row = json.loads((folder/"metadata.json").read_text())
        refresh_figure(folder,pair)
        records.append(row)
        figures.append(Image.open(folder/"figure.png").convert("RGB"))
    if figures:
        sheet = Image.new("RGB", (figures[0].width,sum(p.height for p in figures)),"white")
        y=0
        for picture in figures:
            sheet.paste(picture,(0,y)); y+=picture.height
        sheet.save(root/"contact_sheet.jpg", quality=94)
        for start in range(0,len(figures),4):
            page=figures[start:start+4]
            canvas=Image.new("RGB",(page[0].width,sum(p.height for p in page)),"white")
            offset=0
            for picture in page:
                canvas.paste(picture,(0,offset));offset+=picture.height
            canvas.save(root/f"contact_sheet_page_{start//4+1:02}.jpg",quality=94)
    write_json(root/"summary.json",dict(version=VERSION,fingerprint=m["fingerprint"],
        completed_pairs=len(records),planned_pairs=len(m["pairs"]),
        quantitative_edit_success_claimed=False,
        note="Pixel change is diagnostic, not donor identity/appearance success. Review all fixed pairs including failures.",
        pairs=records))
    fields=["pair_id","condition","inside_change","outside_change","change_energy_inside_fraction","region_fraction"]
    with (root/"change_diagnostics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in records:
            for condition,stats in row["change_diagnostics"].items():
                writer.writerow(dict(pair_id=row["pair_id"],condition=condition,
                                      **{k:stats[k] for k in fields[2:]}))
    lines=["# Fixed object-representation transfer pilot","",
           "Manual IDs; image-only AR captions; native Reader, no GT masks or spatial priors at inference.","",
           "Instructions label manual operations, not freeform instructions interpreted by the model.","",
           "All fixed pairs are retained. See visual_review.json for qualitative review, not benchmark accuracy.","",
           "| Pair | Category | Target / donor slots | Figure |","| --- | --- | --- | --- |"]
    for row in records:
        lines.append(f"| {row['pair_id']} | {row['category']} | {row['target_image_id']}:{row['target_slot']} / {row['donor_image_id']}:{row['donor_slot']} | [figure](pairs/{row['pair_id']}/figure.png) |")
    (root/"results.md").write_text("\n".join(lines)+"\n")


def render(args):
    import torch
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.eval.probe_object_editing import owner_units, condition_from_units
    from pgot.eval.run_compositional_gen import seeded_noise
    from pgot.eval.run_swap_editing import decode_to_image
    from pgot.eval.run_eval import load_rae_decoder
    from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
    from pgot.model.visual_memory import PGOTOneShotMemoryReader
    root,m=manifest(args)
    pairs=m["pairs"][:args.limit] if args.limit else m["pairs"]
    pending=[]
    for pair in pairs:
        path=root/"pairs"/pair["pair_id"]/"metadata.json"
        if path.exists():
            if json.loads(path.read_text())["fingerprint"] != m["fingerprint"]:
                raise ValueError("Completed pair cache provenance mismatch")
        else:
            pending.append(pair)
    if not pending:
        export(root,m);print("All requested transfers cached; no model load.",flush=True);return
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a free GPU")
    model,tokenizer,device,_=load_model_and_tokenizer(args)
    if not isinstance(model.pgot_e8_reader,PGOTOneShotMemoryReader):
        raise ValueError("Native one-shot memory Reader required")
    if (model.pgot_e8_writer.object_memories_per_owner,model.pgot_e8_writer.register_memories_per_owner)!=(8,32):
        raise ValueError("This pilot requires Object 8 / Register 32")
    checkpoint_keys=set(json.loads((Path(args.model_path)/"model.safetensors.index.json").read_text())["weight_map"])
    parameters=dict(model.named_parameters(remove_duplicate=False))
    restored={id(v) for k,v in parameters.items() if k in checkpoint_keys}
    missing=sorted(k for k in parameters if k not in checkpoint_keys and id(parameters[k]) not in restored)
    unexpected=sorted(checkpoint_keys-set(model.state_dict()))
    if missing or unexpected:
        raise ValueError(f"Incomplete model parameter restore: {missing}, unexpected={unexpected}")
    token_ids={}
    for name,token in (("ovt","<ovt>"),("scene_end","<scene_end>"),("thing","<thing>"),("stuff","<stuff>")):
        if token not in tokenizer.get_vocab():
            raise ValueError(f"Checkpoint tokenizer is missing {token}")
        token_ids[name]=tokenizer.convert_tokens_to_ids(token)
        setattr(model,f"pgot_{name}_token_id",token_ids[name])
    decoder=load_rae_decoder(model,device=device,dtype=torch.float32)
    towers=model.get_vision_tower_aux_list()
    write_json(root/"protocol.json",dict(m["config"],fingerprint=m["fingerprint"],
        special_token_ids=token_ids,missing_parameters=missing,unexpected_parameters=unexpected,
        original_other_object_inputs_unchanged=True,original_register_inputs_unchanged=True,
        source_rae_queries_unchanged=True,same_noise_for_all_conditions=True,
        object_memories_per_owner=8,register_memories_per_owner=32,
        text_only="A image + A caption with donor description (and category for cross-category pairs) substituted; take only target semantic state, retain original A visual memory and every other owner",
        semantic_only="donor semantic state, original target visual memory",
        visual_only="original target semantic state, donor visual memory",
        bundle_swap="donor semantic state and visual memory; replace exactly one owner in place",
        no_freeform_instruction_following_claim=True))
    def generate_native(hidden,noise):
        conditioning=model._captionslot_prepare_diffusion_condition(hidden).float()
        if not torch.isfinite(conditioning).all(): raise ValueError("Nonfinite DiT conditioning")
        model.set_diff_fp32()
        latent=model.diff_head.infer(z=conditioning,x_end=noise.clone(),guidance_level=args.guidance_scale)
        if latent is None or not torch.isfinite(latent).all():
            raise ValueError("Nonfinite DiT latent; never silently repair failed generation")
        return decode_to_image(decoder,latent,device)[0]
    keys=["semantic_slots","visual_memory","ovt_object_valid","raw_rae_hidden","rae_hidden"]
    def forward(entry,text):
        with Image.open(entry["image_path"]) as im:
            image=crop(im.convert("RGB"),512)
        ids=torch.tensor(tokenizer.encode(text,add_special_tokens=False),dtype=torch.long)[None]
        count=entry["ovt_count"]
        positions=(ids[0]==token_ids["ovt"]).nonzero().flatten()
        if len(positions)!=count or count>50 or ids.shape[1]>args.max_caption_tokens:
            raise ValueError("AR caption limits/owner indices differ; never silently truncate")
        packed=torch.zeros((1,50),dtype=torch.long);valid=torch.zeros((1,50),dtype=torch.bool)
        packed[0,:count]=positions;valid[0,:count]=True
        result=pgot_forward_eval(model,
            images=towers[0].image_processor.preprocess(image,return_tensors="pt")["pixel_values"],
            target_images=towers[-1].image_processor.preprocess(image,return_tensors="pt")["pixel_values"],
            caption_input_ids=ids,caption_attention_mask=torch.ones_like(ids,dtype=torch.bool),
            ovt_positions_in_caption=packed,ovt_valid_mask=valid)
        if not torch.equal(result["ovt_object_valid"].cpu(),valid):
            raise ValueError("Forward changed packed owner identities")
        things=torch.zeros_like(valid)
        for candidate in ar_pairs(text,count):
            if candidate["slot_index"] is not None: things[0,candidate["slot_index"]]=True
        prediction=build_pred_mask_ovt_owner_eval(result["ovt_object_probs"],result["ovt_void_probs"],
                      valid.to(device),things.to(device),512,1)[0].cpu().numpy()
        data={k:result[k] for k in keys}
        if not all(torch.isfinite(v).all() for v in data.values()):
            raise ValueError("Nonfinite extracted owner representation")
        return data,image,prediction
    def get_owner(iid):
        entry=m["entries"][str(iid)];folder=root/"owners"/str(iid)
        if (folder/"metadata.json").exists():
            info=json.loads((folder/"metadata.json").read_text())
            if info["fingerprint"]!=m["fingerprint"] or info["caption_sha256"]!=entry["caption_sha256"]:
                raise ValueError("Owner-cache provenance mismatch")
            return ({k:v.to(device) for k,v in torch.load(folder/"bundle.pt",map_location="cpu",weights_only=True).items()},
                    Image.open(folder/"image.png").convert("RGB"),np.load(folder/"prediction.npz")["prediction"])
        data,image,prediction=forward(entry,entry["caption"])
        folder.mkdir(parents=True,exist_ok=True)
        torch.save({k:v.detach().cpu() for k,v in data.items()},folder/"bundle.pt")
        image.save(folder/"image.png");np.savez_compressed(folder/"prediction.npz",prediction=prediction.astype(np.uint16))
        write_json(folder/"metadata.json",dict(fingerprint=m["fingerprint"],caption_sha256=entry["caption_sha256"]))
        return data,image,prediction
    with torch.inference_mode():
        for pair in pending:
            started=time.perf_counter();folder=root/"pairs"/pair["pair_id"];folder.mkdir(parents=True,exist_ok=True)
            a,image_a,pred_a=get_owner(pair["target_image_id"])
            b,image_b,pred_b=get_owner(pair["donor_image_id"])
            objects,registers=owner_units(a);donors,_=owner_units(b)
            if len(registers)!=4: raise ValueError("Expected four original background registers")
            k,j=pair["target_slot"],pair["donor_slot"]
            target_mask=pred_a==pair["target_prediction_label"];donor_mask=pred_b==pair["donor_prediction_label"]
            text_result,_,_=forward(m["entries"][str(pair["target_image_id"])],pair["text_only_caption"])
            text_semantic=owner_units(text_result)[0][k][0]
            noise=seeded_noise(model,[pair["target_image_id"]],args.seed,device)
            images,conditions={},{}
            for condition in CONDITIONS:
                units=transfer_objects(objects,k,donors,j,condition,text_semantic)
                hidden=condition_from_units(model,a["raw_rae_hidden"],units,registers,8,32)
                if not torch.isfinite(hidden).all(): raise ValueError("Nonfinite Reader output")
                conditions[condition]=hidden
                if condition=="reconstruction":
                    relative=float((hidden-a["rae_hidden"]).abs().max()/a["rae_hidden"].abs().mean().clamp_min(1e-12))
                    if relative>1e-3: raise ValueError(f"Owner rebuild identity failed: {relative}")
                generated=generate_native(hidden,noise)
                if not torch.isfinite(generated).all(): raise ValueError("Nonfinite decoded image")
                images[condition]=tensor_image(generated);images[condition].save(folder/f"{condition}_native.png")
            image_a.save(folder/"source_A.png");image_b.save(folder/"donor_B.png")
            Image.fromarray(target_mask).save(folder/"target_prediction_mask.png")
            Image.fromarray(donor_mask).save(folder/"donor_prediction_mask.png")
            refresh_figure(folder,pair)
            panel_grid([images[c] for c in CONDITIONS],list(CONDITIONS),folder/"ablation.png",size=256)
            region=np.asarray(Image.fromarray(target_mask|donor_mask).resize(images["reconstruction"].size,Image.Resampling.NEAREST),bool)
            recon=np.asarray(images["reconstruction"]).astype(float)/255
            stats={c:edit_stats(np.asarray(images[c]).astype(float)/255,recon,region) for c in CONDITIONS}
            # Explicit target-only diagnostics prevent donor-native placement from hiding off-target changes.
            target_region=np.asarray(Image.fromarray(target_mask).resize(images["reconstruction"].size,Image.Resampling.NEAREST),bool)
            target_stats={c:edit_stats(np.asarray(images[c]).astype(float)/255,recon,target_region) for c in CONDITIONS}
            condition_change={c:float((h-conditions["reconstruction"]).abs().mean()) for c,h in conditions.items()}
            write_json(folder/"metadata.json",dict(pair,fingerprint=m["fingerprint"],
                identity_relative_error=relative,native_output_size=list(images["reconstruction"].size),
                target_region_pixels=int(target_mask.sum()),donor_region_pixels=int(donor_mask.sum()),
                change_diagnostics=stats,target_only_change_diagnostics=target_stats,
                reader_condition_mean_abs_change=condition_change,
                diagnostic_region="union of target and donor predicted regions, plus separate target-only diagnostics",
                visual_success_verified=False,seconds=time.perf_counter()-started))
            export(root,m)
            print(f"Completed {pair['pair_id']}: identity={relative:.3g}, native size={images['reconstruction'].size}, {time.perf_counter()-started:.1f}s",flush=True)
            del a,b,text_result,objects,registers,donors,conditions
    print(f"Transfer pilot ready: {root/'contact_sheet.jpg'}",flush=True)


def audit(args):
    import torch
    root,m=manifest(args)
    expected=m["pairs"][:args.limit] if args.limit else m["pairs"]
    reports=[]
    for pair in expected:
        folder=root/"pairs"/pair["pair_id"]
        row=json.loads((folder/"metadata.json").read_text())
        assert row["fingerprint"]==m["fingerprint"]
        assert row["identity_relative_error"]<=1e-3
        assert set(row["change_diagnostics"])==set(CONDITIONS)
        assert row["target_region_pixels"]>=0 and row["donor_region_pixels"]>=0
        for condition in CONDITIONS:
            image=np.asarray(Image.open(folder/f"{condition}_native.png").convert("RGB"))
            assert list(image.shape[1::-1])==row["native_output_size"]
            assert np.isfinite(image).all()
        for condition,stats in row["change_diagnostics"].items():
            assert all(np.isfinite(v) for v in stats.values())
            assert 0<=stats["outside_change"]<=1 and 0<=stats["inside_change"]<=1
        assert row["change_diagnostics"]["reconstruction"]["inside_change"]==0
        assert row["change_diagnostics"]["reconstruction"]["outside_change"]==0
        for role in ("target","donor"):
            folder_owner=root/"owners"/str(pair[f"{role}_image_id"])
            info=json.loads((folder_owner/"metadata.json").read_text())
            assert info["fingerprint"]==m["fingerprint"]
            bundle=torch.load(folder_owner/"bundle.pt",map_location="cpu",weights_only=True)
            assert all(torch.isfinite(v).all() for v in bundle.values())
        reports.append(dict(pair_id=pair["pair_id"],identity_error=row["identity_relative_error"],
                            empty_predicted_region=(row["target_region_pixels"]==0 or row["donor_region_pixels"]==0),
                            native_size=row["native_output_size"]))
    write_json(root/"validation.json",dict(runtime_and_cache_validation_passed=True,
               checked_pairs=len(reports),conditions_per_pair=len(CONDITIONS),pairs=reports,
               empty_predicted_region_pairs=sum(r["empty_predicted_region"] for r in reports),
               semantic_edit_success_not_assessed=True))
    print(f"Audit passed: {len(reports)} pairs, {len(CONDITIONS)} conditions each, finite bundles/images, exact reconstruction sanity.",flush=True)


def refresh(args):
    root,m=manifest(args);export(root,m)
    print(f"Refreshed figures: {root/'contact_sheet.jpg'}",flush=True)


def review(args):
    root,m=manifest(args)
    report=json.loads((root/"visual_review.json").read_text())
    items=report["pairs"]
    if set(items)!={p["pair_id"] for p in m["pairs"]}:
        raise ValueError("Review must include every fixed pair, not only successes")
    panels=[]
    for pair in m["pairs"]:
        row=items[pair["pair_id"]]
        folder=root/"pairs"/pair["pair_id"]
        refresh_figure(folder,pair)
        picture=Image.open(folder/"figure.png").convert("RGB")
        panel=Image.new("RGB",(picture.width,picture.height+40),"white")
        panel.paste(picture,(0,40))
        draw=ImageDraw.Draw(panel)
        draw.text((8,5),f"{row['status']} | {pair['pair_id']} | assistant visual pilot review, not blind benchmark",fill="black")
        draw.text((8,23),row["short_note"],fill="black")
        panel.save(folder/"review.png");panels.append(panel)
    sheet=Image.new("RGB",(panels[0].width,sum(p.height for p in panels)),"white")
    y=0
    for panel in panels:
        sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(root/"review_contact_sheet.jpg",quality=94)
    print(f"Reviewed all {len(panels)} fixed pairs: {root/'review_contact_sheet.jpg'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","render","audit","refresh","review"],required=True)
    p.add_argument("--output_dir",default="/home/jovyan/PGOT/outputs/object_representation_transfer_memory8_register32")
    p.add_argument("--model_path",default="/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000")
    p.add_argument("--pairs",default="/home/jovyan/PGOT/configs/object_representation_transfer_pairs.jsonl")
    p.add_argument("--annotations",default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl")
    p.add_argument("--captions",default="/home/jovyan/PGOT/outputs/segeval_pgot_memory8_full_coco_voc_ar/coco_ar/generated_captions.jsonl")
    p.add_argument("--caption_summary",default="/home/jovyan/PGOT/outputs/segeval_pgot_memory8_full_coco_voc_ar/coco_ar/summary.json")
    p.add_argument("--dtype",choices=["fp32"],default="fp32")
    p.add_argument("--diffusion_inference_steps",type=int,default=10)
    p.add_argument("--guidance_scale",type=float,default=1.)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--max_caption_tokens",type=int,default=4096)
    p.add_argument("--cpu_threads",type=int,default=3)
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    if args.limit<0 or min(args.cpu_threads,args.max_caption_tokens,args.diffusion_inference_steps)<1:
        p.error("Limits must be nonnegative; sizes and steps positive")
    globals()[args.stage](args)


if __name__=="__main__":main()
