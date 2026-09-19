"""Lay out actual editing candidates for manual review, without re-rendering."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()
    root = Path(args.output_dir)
    audit = root / 'audit'
    audit.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for mode in ['add', 'background']:
        for variant in ['native', 'predicted_guided']:
            rows = []
            paths = sorted((root / 'candidates' / mode).glob(f'{mode}_{variant}_*/metadata.json'))
            for path in paths:
                meta = json.loads(path.read_text())
                if not (path.parent / 'edited_native.png').exists():
                    continue
                row = Image.new('RGB', (1024, 300), 'white')
                draw = ImageDraw.Draw(row)
                draw.text((3, 2), f"{meta['task_id']} | donor {meta['donor_id']} ({meta['donor_category']})", fill='black', font=font)
                panels = [('original.png', 'Original'), ('panel_1.png', 'Donor region'),
                          ('reconstruction_native.png', 'Unedited reconstruction'), ('edited_native.png', 'Actual edit')]
                for col, (name, title) in enumerate(panels):
                    draw.text((col * 256 + 4, 24), title, fill='black', font=font)
                    panel = Image.open(path.parent / name).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                    row.paste(panel, (col * 256, 44))
                rows.append(row)
            for start in range(0, len(rows), 4):
                page_rows = rows[start:start+4]
                sheet = Image.new('RGB', (1024, 300 * len(page_rows)), 'white')
                for i, row in enumerate(page_rows):
                    sheet.paste(row, (0, 300 * i))
                dest = audit / f'{mode}_{variant}_{start//4+1:02}.jpg'
                sheet.save(dest, quality=95)
                print(dest)
    for variant in ['native', 'predicted_guided']:
        selection_path = root / f'final_{variant}_selection.json'
        if not selection_path.exists():
            continue
        selected = json.loads(selection_path.read_text())['groups']
        successes = [r for mode in ['add', 'background'] for r in selected[mode]
                     if r['visual_review']['status'] == 'success']
        if not successes:
            continue
        sheet = Image.new('RGB', (1024, 48 + 300*len(successes)), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), f'{variant.upper()} | visually selected examples, not a benchmark success rate', fill='black', font=font)
        draw.text((4, 24), 'Success = recognizable operation; no claim of pixel-exact foreground or donor appearance.', fill='black', font=font)
        for i, meta in enumerate(successes):
            folder = Path(meta['directory'])
            y = 48 + i*300
            draw.text((3, y), meta['task_id'], fill='black', font=font)
            for col, (name, title) in enumerate([('original.png', 'Original'), ('panel_1.png', 'Donor region'),
                    ('reconstruction_native.png', 'Unedited reconstruction'), ('edited_native.png', 'Actual edit')]):
                draw.text((col*256+4, y+24), title, fill='black', font=font)
                panel = Image.open(folder/name).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                sheet.paste(panel,(col*256,y+44))
        dest = root / f'best_{variant}.png'
        sheet.save(dest)
        print(dest)


if __name__ == '__main__':
    main()
