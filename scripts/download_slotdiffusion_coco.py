"""Download only the two official COCO ZIP members, with CRC validation.

Uses bounded HTTP Range reads; never downloads the whole 6.25 GB archive.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://drive.usercontent.google.com/download?id=1PSElX2ucqqLuCjjl2_skM-7-qjwb2hWh&export=download&confirm=t"
MEMBERS = ("sa_ldm_dino_coco_params-res224.pth", "vqvae_coco_params-res224.pth")
SHA256 = {
    "sa_ldm_dino_coco_params-res224.pth": "653c7e539aae123b3b9051124414626909d1f1f35106a7cff521ee2afd9e56ca",
    "vqvae_coco_params-res224.pth": "55ef1112fd835364a979ddc71e75980f7ad20cf77507aaab17b0c191510c6092",
}


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1,
                           status_forcelist=(429, 500, 502, 503, 504))))
        with self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError("Server does not support Range; refusing full archive download")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.cache_start, self.cache = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        if not (self.cache_start <= start and start + size <= self.cache_start + len(self.cache)):
            end = min(self.size, start + max(size, 8 * 1024 ** 2)) - 1
            with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60, stream=True) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Unexpected Range response")
                chunks, total = [], 0
                for chunk in response.iter_content(chunk_size=1024 ** 2):
                    total += len(chunk)
                    if total > end - start + 1:
                        raise RuntimeError("Range body exceeds requested bound")
                    chunks.append(chunk)
                self.cache_start, self.cache = start, b"".join(chunks)
            if len(self.cache) != end - start + 1:
                raise RuntimeError("Truncated Range response")
        offset = start - self.cache_start
        self.position += size
        return self.cache[offset:offset + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="/home/jovyan/baselines/SlotDiffusion/pretrained")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "coco_download_manifest.json"
    manifest = {"source": URL, "files": {}}
    with zipfile.ZipFile(RemoteZip(URL)) as archive:
        for name in MEMBERS:
            info = archive.getinfo(name)
            target = root / name
            if target.exists():
                digest = sha256(target)
                if digest != SHA256[name] or target.stat().st_size != info.file_size:
                    raise RuntimeError(f"Existing unverified file: {target}; refusing overwrite")
            else:
                temporary = target.with_suffix(".pth.part")
                print(f"Downloading {name}: {info.file_size / 1024 ** 2:.1f} MiB", flush=True)
                with archive.open(info) as source, temporary.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 ** 2)
                if temporary.stat().st_size != info.file_size:
                    raise RuntimeError("Downloaded size mismatch")
                digest = sha256(temporary)
                if digest != SHA256[name]:
                    raise RuntimeError(f"Official checkpoint SHA256 changed: {name}; refusing installation")
                temporary.replace(target)
            manifest["files"][name] = {"bytes": info.file_size, "zip_crc32": f"{info.CRC:08x}", "sha256": digest}
            # Save after each validated member so interrupted runs can resume safely.
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"Verified {name}: {digest}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
