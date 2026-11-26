"""Download and extract the Kaggle HAM10000 dataset into data/raw/."""

import csv
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from io import SEEK_CUR, SEEK_END, SEEK_SET
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000"
RAW_DIR = Path(__file__).resolve().parent / "raw"
METADATA = "HAM10000_metadata.csv"
IMAGE_PARTS = ("HAM10000_images_part_1", "HAM10000_images_part_2")


class KaggleArchive:
    """Read the public ZIP by byte range with bounded read-ahead."""

    block_size = 4 * 1024 * 1024

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor
        self.pending = {}
        self.position = 0
        self.block_start = -1
        self.block = b""
        self.url = DATASET_URL
        self.size = None
        self._fetch(0, 0)

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(3):
            try:
                return self._fetch_range(start, end)
            except HTTPError as error:
                if attempt == 2 or error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
            except (OSError, HTTPException):
                if attempt == 2:
                    raise

    def _fetch_range(self, start: int, end: int) -> bytes:
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=60) as response:
            content_range = response.headers.get("Content-Range", "")
            prefix = f"bytes {start}-{end}/"
            if response.status != 206 or not content_range.startswith(prefix):
                raise OSError(f"Kaggle returned an unexpected archive range: {content_range!r}")
            size = int(content_range[len(prefix):])
            if self.size is not None and size != self.size:
                raise OSError("The Kaggle archive changed while it was being read.")
            if self.size is None:
                self.size = size
                self.url = response.geturl()
            data = response.read()
        if len(data) != end - start + 1:
            raise OSError("Kaggle returned a truncated archive range.")
        return data

    def _read_block(self, start: int) -> bytes:
        stop = min(start + 6 * self.block_size, self.size)
        needed = range(start, stop, self.block_size)
        for position in list(self.pending):
            if position not in needed:
                self.pending.pop(position).cancel()
        for position in needed:
            if position not in self.pending:
                end = min(position + self.block_size, self.size) - 1
                self.pending[position] = self.executor.submit(self._fetch, position, end)
        return self.pending.pop(start).result()

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = SEEK_SET) -> int:
        base = {SEEK_SET: 0, SEEK_CUR: self.position, SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise OSError("Cannot seek before the beginning of the archive.")
        self.position = base + offset
        return self.position

    def seekable(self) -> bool:
        return True

    def read(self, count: int = -1) -> bytes:
        end = self.size if count < 0 else min(self.position + count, self.size)
        chunks = []
        while self.position < end:
            start = self.position // self.block_size * self.block_size
            if start != self.block_start:
                self.block = self._read_block(start)
                self.block_start = start
            stop = min(end, start + len(self.block))
            chunks.append(self.block[self.position - start:stop - start])
            self.position = stop
        return b"".join(chunks)


def extract_dataset(archive: ZipFile, destination: Path) -> None:
    """Copy metadata and one JPEG per ID, using the two original image parts."""
    with archive.open(METADATA) as source:
        metadata = source.read()
    image_ids = {row["image_id"] for row in csv.DictReader(metadata.decode("utf-8-sig").splitlines())}
    images = {}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (
            len(path.parts) == 2
            and path.parent.name in IMAGE_PARTS
            and path.suffix.lower() == ".jpg"
            and path.stem in image_ids
        ):
            if path.stem in images:
                raise ValueError(f"Duplicate image ID in the two image parts: {path.stem}")
            images[path.stem] = member
    missing = image_ids - images.keys()
    if not image_ids or missing:
        raise ValueError(f"Incomplete HAM10000 archive. Missing image IDs: {sorted(missing)[:10]}")

    for part in IMAGE_PARTS:
        (destination / part).mkdir(exist_ok=True)
    for index, member in enumerate(sorted(images.values(), key=lambda member: member.header_offset), 1):
        path = PurePosixPath(member.filename)
        image_path = destination / path.parent.name / path.name
        if not image_path.is_file() or image_path.stat().st_size != member.file_size:
            partial_path = image_path.with_suffix(".part")
            with archive.open(member) as source, partial_path.open("wb") as target:
                copyfileobj(source, target)
            partial_path.replace(image_path)
        if index % 1000 == 0:
            print(f"Extracted {index:,} of {len(images):,} images.", flush=True)
    (destination / METADATA).write_bytes(metadata)


def main() -> None:
    """Fetch the public Kaggle archive and publish the dataset after extraction."""
    if RAW_DIR.exists():
        raise SystemExit(f"{RAW_DIR} already exists. Run python -m src.data to check its contents.")

    staging = RAW_DIR.parent / ".ham10000-stage.zip"
    staging.mkdir(exist_ok=True)
    print("Reading the HAM10000 Kaggle archive...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        with ZipFile(KaggleArchive(executor)) as archive:
            extract_dataset(archive, staging)
    staging.rename(RAW_DIR)
    print(f"HAM10000 is ready at {RAW_DIR}")


if __name__ == "__main__":
    main()
