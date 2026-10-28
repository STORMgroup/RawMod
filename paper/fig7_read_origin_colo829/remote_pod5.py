#!/usr/bin/env python3
"""
remote_pod5.py — pull individual reads out of remote (S3/HTTPS) pod5 files
without downloading the whole file.

Why this exists
---------------
The COLO829 pod5 files are 0.25-1.0 GB each and hold ~4000 reads apiece, in
acquisition order. Reads that align to one genomic locus are scattered across
essentially as many pod5 files as there are reads (measured: 102 reads over a
10 kb window -> 89 distinct pod5 files), so "download the pod5 files you need"
degenerates to "download the whole 1.2 TB flowcell".

A pod5 file is a container holding three embedded Arrow IPC files (signal,
run_info, reads tables). The signal table is written in record batches of
`signal_batch_row_count` rows (100 here, ~6 MB/batch), and pod5's own reader
materialises one batch at a time. So the bytes actually needed for one read are
its signal batch plus the (small) reads/run_info tables at the tail of the
file — ~7 MB instead of ~250-1000 MB.

How it works
------------
`SparseMirror` keeps a local file that is byte-identical to the remote pod5 but
*sparse*: only the ranges that were actually touched are populated (the rest are
filesystem holes and consume no space). Reads go through a write-through cache:

  1. Fetch the file tail (adaptive) so `lib_pod5` can parse the footer and
     migrate/expose the reads + run_info tables, then open `pod5.Reader` on the
     mirror to resolve read_id -> row index -> signal-table row indices.
  2. Open the embedded signal table with pyarrow through a lazy file object
     wrapping the same mirror, and touch `get_batch(i)` for each batch a wanted
     read needs. Every byte pyarrow reads is fetched over HTTP Range and written
     through into the mirror at its true offset.
  3. Reopen `pod5.Reader` on the mirror. All bytes it needs are now present, so
     `reader.reads(selection=[...])` returns real, fully-decoded reads.

The mirror is only a staging area — `extract_reads()` copies the reads out into
a small, dense local pod5 via `pod5.Writer`, and the mirror can be deleted.

Standalone usage:
    python remote_pod5.py --url URL --read-ids id1,id2 --out out.pod5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence

import requests

# Ranges are fetched in aligned blocks so that many small pyarrow reads inside
# one record batch collapse into a handful of HTTP requests.
BLOCK = 1 << 20          # 1 MiB
# The reads + run_info tables sit at the tail, just before the footer, and are
# ~0.5 MB per 4000 reads; 4 MB covers every file in this dataset on the first
# try (each failed attempt costs a wasted lib_pod5 open + migration).
TAIL_TRIES = (1 << 22, 1 << 24, 1 << 26)            # 4M, 16M, 64M
HEAD_BYTES = 1 << 16     # 64 KiB (container magic + section header)


class RemoteFile:
    """HTTP Range reader with a session and a byte counter."""

    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()
        self.bytes_fetched = 0
        r = self.session.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])

    def get_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        end = min(start + length, self.size) - 1
        for attempt in range(5):
            try:
                r = self.session.get(
                    self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=300)
                r.raise_for_status()
                data = r.content
                self.bytes_fetched += len(data)
                return data
            except Exception:
                if attempt == 4:
                    raise
        raise RuntimeError('unreachable')


class SparseMirror:
    """A local sparse file mirroring a remote pod5, populated on demand."""

    def __init__(self, remote: RemoteFile, path: Path):
        self.remote = remote
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate to full size; unwritten regions stay as filesystem holes.
        if not self.path.exists() or self.path.stat().st_size != remote.size:
            with open(self.path, 'wb') as fh:
                fh.truncate(remote.size)
            self._have: set[int] = set()
        else:
            self._have = set()
        self._fh = open(self.path, 'r+b')

    @property
    def size(self) -> int:
        return self.remote.size

    def ensure(self, start: int, length: int) -> None:
        """Make sure [start, start+length) is present locally."""
        if length <= 0:
            return
        start = max(0, start)
        stop = min(self.remote.size, start + length)
        first, last = start // BLOCK, (stop - 1) // BLOCK
        run_start = None
        for blk in range(first, last + 2):
            missing = blk <= last and blk not in self._have
            if missing and run_start is None:
                run_start = blk
            elif not missing and run_start is not None:
                self._fetch_blocks(run_start, blk - 1)
                run_start = None

    def _fetch_blocks(self, first_blk: int, last_blk: int) -> None:
        off = first_blk * BLOCK
        nbytes = min((last_blk + 1) * BLOCK, self.remote.size) - off
        data = self.remote.get_range(off, nbytes)
        self._fh.seek(off)
        self._fh.write(data)
        self._fh.flush()
        for blk in range(first_blk, last_blk + 1):
            self._have.add(blk)

    def read_at(self, offset: int, length: int) -> bytes:
        self.ensure(offset, length)
        self._fh.seek(offset)
        return self._fh.read(length)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class LazySliceFile:
    """File-like view of [offset, offset+length) of a SparseMirror.

    Passed to ``pyarrow.PythonFile`` so pyarrow's Arrow-IPC reader can seek
    around an embedded table while the underlying bytes are fetched on demand.
    """

    def __init__(self, mirror: SparseMirror, offset: int, length: int):
        self.mirror = mirror
        self.offset = offset
        self.length = length
        self.pos = 0
        self.closed = False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        else:
            self.pos = self.length + pos
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.length - self.pos
        n = max(0, min(n, self.length - self.pos))
        data = self.mirror.read_at(self.offset + self.pos, n)
        self.pos += len(data)
        return data

    def close(self):
        self.closed = True

    def flush(self):
        pass


def _open_mirror_reader(mirror: SparseMirror):
    """Fetch head + an adaptive tail until pod5 can open the mirror."""
    import pod5

    mirror.ensure(0, HEAD_BYTES)
    last_err = None
    for tail in TAIL_TRIES:
        mirror.ensure(max(0, mirror.size - tail), min(tail, mirror.size))
        try:
            return pod5.Reader(mirror.path)
        except Exception as exc:          # footer/tables not fully covered yet
            last_err = exc
    raise RuntimeError(f'could not open pod5 mirror {mirror.path}: {last_err}')


def _signal_rows_for(reader, wanted: set[str]) -> Dict[str, List[int]]:
    """read_id -> the signal-table row indices holding that read's signal.

    Walks the (already-local, small) reads table rather than touching signal.
    """
    all_ids = list(map(str, reader.read_ids))
    positions = {rid: i for i, rid in enumerate(all_ids) if rid in wanted}
    if not positions:
        return {}
    by_pos = {}
    table = reader.read_table
    start = 0
    for b in range(table.num_record_batches):
        batch = table.get_batch(b)
        n = batch.num_rows
        sig = batch.column('signal')
        for rid, p in positions.items():
            if start <= p < start + n:
                by_pos[rid] = list(sig[p - start].as_py())
        start += n
    return by_pos


def fetch_reads_to_mirror(url: str, read_ids: Sequence[str], mirror_path: Path,
                          session: requests.Session | None = None,
                          signal_batch_rows: int | None = None,
                          verbose: bool = True):
    """Populate a sparse mirror of ``url`` with everything needed for ``read_ids``.

    ``signal_batch_rows`` is the signal table's record-batch height. It is a
    file-level constant chosen by the writing software, so it is learned once
    (costing one batch fetch) and then reused across every file from the same
    run. Pass None to learn it here.

    Returns (pod5.Reader on the mirror, mirror, found_read_ids, batch_rows).
    """
    import pod5
    import pyarrow as pa

    remote = RemoteFile(url, session)
    mirror = SparseMirror(remote, mirror_path)
    reader = _open_mirror_reader(mirror)

    wanted = set(read_ids)
    rid_to_rows = _signal_rows_for(reader, wanted)
    sig_loc = reader._file_reader.get_file_signal_table_location()
    reader.close()

    if not rid_to_rows:
        return None, mirror, [], signal_batch_rows

    slice_file = LazySliceFile(mirror, sig_loc.offset, sig_loc.length)
    ipc = pa.ipc.open_file(pa.PythonFile(slice_file, mode='r'))
    if signal_batch_rows is None:
        signal_batch_rows = ipc.get_batch(0).num_rows
    needed = {r // signal_batch_rows
              for rows in rid_to_rows.values() for r in rows}
    for b in sorted(needed):
        if 0 <= b < ipc.num_record_batches:
            ipc.get_batch(b)
    del ipc

    reader = pod5.Reader(mirror.path)
    if verbose:
        mb = remote.bytes_fetched / 1024 ** 2
        print(f'  {Path(url).name}: {len(rid_to_rows)} reads, '
              f'{len(needed)} signal batches, {mb:.1f} MB fetched '
              f'(file is {remote.size / 1024 ** 2:.0f} MB)', flush=True)
    return reader, mirror, sorted(rid_to_rows), signal_batch_rows


def extract_reads(url: str, read_ids: Sequence[str], writer,
                  mirror_dir: Path, session: requests.Session | None = None,
                  signal_batch_rows: int | None = None,
                  keep_mirror: bool = False, verbose: bool = True):
    """Copy ``read_ids`` from a remote pod5 into an open ``pod5.Writer``.

    Returns (written_read_ids, signal_batch_rows).
    """
    mirror_path = Path(mirror_dir) / (Path(url).name + '.mirror')
    reader, mirror, found, batch_rows = fetch_reads_to_mirror(
        url, read_ids, mirror_path, session, signal_batch_rows, verbose)
    written = []
    if found:
        for rec in reader.reads(selection=found, preload={'samples'}):
            writer.add_read(rec.to_read())
            written.append(str(rec.read_id))
        reader.close()
    mirror.close()
    if not keep_mirror:
        try:
            os.unlink(mirror_path)
        except OSError:
            pass
    return written, batch_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--read-ids', required=True,
                    help='comma-separated read ids, or @file with one per line')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mirror-dir', default='/tmp')
    a = ap.parse_args()

    import pod5
    if a.read_ids.startswith('@'):
        ids = [l.strip() for l in open(a.read_ids[1:]) if l.strip()]
    else:
        ids = a.read_ids.split(',')

    with pod5.Writer(a.out) as w:
        got, _ = extract_reads(a.url, ids, w, Path(a.mirror_dir))
    print(f'wrote {len(got)} reads to {a.out}')


if __name__ == '__main__':
    main()
