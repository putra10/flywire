"""Independent audit: re-sum public CSV rows and compare every exported edge."""
from pathlib import Path
import csv, gzip, hashlib, json, struct
from collections import defaultdict

root = Path(__file__).resolve().parents[1]
meta = json.loads((root / 'public/data/neurons.json').read_text())
binary = (root / 'public/data/connections.bin').read_bytes()
ids = {n['id']: i for i,n in enumerate(meta['neurons'])}
actual = defaultdict(int)
path = root / '.cache/flywire-783/connections.csv.gz'
with gzip.open(path, 'rt', newline='') as handle:
    for row in csv.DictReader(handle):
        if row['pre_root_id'] in ids and row['post_root_id'] in ids:
            actual[ids[row['pre_root_id']], ids[row['post_root_id']]] += int(row['syn_count'])
flat = struct.unpack('<' + 'f' * (len(binary)//4),binary)
for source in range(len(ids)):
    for target in range(len(ids)):
        assert flat[source*len(ids)+target] == actual[source,target] * meta['neurons'][source]['sign']
for source in meta['sources']:
    path = root / '.cache/flywire-783' / source['url'].split('/')[-1]
    with path.open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==source['sha256']
print(f'PASS: every matrix entry matches measured source counts; all {len(meta["sources"])} source hashes verified.')
