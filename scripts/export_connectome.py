#!/usr/bin/env python3
"""Export the FlyWire FAFB v783 EB/PB compass subgraph. Requires numpy only.

Raw downloads are cached locally; only public/data belongs in the deployment.
Use --offline to rebuild from cache without contacting FlyWire.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from collections import Counter, defaultdict

import numpy as np

BASE = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/'
TYPES = {'EPG': 'EPG', 'PEG': 'PEG', 'PEN_a/PEN1': 'PENa',
         'PEN_b/PEN2': 'PENb', 'Delta7': 'Delta7'}
FILES = ('consolidated_cell_types', 'neurons', 'classification', 'connections')
REGIONS = {'EB', 'PB'}


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def download(name, cache, offline):
    path = cache / (name + '.csv.gz')
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'Missing cached source: {path}')
        print(f'Downloading {BASE}{path.name}', flush=True)
        partial = path.with_suffix('.part')
        try:
            with urllib.request.urlopen(BASE + path.name, timeout=120) as response:
                with partial.open('wb') as handle:
                    shutil.copyfileobj(response, handle)
            # Validate the gzip before promoting a partial download to the cache.
            with gzip.open(partial, 'rb') as handle:
                while handle.read(1024 * 1024):
                    pass
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def infer_angles(counts, neurons):
    """Spectral layout, NOT measured morphology or anatomical EB wedges."""
    epg = [i for i, n in enumerate(neurons) if n['type'] == 'EPG']
    # Shared EPG output partners include PB interneurons and relay cells;
    # this also embeds EPGs missing direct recurrent edges in the thresholded table.
    profiles = counts[epg, :].astype(float)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-9)
    affinity = profiles @ profiles.T
    np.fill_diagonal(affinity, 0)
    if np.any(affinity.sum(axis=1) == 0):
        raise ValueError('Cannot infer layout: an EPG has no recurrent neighbors')
    degree = 1 / np.sqrt(affinity.sum(axis=1))
    _, vectors = np.linalg.eigh(affinity * degree[:, None] * degree[None, :])
    embedding = vectors[:, -3:-1] * degree[:, None]
    angles = np.arctan2(embedding[:, 0], embedding[:, 1])
    # Fix rotation/reflection deterministically; zero is arbitrary, not north.
    angles -= angles[0]
    if np.sin(angles[1]) < 0:
        angles *= -1
    result = np.zeros(len(neurons))
    result[epg] = angles % (2 * np.pi)
    for i, neuron in enumerate(neurons):
        if i in epg:
            continue
        weights = counts[epg, i]  # incoming EPG partners
        z = (weights * np.exp(1j * result[epg])).sum()
        result[i] = np.angle(z) % (2 * np.pi)
    return result


def export(cache, output, offline=False):
    cache.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name, cache, offline) for name in FILES}
    selected = {r['root_id']: r['primary_type'] for r in rows(paths[FILES[0]])
                if r['primary_type'] in TYPES}
    predictions = {r['root_id']: r for r in rows(paths['neurons']) if r['root_id'] in selected}
    classes = {r['root_id']: r for r in rows(paths['classification']) if r['root_id'] in selected}
    # First prove EB/PB membership from synapses, then retain all measured
    # connections between those cells (including their NO loops).
    membership = defaultdict(set)
    edges = defaultdict(int)
    source_edge_rows = 0
    for r in rows(paths['connections']):
        pre, post, region = r['pre_root_id'], r['post_root_id'], r['neuropil']
        count = int(r['syn_count'])
        if count <= 0:
            raise ValueError('Non-positive source synapse count')
        if region in REGIONS:
            for root in (pre, post):
                if root in selected:
                    membership[root].add(region)
        if pre in selected and post in selected:
            edges[pre, post] += count
            source_edge_rows += 1
    excluded = [{'id': root, 'sourceType': selected[root], 'reason': 'unresolved predicted neurotransmitter'}
                for root in sorted(selected) if membership[root] and
                predictions.get(root, {}).get('nt_type', '').upper() not in {'ACH', 'GABA', 'GLUT'}]
    excluded_ids = {n['id'] for n in excluded}
    ids = sorted(root for root in selected if membership[root] and root not in excluded_ids)
    if not ids:
        raise ValueError('No compass neurons found; verify source schema and release')
    neurons = []
    for root in ids:
        prediction = predictions[root]
        nt = prediction['nt_type'].upper()
        # Glutamate sign is receptor dependent. Inhibitory here is an explicit
        # CX modeling assumption, not a measured sign for every synapse.
        if nt not in {'ACH', 'GABA', 'GLUT'}:
            raise ValueError(f'Unresolved neurotransmitter {nt!r} for {root}; do not guess a sign')
        neurons.append({'id': root, 'type': TYPES[selected[root]], 'sourceType': selected[root],
                        'nt': nt, 'ntScore': float(prediction['nt_type_score']),
                        'sign': 1 if nt == 'ACH' else -1,
                        'side': classes.get(root, {}).get('side', ''),
                        'neuropils': sorted(membership[root])})
    index = {root: i for i, root in enumerate(ids)}
    counts = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for (pre, post), count in edges.items():
        if pre in index and post in index:
            counts[index[pre], index[post]] = count
    angles = infer_angles(counts, neurons)
    for neuron, angle in zip(neurons, angles):
        neuron['angle'] = round(float(angle), 7)
    signed = counts * np.array([n['sign'] for n in neurons], dtype=np.float32)[:, None]
    binary = signed.astype('<f4').tobytes(order='C')
    manifest = {
        'schemaVersion': 1, 'dataset': 'FlyWire FAFB', 'version': 783,
        'neuronCount': len(ids), 'edgeCount': int(np.count_nonzero(counts)),
        'synapseCount': int(counts.sum()), 'sourceEdgeRows': source_edge_rows,
        'typeCounts': dict(Counter(n['type'] for n in neurons)),
        'matrix': {'file': 'connections.bin', 'dtype': 'float32', 'endian': 'little',
                   'shape': [len(ids), len(ids)], 'order': 'row-major',
                   'orientation': 'source,target', 'values': 'signed measured synapse counts',
                   'sha256': hashlib.sha256(binary).hexdigest()},
        'selection': 'EPG, PEG, PEN_a/PEN1, PEN_b/PEN2, Delta7 with synapses in EB or PB; all intra-subgraph neuropils retained. EPGt excluded.',
        'signPolicy': 'ACH +1; GABA -1; GLUT -1 (receptor-dependent modeling assumption). Unresolved predictions are excluded and listed.',
        'excludedNeurons': excluded,
        'layout': 'Angles inferred from two nonconstant eigenvectors of normalized cosine similarity of EPG outgoing connectivity profiles; other cells follow weighted EPG input angle. Arbitrary rotation/reflection; not anatomical positions.',
        'sourceThreshold': 'Published Codex connections table; no additional threshold applied. Missing rows are treated as zero.',
        'sources': [{'url': BASE + path.name, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
                    for path in paths.values()],
        'citations': [
            {'title': 'Dorkenwald et al. (2024), Neuronal wiring diagram of an adult brain', 'url': 'https://doi.org/10.1038/s41586-024-07558-y'},
            {'title': 'Schlegel et al. (2024), Whole-brain annotation and multi-connectome cell typing', 'url': 'https://doi.org/10.1038/s41586-024-07686-5'},
            {'title': 'Turner-Evans et al. (2020), The neuroanatomical ultrastructure and function of a biological ring attractor', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8356802/'},
            {'title': 'FlyWire connectivity data, CC BY 4.0', 'url': 'https://zenodo.org/records/10676866'}],
        'neurons': neurons}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'connections.bin').write_bytes(binary)
    (output / 'neurons.json').write_text(json.dumps(manifest, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f"Exported {len(ids)} neurons, {manifest['edgeCount']} edges, {manifest['synapseCount']} synapses; {len(binary):,} matrix bytes", flush=True)
    print(manifest['typeCounts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/flywire-783'))
    parser.add_argument('--output', type=Path, default=Path('public/data'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    export(args.cache, args.output, args.offline)
