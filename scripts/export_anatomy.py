#!/usr/bin/env python3
"""Fetch matching FlyWire v783 skeletons and export a mobile-sized 3D asset.

Uses the per-neuron endpoint documented by fafbseg.flywire.get_skeletons.
Dependencies: numpy. No CAVE credentials, meshes, or whole-brain download.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import numpy as np

BASE = 'https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/'


def decode_skeleton(data):
    if len(data) < 8:
        raise ValueError('Truncated skeleton')
    vertices, edges = struct.unpack_from('<II', data)
    # The documented endpoint includes one float32 radius per vertex.
    if vertices == 0 or len(data) != 8 + 16 * vertices + 8 * edges:
        raise ValueError('Unexpected skeleton byte length')
    points = np.frombuffer(data, dtype='<f4', count=vertices*3, offset=8).reshape(-1, 3).astype(float)
    links = np.frombuffer(data, dtype='<u4', count=edges*2, offset=8+vertices*12).reshape(-1, 2)
    if not np.isfinite(points).all() or (edges and links.max() >= vertices):
        raise ValueError('Invalid skeleton coordinates or indices')
    return points, links


def simplify_path(path, points, tolerance):
    """RDP on an unbranched path; returns only actual source vertices."""
    if len(path) <= 2:
        return path
    selected = {0, len(path)-1}
    pending = [(0, len(path)-1)]
    p = points[path]
    while pending:
        start, end = pending.pop()
        if end-start < 2:
            continue
        vector = p[end]-p[start]
        norm = vector @ vector
        interior = p[start+1:end]
        fraction = np.clip((interior-p[start]) @ vector / max(norm, 1e-12), 0, 1)
        distances = np.linalg.norm(interior-(p[start]+fraction[:, None]*vector), axis=1)
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = start+1+index
            selected.add(split)
            pending.extend([(start, split), (split, end)])
    return [path[i] for i in sorted(selected)]


def simplify_skeleton(points, links, tolerance):
    adjacent = [[] for _ in points]
    for a, b in links:
        adjacent[a].append(int(b)); adjacent[b].append(int(a))
    seen = set()
    result = []
    # Branch points, roots and leaves are preserved. No twigs are pruned.
    starts = [i for i, neighbors in enumerate(adjacent) if len(neighbors) != 2]
    starts += [i for i, neighbors in enumerate(adjacent) if len(neighbors) == 2]
    for start in starts:
        for neighbor in adjacent[start]:
            edge = tuple(sorted((start, neighbor)))
            if edge in seen:
                continue
            seen.add(edge)
            path = [start, neighbor]
            while len(adjacent[path[-1]]) == 2:
                next_node = next(n for n in adjacent[path[-1]] if n != path[-2])
                edge = tuple(sorted((path[-1], next_node)))
                if edge in seen:
                    break
                seen.add(edge); path.append(next_node)
            simple = simplify_path(path, points, tolerance)
            result.extend(zip(simple[:-1], simple[1:]))
    if len(seen) != len({tuple(sorted(map(int,e))) for e in links}):
        raise ValueError('Skeleton traversal missed an edge')
    return np.asarray(result, dtype=int).reshape(-1, 2)


def fetch(root, cache, offline):
    path = cache / f'{root}.bin'
    if not path.exists():
        if offline:
            raise FileNotFoundError(path)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE+root, timeout=60) as response:
                    data = response.read()
                decode_skeleton(data)
                temporary = path.with_suffix('.part')
                temporary.write_bytes(data); temporary.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    data = path.read_bytes()
    return data, *decode_skeleton(data)


def export(meta_path, cache, output, tolerance, offline):
    neurons = json.loads(meta_path.read_text())['neurons']
    cache.mkdir(parents=True, exist_ok=True)
    collected = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = pool.map(lambda n: fetch(n['id'], cache, offline), neurons)
        for i, (neuron, (raw, points, links)) in enumerate(zip(neurons, jobs)):
            simplified = simplify_skeleton(points, links, tolerance)
            collected.append((neuron, raw, points, links, simplified))
            if (i+1) % 15 == 0 or i == len(neurons)-1:
                print(f'Skeletons: {i+1}/{len(neurons)}', flush=True)
    all_points = np.concatenate([item[2] for item in collected])
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    center = (maximum+minimum)/2
    # Keep raw coordinates and use a fixed axis mapping, no spatial distortion.
    # Screen default: raw x horizontal, -raw y vertical, raw z depth.
    scale = 2.35 / np.max(maximum-minimum)
    transform = np.diag([1., -1., 1.])
    segments, manifest_cells = [], []
    offset = 0
    for neuron, raw, points, original, simplified in collected:
        normalized = ((points-center) @ transform)*scale
        segments.append(normalized[simplified].reshape(-1, 6))
        manifest_cells.append({'id': neuron['id'], 'type': neuron['type'], 'start': offset,
            'count': len(simplified), 'sourceVertices': len(points), 'sourceEdges': len(original),
            'sourceSha256': hashlib.sha256(raw).hexdigest(), 'sourceUrl': BASE+neuron['id']})
        offset += len(simplified)
    binary = np.concatenate(segments).astype('<f4').tobytes()
    manifest = {'schemaVersion': 1, 'dataset': 'FlyWire FAFB', 'version': 783,
        'format': 'float32-le-segments-xyzxyz', 'file': 'anatomy.bin',
        'neuronCount': len(neurons), 'segmentCount': offset, 'byteLength': len(binary),
        'sha256': hashlib.sha256(binary).hexdigest(),
        'source': BASE, 'sourceFormat': 'Neuroglancer precomputed skeleton with float32 radius',
        'sourceUnits': 'nanometers', 'centerNm': center.tolist(), 'scale': float(scale),
        'axisTransform': [1, -1, 1], 'boundsNm': [minimum.tolist(), maximum.tolist()],
        'simplification': {'method': 'Ramer-Douglas-Peucker per unbranched path', 'toleranceNm': tolerance,
            'preserves': 'All branch points and endpoints; every exported endpoint is a source vertex; no branches pruned.'},
        'description': 'Skeleton centerlines, not surfaces or synaptic edges. All 147 simulated neuron IDs match the v783 morphology source. Display brightness is per-neuron model activity.',
        'citation': {'title': 'Schlegel et al. (2024), Whole-brain annotation and multi-connectome cell typing',
            'url': 'https://doi.org/10.1038/s41586-024-07686-5'},
        'archive': 'https://zenodo.org/records/10877326', 'neurons': manifest_cells}
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('anatomy.bin', binary), ('anatomy.json', (json.dumps(manifest, separators=(',', ':'))+'\n').encode())]:
        temporary = output / (name+'.tmp'); temporary.write_bytes(data); temporary.replace(output/name)
    print(f'Exported {len(neurons)} skeletons, {offset:,} segments, {len(binary):,} bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('public/data/neurons.json'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/skeletons-783'))
    parser.add_argument('--output', type=Path, default=Path('public/data'))
    parser.add_argument('--tolerance-nm', type=float, default=800)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.tolerance_nm <= 0:
        parser.error('Tolerance must be positive')
    export(args.metadata, args.cache, args.output, args.tolerance_nm, args.offline)
