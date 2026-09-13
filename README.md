# Inner Compass

A static, single-page FlyWire heading-circuit experiment. Plain HTML, CSS, JavaScript modules, Canvas 2D and WebGL. No framework, backend, runtime package dependencies, or API keys. The measured wiring is included in `public/data`; the optional anatomy view fetches published v783 skeleton endpoints only when opened.

## Preview

From the project root:

```powershell
rtk python -m http.server 8000 --bind 127.0.0.1 --directory public
```

Open http://127.0.0.1:8000. Use an HTTP server; opening `index.html` through `file://` cannot fetch the binary. The landing page includes a 3D WebGL ring, a 2D fallback, and a lazy **Real anatomy** view. Drag the 3D canvas to orbit; pinch or use the +/- controls to zoom; Home resets the camera; tap a neuron to inspect its ID and activity. On touch devices controls use 44px targets and the canvas does not capture normal page scrolling. Move the slider left or right, center it to remove the turn cue, then switch to **Shuffled wiring**. Reset restores an identical initial patch in either mode. Switching modes retains current rates; Shuffle again chooses a new reproducible seed. Pause freezes dynamics; Escape centers the slider. Reduced-motion preferences start the simulation paused.

## Vercel (free Hobby tier)

Import this folder as a Git repository into Vercel. Choose **Other** for the framework, leave the Build Command empty, and use **public** as the Output Directory. `vercel.json` sets the static output directory. There is no build step or server function; Python is used only offline to regenerate data. Commit both exported data files. The raw `.cache` directory is ignored and must not be deployed.

Alternatively, run `rtk npx vercel` from the root and follow Vercel's account/project prompts. No deployment has been made by this project setup. [Vercel static build settings](https://vercel.com/docs/builds/configure-a-build#skip-build-step).

## Rebuild the real data

Python 3.10+ and NumPy are required only for export:

```powershell
rtk python -m pip install -r requirements.txt
rtk python scripts/export_connectome.py
```

The exporter downloads four gzipped CSV tables from the [public FlyWire v783 storage archive](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/consolidated_cell_types.csv.gz). This is the archive used by the [official Codex loader](https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py); it does not depend on the interactive Codex sign-in endpoint. The first download includes the whole connection CSV and can take a few minutes. Sources are cached in `.cache/flywire-783`. Downloads use temporary files and validate gzip before promoting them to cache.

```powershell
rtk python scripts/export_connectome.py --offline
```

Selection: exact primary types EPG, PEG, PEN_a/PEN1, PEN_b/PEN2 and Delta7, with at least one source-table synapse in EB or PB. All connections between the retained neurons are aggregated across neuropils, including NO. EPGt is excluded. The published source table is already thresholded; omitted weak connections cannot be recovered. No further threshold is imposed. The matrix contains **147 neurons, 2,529 directed nonzero pairs, and 35,781 synapses**. It is a selected subgraph, not the entire central complex.

Four of the 151 candidate cells have no resolved transmitter prediction (two Delta7, two PENb). They are excluded and individually listed in the JSON. Acetylcholine receives +1, GABA −1, glutamate −1. Unknown or modulatory transmitters are never silently treated as excitatory. Glutamate's actual effect depends on receptors; the negative sign is an explicit modeling assumption motivated by the CX literature, not a measured per-edge sign.

## File contract

* `public/data/connections.bin`: **86,436 bytes**, dense 147×147 little-endian float32, source-major row order. Element `[source * N + target]` is the **signed integer synapse count**. Zero means absent in this export. Self-connections present in the source are retained.
* `public/data/neurons.json`: compact manifest plus neurons in exact matrix order. Root IDs are strings to avoid JavaScript's 53-bit integer limit. Includes original/canonical cell type, side, predicted transmitter and score, modeled sign, EB/PB membership and inferred layout angle. Also includes input URLs and SHA-256 checksums, output checksum, selection policy, excluded IDs, and citations.

The browser checks dimensions, signs, finite integer counts, connection/synapse totals and the binary SHA-256 (when Web Crypto is available). Failed loading produces an explicit error; there is no synthetic fallback.

## Dynamics and limits

Every neuron obeys `tau * dr/dt = -r + clip(W*r + baseline + cue, 0, 1)`, with tau=150 ms. Euler integration uses fixed 1/120 s steps with a bounded frame accumulator, synchronous rates, and no background-tab catch-up. All edges contribute every step; only the drawn edge sample is reduced.

`model.mjs` normalizes each target's incoming synapse counts separately for each source population, then applies gains: EPG→EPG 1.4; EPG→other 1.0; Delta7→EPG 1.4 with negative sign; Delta7→other 0.2 with negative sign; all remaining source populations 0.15. Baselines are 0.025 for EPG and 0.02 for other cells. These are hand-selected model parameters, not experimentally fitted physiological constants.

Turning advances a cue angle at the selected degrees/second. EPG input is `2.8 * exp(6 * (cos(neuronAngle - cueAngle) - 1)) - 1.3` while turning, and zero at rest. This is a simplified sensory heading cue, **not a demonstrated reconstruction of PEN-mediated angular-velocity integration**. A strong cue can briefly organize either wiring condition; compare persistence after centering the slider. Real wiring can drift toward preferred discrete headings after stopping. It does not retain every arbitrary angle perfectly.

For layout, the exporter computes cosine similarity of EPG outgoing connectivity profiles, embeds the normalized affinity with two nonconstant eigenvectors, and maps it onto a circle. Relay/inhibitory positions use the circular mean of their EPG inputs. Angles are inferred from wiring, not measured anatomical positions, and their origin and handedness are arbitrary.

Shuffling uses seeded Fisher–Yates to permute all destination slots within each source row, including zeros, **after** normalization. This exactly preserves each source's out-degree, weight multiset and sign (and hence the total connection count). It does not preserve in-degree, target population totals or self-loop count. No special noise, decay, fabricated connections or dynamics are added in shuffled mode. The same initialization, input, solver and gains apply in both modes. The displayed coherence is `abs(sum(r_EPG * exp(i*angle))) / sum(r_EPG)`; it measures spatial concentration, not agreement with real neural activity.

This is a connectome-constrained educational model, not proof that connectivity alone uniquely determines biological function.

## Verification

```powershell
rtk node --test tests/model.test.mjs
rtk python tests/verify_sources.py
```

The Node tests cover binary corruption, exact shuffle invariants, stable patches from six initial headings, loss of coherence across 20 independent shuffles, full turns in both directions, bounded rates and integration input checks. The independent Python audit checks every matrix entry against source CSV counts and verifies all source hashes (requires the downloaded cache).

## Scientific attribution

* [Dorkenwald et al., 2024 — Neuronal wiring diagram of an adult brain](https://doi.org/10.1038/s41586-024-07558-y).
* [Schlegel et al., 2024 — Whole-brain annotation and multi-connectome cell typing](https://doi.org/10.1038/s41586-024-07686-5).
* [Eckstein et al., 2024 — Neurotransmitter classification from electron microscopy images](https://doi.org/10.1016/j.cell.2024.03.016).
* [Turner-Evans et al., 2020 — The neuroanatomical ultrastructure and function of a biological ring attractor](https://pmc.ncbi.nlm.nih.gov/articles/PMC8356802/).
* [FlyWire Consortium v783 connectivity release](https://zenodo.org/records/10676866), CC BY 4.0. Bundled files are a transformed subset; the source papers and dataset retain their respective rights and attribution.
