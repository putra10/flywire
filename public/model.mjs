// All dynamics are shared by both modes. No angle-based recurrent connections
// are added; angles are used only for initialization, input, and the readout.
export const TAU = Math.PI * 2;
export const STEP = 1 / 120;
export const wrap = (a) => ((a % TAU) + TAU) % TAU;
export function randomGenerator(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function decodeMatrix(buffer, metadata) {
  const n = metadata.neuronCount;
  if (
    metadata.schemaVersion !== 1 ||
    metadata.neurons.length !== n ||
    metadata.matrix.dtype !== 'float32' ||
    metadata.matrix.endian !== 'little' ||
    metadata.matrix.orientation !== 'source,target' ||
    buffer.byteLength !== n * n * 4
  ) {
    throw new Error('The connectome files do not match. Re-export both data files.');
  }
  const view = new DataView(buffer);
  const matrix = new Float32Array(n * n);
  let edges = 0,
    synapses = 0;
  for (let i = 0; i < matrix.length; i++) {
    const value = view.getFloat32(i * 4, true);
    if (
      !Number.isFinite(value) ||
      Math.abs(value) % 1 ||
      (value && Math.sign(value) !== metadata.neurons[Math.floor(i / n)].sign)
    ) {
      throw new Error('Invalid synapse count or transmitter sign in the matrix.');
    }
    matrix[i] = value;
    if (value) edges++;
    synapses += Math.abs(value);
  }
  if (edges !== metadata.edgeCount || synapses !== metadata.synapseCount) {
    throw new Error('Connection totals do not match the metadata.');
  }
  return matrix;
}

export function normalizeWeights(counts, neurons) {
  const n = neurons.length,
    weights = new Float32Array(n * n);
  const types = [...new Set(neurons.map((n) => n.type))];
  for (const sourceType of types) {
    for (let target = 0; target < n; target++) {
      let total = 0;
      for (let source = 0; source < n; source++) {
        if (neurons[source].type === sourceType) total += Math.abs(counts[source * n + target]);
      }
      let gain = 0.15;
      if (sourceType === 'EPG') gain = neurons[target].type === 'EPG' ? 1.4 : 1;
      if (sourceType === 'Delta7') gain = neurons[target].type === 'EPG' ? 1.4 : 0.2;
      for (let source = 0; source < n; source++) {
        if (neurons[source].type === sourceType && total) {
          weights[source * n + target] = (gain * counts[source * n + target]) / total;
        }
      }
    }
  }
  return weights;
}

// Fisher-Yates over every destination slot in each source row, including zeros.
// It preserves exact out-degree, outgoing weight multiset, and Dale sign.
export function shuffleWeights(matrix, n, seed) {
  const random = randomGenerator(seed),
    shuffled = matrix.slice();
  for (let source = 0; source < n; source++) {
    const offset = source * n;
    for (let target = n - 1; target > 0; target--) {
      const other = Math.floor(random() * (target + 1));
      const value = shuffled[offset + target];
      shuffled[offset + target] = shuffled[offset + other];
      shuffled[offset + other] = value;
    }
  }
  return shuffled;
}

export class Circuit {
  constructor(metadata, counts, seed = 42) {
    this.neurons = metadata.neurons;
    this.n = this.neurons.length;
    this.real = normalizeWeights(counts, this.neurons);
    this.shuffled = shuffleWeights(this.real, this.n, seed);
    this.rates = new Float64Array(this.n);
    this.inputs = new Float64Array(this.n);
    this.epg = this.neurons.map((n, i) => (n.type === 'EPG' ? i : -1)).filter((i) => i >= 0);
    this.mode = 'real';
    this.velocity = 0;
    this.setMode('real');
    this.reset();
  }
  setMode(mode) {
    if (!['real', 'shuffled'].includes(mode)) throw new Error('Unknown wiring mode');
    this.mode = mode;
    const weights = mode === 'real' ? this.real : this.shuffled;
    this.edges = [];
    for (let source = 0; source < this.n; source++) {
      for (let target = 0; target < this.n; target++) {
        const weight = weights[source * this.n + target];
        if (weight) this.edges.push({ source, target, weight });
      }
    }
    // Activity is deliberately retained when the connectivity changes.
  }
  reshuffle(seed) {
    this.shuffled = shuffleWeights(this.real, this.n, seed);
    this.setMode(this.mode);
  }
  reset(phase = 1) {
    this.cueAngle = wrap(phase);
    this.elapsed = 0;
    for (let i = 0; i < this.n; i++) {
      this.rates[i] = 0.8 * Math.exp(5 * (Math.cos(this.neurons[i].angle - phase) - 1));
    }
  }
  step(dt = STEP) {
    if (!(dt > 0 && dt <= 1 / 30)) throw new Error('Use fixed, bounded integration steps');
    this.elapsed += dt;
    this.cueAngle = wrap(this.cueAngle + ((this.velocity * Math.PI) / 180) * dt);
    for (let i = 0; i < this.n; i++) this.inputs[i] = this.neurons[i].type === 'EPG' ? 0.025 : 0.02;
    for (const { source, target, weight } of this.edges) this.inputs[target] += weight * this.rates[source];
    // A simplified moving sensory cue. No directional input is applied at rest.
    if (this.velocity !== 0) {
      for (const i of this.epg) {
        this.inputs[i] += 2.8 * Math.exp(6 * (Math.cos(this.neurons[i].angle - this.cueAngle) - 1)) - 1.3;
      }
    }
    // Synchronous Euler update, rectified saturating rate, tau = 150 ms.
    for (let i = 0; i < this.n; i++) {
      const target = Math.max(0, Math.min(1, this.inputs[i]));
      this.rates[i] += (dt / 0.15) * (target - this.rates[i]);
    }
  }
  readout() {
    let x = 0,
      y = 0,
      total = 0,
      peak = 0;
    for (const i of this.epg) {
      const rate = this.rates[i];
      total += rate;
      peak = Math.max(peak, rate);
      x += rate * Math.cos(this.neurons[i].angle);
      y += rate * Math.sin(this.neurons[i].angle);
    }
    return { angle: wrap(Math.atan2(y, x)), coherence: total > 1e-6 ? Math.hypot(x, y) / total : 0, peak, total };
  }
}
