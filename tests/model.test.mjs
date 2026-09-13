import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Circuit, decodeMatrix, shuffleWeights, STEP } from '../public/model.mjs';

const metadata = JSON.parse(readFileSync(new URL('../public/data/neurons.json', import.meta.url)));
const bytes = readFileSync(new URL('../public/data/connections.bin', import.meta.url));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const counts = decodeMatrix(buffer, metadata);
const simulate = (c, seconds) => { for (let i = 0; i < Math.round(seconds / STEP); i++) c.step(); };
const delta = (a,b) => Math.atan2(Math.sin(a-b),Math.cos(a-b));

test('shipped measured data match their hash, counts, and string root IDs', () => {
  assert.equal(createHash('sha256').update(bytes).digest('hex'), metadata.matrix.sha256);
  assert.equal(metadata.neuronCount,147);
  assert.equal(metadata.edgeCount,2529);
  assert.equal(metadata.synapseCount,35781);
  assert.equal(new Set(metadata.neurons.map(n=>n.id)).size,metadata.neuronCount);
  for(const n of metadata.neurons){assert.match(n.id,/^\d{18}$/);assert.ok(Number.isFinite(n.angle));}
});

test('invalid binary, nonfinite weights, and inconsistent totals fail visibly', () => {
  assert.throws(()=>decodeMatrix(buffer.slice(0,-4),metadata),/do not match/);
  const corrupt=buffer.slice(0);new DataView(corrupt).setFloat32(0,NaN,true);
  assert.throws(()=>decodeMatrix(corrupt,metadata),/Invalid/);
  assert.throws(()=>decodeMatrix(buffer,{...metadata,edgeCount:1}),/totals/);
});

test('shuffle preserves every source row weight multiset and exact edge count', () => {
  for(const seed of [1,42,100]){
    const result=shuffleWeights(counts,metadata.neuronCount,seed);
    assert.notDeepEqual(result,counts);
    assert.deepEqual(result,shuffleWeights(counts,metadata.neuronCount,seed));
    for(let source=0;source<metadata.neuronCount;source++){
      const from=source*metadata.neuronCount,to=from+metadata.neuronCount;
      assert.deepEqual(result.slice(from,to).sort(),counts.slice(from,to).sort());
    }
  }
});

test('real wiring holds a localized stationary patch with no continuing cue', () => {
  for(const phase of [0,1,2,3,4,5]){
    const c=new Circuit(metadata,counts);c.reset(phase);simulate(c,20);
    const first=c.readout();simulate(c,10);const last=c.readout();
    assert.ok(last.coherence>.6,`phase ${phase}: ${last.coherence}`);
    assert.ok(last.peak>.7);
    assert.ok(Math.abs(delta(last.angle,first.angle))<.06,'settled heading is stable');
    assert.equal(c.velocity,0);
    assert.ok(c.rates.every(r=>Number.isFinite(r)&&r>=0&&r<=1));
  }
});

test('default and independently seeded shuffled graphs lose spatial coherence', () => {
  const scores=[];
  for(let seed=1;seed<=20;seed++){
    const c=new Circuit(metadata,counts,seed);simulate(c,8);
    const before=c.rates.slice();c.setMode('shuffled');assert.deepEqual(c.rates,before);
    simulate(c,12);scores.push(c.readout().coherence);
    assert.equal(c.edges.length,metadata.edgeCount);
    assert.ok(c.rates.every(r=>Number.isFinite(r)&&r>=0&&r<=1));
  }
  scores.sort((a,b)=>a-b);
  assert.ok(scores[10]<.3,`median shuffled coherence ${scores[10]}`);
  const c=new Circuit(metadata,counts,42);simulate(c,8);c.setMode('shuffled');simulate(c,12);
  assert.ok(c.readout().coherence<.25);
});

test('left and right cues move the patch around a complete ring, then it persists', () => {
  for(const velocity of [-45,45]){
    const c=new Circuit(metadata,counts);simulate(c,10);
    c.cueAngle=c.readout().angle;c.velocity=velocity;
    let previous=c.readout().angle,travel=0;
    for(let second=0;second<8;second++){
      simulate(c,1);const r=c.readout();travel+=delta(r.angle,previous);previous=r.angle;
      assert.ok(r.coherence>.7);
    }
    assert.ok(travel*Math.sign(velocity)>5.5,`turn travel ${travel}`);
    c.velocity=0;simulate(c,10);assert.ok(c.readout().coherence>.6);
  }
});

test('fixed timestep rejects unstable caller inputs', () => {
  const c=new Circuit(metadata,counts);
  for(const dt of [0,-1,1,NaN]) assert.throws(()=>c.step(dt),/bounded/);
});
