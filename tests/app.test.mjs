import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { webcrypto } from 'node:crypto';
import { Circuit, decodeMatrix, STEP, TAU } from '../public/model.mjs';
import { decodeAnatomy } from '../public/scene3d.mjs';

const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const source = readFileSync(new URL('../public/app.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
const meta = JSON.parse(readFileSync(new URL('../public/data/neurons.json', import.meta.url)));
const files = Object.fromEntries(
  ['neurons.json', 'connections.bin', 'anatomy.json', 'anatomy.bin'].map((name) => [
    name,
    readFileSync(new URL(`../public/data/${name}`, import.meta.url)),
  ]),
);
const bytes = files['connections.bin'];

async function boot({ webgl = true, data = true } = {}) {
  const elements = new Map();
  const context2d = { setTransform() {} };
  for (const tag of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const attrs = new Map(),
      events = new Map();
    elements.set(tag[1], {
      hidden: /\bhidden\b/.test(tag[0]),
      disabled: /\bdisabled\b/.test(tag[0]),
      style: {},
      dataset: {},
      textContent: '',
      innerHTML: '',
      setAttribute(name, value) {
        attrs.set(name, value);
      },
      getAttribute(name) {
        return attrs.get(name);
      },
      addEventListener(name, fn) {
        events.set(name, fn);
      },
      click() {
        assert.equal(this.disabled, false, 'button must be enabled');
        return events.get('click')?.();
      },
      getBoundingClientRect() {
        return this.hidden ? { width: 0, height: 0 } : { width: 800, height: 600 };
      },
      getContext() {
        return context2d;
      },
    });
  }
  let scene, observed;
  const requests = [],
    errors = [];
  class FakeScene {
    constructor() {
      if (!webgl) throw new Error('WebGL unavailable');
      scene = this;
    }
    setEdges() {}
    setView(view) {
      this.view = view;
    }
    resize(width, height) {
      this.width = width;
      this.height = height;
    }
    setAnatomy(anatomy) {
      this.anatomy = anatomy;
    }
  }
  const sandbox = {
    Circuit,
    decodeMatrix,
    STEP,
    TAU,
    Scene3D: FakeScene,
    decodeAnatomy,
    document: {
      getElementById: (id) => elements.get(id) ?? null,
      body: { classList: { toggle() {} } },
      addEventListener() {},
    },
    matchMedia: () => ({ matches: false }),
    devicePixelRatio: 1,
    crypto: webcrypto,
    ResizeObserver: class {
      constructor(fn) {
        this.callback = fn;
      }
      observe(element) {
        observed = element;
      }
    },
    requestAnimationFrame() {},
    console: { error: (e) => errors.push(e), warn() {} },
    async fetch(url) {
      requests.push(url);
      const file = files[url.replace('data/', '')];
      return {
        ok: data && Boolean(file),
        status: data && file ? 200 : 404,
        async json() {
          return JSON.parse(file);
        },
        async arrayBuffer() {
          return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
        },
      };
    },
  };
  // Run the shipped startup and event handlers with the real model/data;
  // substitute only the browser DOM, renderer and network boundaries.
  await runInNewContext(
    source.replace('requestAnimationFrame(frame);init();', 'requestAnimationFrame(frame);globalThis.ready=init();'),
    sandbox,
  );
  await sandbox.ready;
  return { elements, scene, observed, requests, errors };
}

test('startup enables all views and switches from 3D to 2D and real anatomy', async () => {
  const { elements: el, scene, observed, requests, errors } = await boot();
  assert.deepEqual(errors, []);
  assert.equal(el.get('run-status').textContent, 'SIMULATION LIVE');
  assert.equal(el.get('loading').hidden, true);
  assert.equal(observed, el.get('visualization'));
  for (const id of ['view-ring', 'view-flat', 'view-anatomy']) assert.equal(el.get(id).disabled, false);
  el.get('view-flat').click();
  assert.equal(el.get('circuit').hidden, false);
  assert.equal(el.get('scene3d').hidden, true);
  assert.equal(el.get('view-flat').getAttribute('aria-pressed'), 'true');
  el.get('view-ring').click();
  assert.equal(scene.view, 'ring');
  assert.equal(scene.width, 800);
  el.get('view-anatomy').click();
  for (let i = 0; i < 200 && !scene.anatomy && !errors.length; i++)
    await new Promise((resolve) => setTimeout(resolve, 25));
  assert.deepEqual(errors, []);
  assert.equal(scene.view, 'anatomy');
  assert.equal(scene.anatomy.cells.length, meta.neuronCount);
  assert.equal(el.get('anatomy-filter').hidden, false);
  assert.equal(el.get('anatomy-status').hidden, true);
  assert.deepEqual(requests, ['data/neurons.json', 'data/connections.bin', 'data/anatomy.json', 'data/anatomy.bin']);
});

test('WebGL failure leaves a live 2D compass and disables only 3D views', async () => {
  const { elements: el, errors } = await boot({ webgl: false });
  assert.deepEqual(errors, []);
  assert.equal(el.get('run-status').textContent, 'SIMULATION LIVE');
  assert.equal(el.get('view-flat').disabled, false);
  assert.equal(el.get('circuit').hidden, false);
  for (const id of ['view-ring', 'view-anatomy']) assert.equal(el.get(id).disabled, true);
  el.get('view-flat').click();
});

test('missing connectome files show an explicit data error', async () => {
  const { elements: el, errors } = await boot({ data: false });
  assert.equal(errors.length, 1);
  assert.equal(el.get('run-status').textContent, 'DATA UNAVAILABLE');
  assert.equal(el.get('loading').hidden, false);
  assert.match(el.get('loading').textContent, /both data files/);
});
