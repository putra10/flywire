import { Circuit, decodeMatrix, STEP, TAU } from './model.mjs';
import { Scene3D, anatomyFromSkeletonBuffers, decodeAnatomy } from './scene3d.mjs';

const $ = id => document.getElementById(id);
const canvas = $('circuit'), ctx = canvas.getContext('2d');
const ui = Object.fromEntries(['turn','velocity','stop','real','shuffled','pause','reset','reshuffle','loading','heading','heading-note','coherence','coherence-bar','run-status','mode-label','mode-description','view-caption','data-stats','fly','visualization','scene3d','anatomy-filter','anatomy-status','orbit-tools','orbit-hint','neuron-inspector'].map(id => [id, $(id)]));
let circuit, metadata, paused = false, last = 0, accumulator = 0, width = 0, height = 0, seed = 42;
let positions = [], displayEdges = [], hudClock = 0, scene, view = 'ring', anatomyLoading = false;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

function resize() {
  const activeCanvas = view === 'flat' ? canvas : ui.scene3d;
  const rect = activeCanvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
  width = rect.width; height = rect.height;
  if (view === 'flat') {
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  if (circuit) layout();
  if (scene) scene.resize(width,height);
}

function layout() {
  const radius = Math.min(width * 0.355, height * 0.345);
  const lanes = { EPG: 1, PENa: 0.865, PENb: 0.825, PEG: 0.785, Delta7: 0.665 };
  // Small radial offsets make coincident cells visible without changing angle.
  positions = circuit.neurons.map((neuron, i) => {
    const r = radius * lanes[neuron.type] + ((i % 3) - 1) * 3;
    const a = neuron.angle - Math.PI / 2;
    return { x: width / 2 + Math.cos(a) * r, y: height * 0.49 + Math.sin(a) * r, r, a };
  });
  canvas.dataset.radius = radius;
  selectEdges();
}

function selectEdges() {
  // Drawing uses a deterministic sample; the simulation always uses every edge.
  displayEdges = circuit.edges.filter((edge, i) => i % 13 === 0);
}

function draw() {
  if (!circuit) return;
  if (scene && view !== 'flat') scene.draw(circuit.rates);
  if (view !== 'flat') return;
  ctx.clearRect(0, 0, width, height);
  const cx = width / 2, cy = height * 0.49, radius = Number(canvas.dataset.radius);
  const warm = circuit.mode === 'shuffled';
  const color = warm ? [223,179,131] : [196,232,132];
  const rgba = alpha => `rgba(${color.join(',')},${alpha})`;
  const readout = circuit.readout();
  // Quiet instrument markings.
  for (const factor of [0.665, 0.825, 1, 1.14]) {
    ctx.beginPath(); ctx.arc(cx, cy, radius * factor, 0, TAU);
    ctx.strokeStyle = rgba(factor === 1.14 ? .06 : .10); ctx.lineWidth = 1; ctx.stroke();
  }
  for (let tick = 0; tick < 72; tick++) {
    const a = tick / 72 * TAU, major = tick % 18 === 0;
    const r1 = radius * 1.105, r2 = r1 + (major ? 8 : tick % 6 === 0 ? 5 : 2);
    ctx.beginPath(); ctx.moveTo(cx + Math.sin(a) * r1, cy - Math.cos(a) * r1);
    ctx.lineTo(cx + Math.sin(a) * r2, cy - Math.cos(a) * r2);
    ctx.strokeStyle = rgba(major ? .4 : .17); ctx.stroke();
  }
  ctx.font = '8px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = rgba(.4);
  [[0,'0°'],[90,'90°'],[180,'180°'],[270,'270°']].forEach(([degrees,label]) => {
    const a = degrees * Math.PI / 180;
    ctx.fillText(label,cx + Math.sin(a) * (radius * 1.14 + 18),cy - Math.cos(a) * (radius * 1.14 + 18));
  });
  // Synaptic paths; brightness depends on presynaptic simulated activity.
  for (const edge of displayEdges) {
    const p = positions[edge.source], q = positions[edge.target];
    const activity = circuit.rates[edge.source];
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    // Pull connections through the ring, keeping the middle legible.
    ctx.quadraticCurveTo((p.x + q.x) / 2 * .65 + cx * .35, (p.y + q.y) / 2 * .65 + cy * .35, q.x, q.y);
    ctx.strokeStyle = edge.weight < 0 ? `rgba(207,161,115,${.012 + activity * .032})` : rgba(.012 + activity * .055);
    ctx.lineWidth = .65; ctx.stroke();
  }
  // Dark central aperture provides a readable instrument face.
  const aperture = ctx.createRadialGradient(cx,cy,radius*.35,cx,cy,radius*.61);
  aperture.addColorStop(0,warm ? '#201e17' : '#1c2218'); aperture.addColorStop(.8,warm ? 'rgba(32,30,23,.95)' : 'rgba(28,34,24,.95)'); aperture.addColorStop(1,'rgba(25,30,20,0)');
  ctx.fillStyle=aperture;ctx.beginPath();ctx.arc(cx,cy,radius*.61,0,TAU);ctx.fill();
  // Individual neurons, never a painted or scripted bump.
  for (let i = 0; i < circuit.n; i++) {
    const p = positions[i], rate = circuit.rates[i], type = circuit.neurons[i].type;
    const rgb = type === 'Delta7' ? [216,169,130] : type === 'EPG' ? color : [142,166,112];
    const size = type === 'EPG' ? 3.3 : type === 'Delta7' ? 2 : 2.35;
    if (rate > .08) {
      const glow = ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,size*6);
      glow.addColorStop(0,`rgba(${rgb},${rate*.38})`); glow.addColorStop(1,`rgba(${rgb},0)`);
      ctx.fillStyle=glow;ctx.beginPath();ctx.arc(p.x,p.y,size*6,0,TAU);ctx.fill();
    }
    ctx.beginPath();ctx.arc(p.x,p.y,size + rate*.75,0,TAU);
    ctx.fillStyle=`rgba(${rgb},${.16+rate*.84})`;ctx.fill();
    if (rate > .7) {ctx.beginPath();ctx.arc(p.x,p.y,1.2,0,TAU);ctx.fillStyle=`rgba(244,255,222,${rate*.8})`;ctx.fill();}
  }
  // The readout pointer fades naturally as heading coherence is lost.
  const a = readout.angle - Math.PI / 2, r = radius * 1.07;
  ctx.save();ctx.translate(cx+Math.cos(a)*r,cy+Math.sin(a)*r);ctx.rotate(a);
  ctx.beginPath();ctx.moveTo(-4,-3);ctx.lineTo(3,0);ctx.lineTo(-4,3);ctx.closePath();
  ctx.fillStyle=rgba(readout.coherence*.8);ctx.fill();ctx.restore();
}

function updateHUD() {
  const r = circuit.readout(), coherent = r.coherence > .25 && r.peak > .1;
  ui.heading.innerHTML = coherent ? `${Math.round(r.angle*180/Math.PI)%360}<small>°</small>` : '—<small>°</small>';
  ui['heading-note'].textContent = coherent ? (circuit.velocity ? 'Following the turn' : 'Holding a heading') : 'No coherent heading';
  ui.coherence.textContent = `${Math.round(r.coherence*100)}%`;
  ui['coherence-bar'].style.width = `${r.coherence*100}%`;
  ui.fly.style.transform = `rotate(${Math.sin(circuit.cueAngle)*8+circuit.velocity*.24}deg)`;
}

function frame(now) {
  if (!last) last=now;
  const dt = Math.min((now-last)/1000,.05);last=now;
  if (circuit && !paused && !document.hidden) {
    accumulator += dt;
    while (accumulator >= STEP) { circuit.step(STEP); accumulator-=STEP; }
  }
  draw();hudClock+=dt;
  if(circuit && hudClock>.1){updateHUD();hudClock=0;}
  requestAnimationFrame(frame);
}

function setPaused(value) {
  paused=value;
  ui.pause.innerHTML=paused?'▶ <span>Resume</span>':'Ⅱ <span>Pause</span>';
  ui.pause.setAttribute('aria-label',paused?'Resume simulation':'Pause simulation');
  ui['run-status'].textContent=paused?'SIMULATION PAUSED':'SIMULATION LIVE';
  $('live-dot').style.opacity=paused?'.3':'1';
}
function turn(value) {
  if (!circuit) return;
  if (circuit.velocity === 0 && value !== 0) circuit.cueAngle=circuit.readout().angle;
  circuit.velocity=value;ui.turn.value=value;
  ui.velocity.textContent=`${value<0?'−':value>0?'+':''}${Math.abs(value)}° / s`;
  ui.turn.setAttribute('aria-valuetext',value===0?'Still':`${Math.abs(value)} degrees per second ${value<0?'left':'right'}`);
}
function mode(value) {
  circuit.setMode(value);selectEdges();
  if (scene) scene.setEdges(circuit.edges);
  const shuffled=value==='shuffled';
  document.body.classList.toggle('shuffled-mode',shuffled);
  ui.real.setAttribute('aria-pressed',String(!shuffled));ui.shuffled.setAttribute('aria-pressed',String(shuffled));
  ui['mode-label'].textContent=shuffled?'PERMUTED DESTINATIONS':'MEASURED WIRING';
  ui['mode-description'].textContent=shuffled?'The same connections, sent to random destinations. Watch the patch lose its shape.':'The measured connections let activity gather into a persistent patch.';
  ui.reshuffle.hidden=!shuffled;
}

function showInspector(index) {
  if(index<0){ui['neuron-inspector'].hidden=true;return;}
  const n=circuit.neurons[index];ui['neuron-inspector'].innerHTML=`<strong>${n.type}</strong> <span>${n.nt} · ${Math.round(circuit.rates[index]*100)}% active · ${n.id}</span>`;ui['neuron-inspector'].hidden=false;
}

async function loadAnatomy() {
  if(scene.anatomy||anatomyLoading)return;
  anatomyLoading=true;ui['anatomy-status'].hidden=false;ui['anatomy-status'].textContent='Fetching the published v783 skeletons…';
  try {
    const buffers=await Promise.all(circuit.neurons.map(async n=>{const response=await fetch(`https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/${n.id}`);if(!response.ok)throw new Error(`FlyWire returned ${response.status}.`);return response.arrayBuffer();}));
    const anatomy=anatomyFromSkeletonBuffers(circuit.neurons,buffers);scene.setAnatomy(anatomy);ui['anatomy-status'].hidden=true;
  } catch(error) {
    ui['anatomy-status'].innerHTML=`The published skeleton service could not be reached. <button id="retry-anatomy">Try again</button>`;$('retry-anatomy').addEventListener('click',loadAnatomy);
  } finally { anatomyLoading=false; }
}

function setView(next) {
  view=next;ui.visualization.setAttribute('data-view',next);
  $('visualization').dataset.view=next;
  ui['view-ring'].setAttribute('aria-pressed',String(next==='ring'));ui['view-anatomy'].setAttribute('aria-pressed',String(next==='anatomy'));ui['view-flat'].setAttribute('aria-pressed',String(next==='flat'));
  canvas.hidden=next!=='flat';$('scene3d').hidden=next==='flat';ui['anatomy-filter'].hidden=next!=='anatomy';ui['orbit-tools'].hidden=next==='flat';ui['orbit-hint'].hidden=next==='flat';
  ui['view-caption'].textContent=next==='anatomy'?'PUBLISHED SKELETONS · v783':next==='flat'?'2D COMPASS · 147 NEURONS':'SCHEMATIC 3D · 147 NEURONS';
  if(scene)scene.setView(next==='flat'?'ring':next);if(next==='anatomy')loadAnatomy();
  if(next==='flat'){resize();}else{scene?.resize(width,height);}
}

async function init() {
  try {
    const responses=await Promise.all([fetch('data/neurons.json'),fetch('data/connections.bin')]);
    if(responses.some(r=>!r.ok))throw new Error('Could not load the connectome. Check that both data files are deployed.');
    const [meta,buffer]=await Promise.all([responses[0].json(),responses[1].arrayBuffer()]);
    metadata=meta;
    if(crypto.subtle){
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
      if(hash!==meta.matrix.sha256)throw new Error('Data integrity check failed. Re-export both connectome files.');
    }
    circuit=new Circuit(meta,decodeMatrix(buffer,meta));
    try {
      scene=new Scene3D($('scene3d'),circuit.neurons,showInspector,restored=>{if(!restored)ui['run-status'].textContent='3D CONTEXT LOST · SWITCH TO 2D';});
      scene.setEdges(circuit.edges);
      resize();
    } catch(error) {
      console.warn(error);
      scene=null;
      ui['view-ring'].disabled=true;ui['view-anatomy'].disabled=true;ui['view-flat'].disabled=false;
      ui['view-ring'].setAttribute('aria-label','3D unavailable in this browser');
    }
    // Let the initial condition settle before showing the instrument.
    for(let i=0;i<360;i++)circuit.step();
    layout();ui.loading.hidden=true;
    for(const id of ['turn','stop','real','shuffled','pause','reset','view-ring','view-anatomy','view-flat'])ui[id].disabled=false;
    ui['data-stats'].textContent=`${meta.neuronCount} NEURONS · ${meta.edgeCount.toLocaleString()} CONNECTIONS · v${meta.version}`;
    ui.turn.addEventListener('input',()=>turn(Number(ui.turn.value)));
    ui.stop.addEventListener('click',()=>turn(0));
    ui.real.addEventListener('click',()=>mode('real'));ui.shuffled.addEventListener('click',()=>mode('shuffled'));
    ui.pause.addEventListener('click',()=>setPaused(!paused));
    ui.reset.addEventListener('click',()=>{turn(0);circuit.reset();updateHUD();});
    ui.reshuffle.addEventListener('click',()=>{circuit.reshuffle(++seed);selectEdges();});
    $('view-ring').addEventListener('click',()=>setView('ring'));$('view-flat').addEventListener('click',()=>setView('flat'));$('view-anatomy').addEventListener('click',()=>setView('anatomy'));
    $('zoom-in').addEventListener('click',()=>scene.zoom(.88));$('zoom-out').addEventListener('click',()=>scene.zoom(1.14));$('camera-reset').addEventListener('click',()=>scene.resetCamera());
    $('cell-filter').addEventListener('change',event=>scene.setFilter(event.target.value));
    document.addEventListener('keydown',event=>{if(event.key==='Escape')turn(0);});
    document.addEventListener('visibilitychange',()=>{last=0;accumulator=0;});
    setView(scene?'ring':'flat');setPaused(reducedMotion);updateHUD();
  } catch(error) {
    console.error(error);
    ui.loading.textContent=`${error.message} Reload this page to retry.`;
    ui.loading.setAttribute('role','alert');ui['run-status'].textContent='DATA UNAVAILABLE';
  }
}
new ResizeObserver(resize).observe(canvas);
resize();requestAnimationFrame(frame);init();
