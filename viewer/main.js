// MonoWorld viewer: Three.js + custom Gaussian Splat renderer.
// Three render modes: Mesh (.glb), Splats (.ply 3DGS), Points (.ply).
// The splat renderer is a custom shader — no external libraries needed.

import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';
import { parseSplatPLY, createSplatPoints } from './splat-renderer.js';

// --- Scene setup -----------------------------------------------------------

const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0e1116, 1);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  60, window.innerWidth / window.innerHeight, 0.01, 1000,
);

const axes = new THREE.AxesHelper(0.5);
axes.visible = false;
scene.add(axes);

scene.add(new THREE.AmbientLight(0xffffff, 1.0));

// --- Controls --------------------------------------------------------------

const controls = new PointerLockControls(camera, renderer.domElement);
scene.add(controls.getObject());

const lockOverlay = document.getElementById('lock-overlay');
lockOverlay.addEventListener('click', () => controls.lock());
controls.addEventListener('lock',   () => lockOverlay.classList.add('hidden'));
controls.addEventListener('unlock', () => lockOverlay.classList.remove('hidden'));

const move = { fwd: false, back: false, left: false, right: false, up: false, boost: false };
const BASE_SPEED = 1.5;
const BOOST_MULT = 3.0;

window.addEventListener('keydown', (e) => {
  switch (e.code) {
    case 'KeyW': case 'ArrowUp':    move.fwd = true; break;
    case 'KeyS': case 'ArrowDown':  move.back = true; break;
    case 'KeyA': case 'ArrowLeft':  move.left = true; break;
    case 'KeyD': case 'ArrowRight': move.right = true; break;
    case 'Space':                   move.up = true; e.preventDefault(); break;
    case 'ShiftLeft': case 'ShiftRight': move.boost = true; break;
    case 'KeyR': resetCamera(); break;
  }
});
window.addEventListener('keyup', (e) => {
  switch (e.code) {
    case 'KeyW': case 'ArrowUp':    move.fwd = false; break;
    case 'KeyS': case 'ArrowDown':  move.back = false; break;
    case 'KeyA': case 'ArrowLeft':  move.left = false; break;
    case 'KeyD': case 'ArrowRight': move.right = false; break;
    case 'Space':                   move.up = false; break;
    case 'ShiftLeft': case 'ShiftRight': move.boost = false; break;
  }
});

// --- State -----------------------------------------------------------------

const plyLoader = new PLYLoader();
const gltfLoader = new GLTFLoader();

let currentObject = null;
let currentSceneBounds = null;
let currentScene = null;
let currentMode = 'mesh';
let layerMeshes = [];
let sceneList = [];

const loadingEl = document.getElementById('loading');
const sceneSelect = document.getElementById('scene-select');
const primitiveCountEl = document.getElementById('primitive-count');
const pointSizeRow = document.getElementById('point-size-row');
const pointSizeInput = document.getElementById('point-size');
const pointSizeVal = document.getElementById('point-size-val');
const wireframeToggle = document.getElementById('wireframe');
const doubleSideToggle = document.getElementById('double-side');
const modeMeshBtn = document.getElementById('mode-mesh');
const modeSplatsBtn = document.getElementById('mode-splats');
const modePointsBtn = document.getElementById('mode-points');
const layerTogglesEl = document.getElementById('layer-toggles');
const layerCheckboxesEl = document.getElementById('layer-checkboxes');

const LAYER_COLORS = ['#5050ff', '#ffdc50', '#78ff50', '#ff50b4', '#50c8ff', '#b4b4b4'];

function showLoading(msg = 'Loading scene…') { loadingEl.textContent = msg; loadingEl.classList.add('visible'); }
function hideLoading() { loadingEl.classList.remove('visible'); }

function disposeCurrent() {
  if (!currentObject) return;
  scene.remove(currentObject);
  currentObject.traverse?.((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      mats.forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
    }
  });
  if (currentObject.geometry) currentObject.geometry.dispose();
  if (currentObject.material) {
    if (currentObject.material.map) currentObject.material.map.dispose();
    currentObject.material.dispose();
  }
  currentObject = null;
  layerMeshes = [];
  layerCheckboxesEl.innerHTML = '';
  layerTogglesEl.style.display = 'none';
}

async function fetchSceneIndex() {
  try {
    const res = await fetch('/scenes.json', { cache: 'no-store' });
    if (!res.ok) return [];
    const data = await res.json();
    return data.scenes || [];
  } catch (err) {
    console.warn('No scenes.json found.', err);
    return [];
  }
}

// --- Loaders ---------------------------------------------------------------

function loadGLB(url) {
  showLoading('Loading mesh…');
  gltfLoader.load(
    url,
    (gltf) => {
      disposeCurrent();
      const root = gltf.scene;

      let triCount = 0;
      const foundLayerMeshes = [];
      root.traverse((child) => {
        if (child.isMesh) {
          const oldMat = child.material;
          const newMat = new THREE.MeshBasicMaterial({
            map: oldMat.map || null,
            side: doubleSideToggle.checked ? THREE.DoubleSide : THREE.FrontSide,
            wireframe: wireframeToggle.checked,
            color: oldMat.map ? 0xffffff : 0xcccccc,
          });
          if (newMat.map) {
            newMat.map.colorSpace = THREE.SRGBColorSpace;
            newMat.map.needsUpdate = true;
          }
          child.material = newMat;
          oldMat.dispose?.();

          if (child.geometry?.index) triCount += child.geometry.index.count / 3;
          else if (child.geometry?.attributes?.position) triCount += child.geometry.attributes.position.count / 3;

          const name = child.name || child.parent?.name || '';
          const m = name.match(/^layer_(\d+)/);
          if (m) foundLayerMeshes.push({ layerId: parseInt(m[1], 10), mesh: child });
        }
      });

      scene.add(root);
      currentObject = root;
      currentSceneBounds = new THREE.Box3().setFromObject(root);
      primitiveCountEl.textContent = `Tris: ${Math.round(triCount).toLocaleString()}`;
      pointSizeRow.style.display = 'none';

      if (foundLayerMeshes.length > 0) {
        foundLayerMeshes.sort((a, b) => a.layerId - b.layerId);
        layerMeshes = foundLayerMeshes;
        renderLayerCheckboxes();
        layerTogglesEl.style.display = '';
      }

      resetCamera();
      hideLoading();
    },
    (xhr) => { if (xhr.lengthComputable) showLoading(`Loading mesh… ${(xhr.loaded / xhr.total * 100).toFixed(0)}%`); },
    (err) => { console.error('GLB load failed:', err); showLoading('GLB load failed'); },
  );
}

async function loadSplat(url) {
  showLoading('Loading Gaussian splats…');
  disposeCurrent();

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();

    const splatData = parseSplatPLY(buffer);
    const splatPoints = createSplatPoints(splatData);

    scene.add(splatPoints);
    currentObject = splatPoints;
    currentSceneBounds = new THREE.Box3().setFromObject(splatPoints);
    primitiveCountEl.textContent = `Splats: ${splatData.count.toLocaleString()}`;
    pointSizeRow.style.display = 'none';
    layerTogglesEl.style.display = 'none';

    resetCamera();
    hideLoading();
  } catch (err) {
    console.error('Splat load failed:', err);
    showLoading(`Splat load failed: ${err.message}`);
  }
}

function renderLayerCheckboxes() {
  layerCheckboxesEl.innerHTML = '';
  for (const { layerId, mesh } of layerMeshes) {
    const wrap = document.createElement('label');
    wrap.className = 'layer-cb';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', () => { mesh.visible = cb.checked; });
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = LAYER_COLORS[layerId] || '#888';
    const label = document.createElement('span');
    const tag = layerId === 0 ? ' (near)' : layerId === layerMeshes.length - 1 ? ' (far)' : '';
    label.textContent = `L${layerId}${tag}`;
    wrap.appendChild(cb);
    wrap.appendChild(swatch);
    wrap.appendChild(label);
    layerCheckboxesEl.appendChild(wrap);
  }
}

function loadPLY(url) {
  showLoading('Loading points…');
  plyLoader.load(
    url,
    (geometry) => {
      disposeCurrent();
      geometry.computeBoundingBox();
      const hasColor = !!geometry.getAttribute('color');
      const material = new THREE.PointsMaterial({
        size: parseFloat(pointSizeInput.value),
        vertexColors: hasColor,
        sizeAttenuation: true,
        color: hasColor ? 0xffffff : 0xcccccc,
      });
      const points = new THREE.Points(geometry, material);
      scene.add(points);
      currentObject = points;
      currentSceneBounds = new THREE.Box3().setFromObject(points);
      const n = geometry.getAttribute('position').count;
      primitiveCountEl.textContent = `Points: ${n.toLocaleString()}`;
      pointSizeRow.style.display = '';
      resetCamera();
      hideLoading();
    },
    (xhr) => { if (xhr.lengthComputable) showLoading(`Loading points… ${(xhr.loaded / xhr.total * 100).toFixed(0)}%`); },
    (err) => { console.error('PLY load failed:', err); showLoading('PLY load failed'); },
  );
}

function loadCurrentScene() {
  if (!currentScene) return;
  if (currentMode === 'splats' && currentScene.splat) {
    loadSplat(currentScene.splat);
    setActiveModeButton('splats');
  } else if (currentMode === 'mesh' && currentScene.glb) {
    loadGLB(currentScene.glb);
    setActiveModeButton('mesh');
  } else if (currentMode === 'points' && currentScene.ply) {
    loadPLY(currentScene.ply);
    setActiveModeButton('points');
  } else if (currentScene.glb) {
    loadGLB(currentScene.glb);
    setActiveModeButton('mesh');
    currentMode = 'mesh';
  } else if (currentScene.ply) {
    loadPLY(currentScene.ply);
    setActiveModeButton('points');
    currentMode = 'points';
  } else {
    showLoading('No loadable scene files found.');
  }
}

function setActiveModeButton(mode) {
  modeMeshBtn.classList.toggle('active', mode === 'mesh');
  modeSplatsBtn.classList.toggle('active', mode === 'splats');
  modePointsBtn.classList.toggle('active', mode === 'points');
}

// --- Camera ----------------------------------------------------------------

function resetCamera() {
  if (!currentSceneBounds) return;
  const center = new THREE.Vector3();
  currentSceneBounds.getCenter(center);
  const size = new THREE.Vector3();
  currentSceneBounds.getSize(size);
  const margin = 0.3 * Math.max(size.x, size.y);
  camera.position.set(center.x, center.y, currentSceneBounds.min.z - margin);
  camera.lookAt(center);
}

// --- HUD wiring ------------------------------------------------------------

pointSizeInput.addEventListener('input', () => {
  const s = parseFloat(pointSizeInput.value);
  pointSizeVal.textContent = s.toFixed(3);
  if (currentObject?.material?.size !== undefined) currentObject.material.size = s;
});

document.getElementById('reset-cam').addEventListener('click', resetCamera);
document.getElementById('toggle-axes').addEventListener('click', () => { axes.visible = !axes.visible; });

wireframeToggle.addEventListener('change', () => {
  if (!currentObject) return;
  currentObject.traverse?.((child) => {
    if (child.isMesh) child.material.wireframe = wireframeToggle.checked;
  });
});

doubleSideToggle.addEventListener('change', () => {
  if (!currentObject) return;
  currentObject.traverse?.((child) => {
    if (child.isMesh) {
      child.material.side = doubleSideToggle.checked ? THREE.DoubleSide : THREE.FrontSide;
      child.material.needsUpdate = true;
    }
  });
});

modeMeshBtn.addEventListener('click', () => { if (currentMode !== 'mesh') { currentMode = 'mesh'; loadCurrentScene(); } });
modeSplatsBtn.addEventListener('click', () => { if (currentMode !== 'splats') { currentMode = 'splats'; loadCurrentScene(); } });
modePointsBtn.addEventListener('click', () => { if (currentMode !== 'points') { currentMode = 'points'; loadCurrentScene(); } });

sceneSelect.addEventListener('change', () => {
  const id = sceneSelect.value;
  if (!id) return;
  currentScene = sceneList.find((s) => s.id === id) || null;
  loadCurrentScene();
});

document.getElementById('reload-scenes').addEventListener('click', refreshScenes);

async function refreshScenes() {
  sceneList = await fetchSceneIndex();
  sceneSelect.innerHTML = '';
  if (sceneList.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = '(no scenes — run python run.py first)';
    opt.value = '';
    sceneSelect.appendChild(opt);
    return;
  }
  for (const s of sceneList) {
    const opt = document.createElement('option');
    opt.value = s.id;
    const tags = [s.glb ? 'mesh' : null, s.splat ? 'splat' : null, s.ply ? 'pts' : null].filter(Boolean).join('+');
    opt.textContent = `${s.id}  [${tags}]`;
    sceneSelect.appendChild(opt);
  }
  const last = sceneList[sceneList.length - 1];
  sceneSelect.value = last.id;
  currentScene = last;
  // Default to mesh (most reliable), user can switch.
  currentMode = last.glb ? 'mesh' : (last.splat ? 'splats' : 'points');
  loadCurrentScene();
}

// --- Render loop -----------------------------------------------------------

const fpsEl = document.getElementById('fps');
let fpsLastTime = performance.now();
let fpsFrames = 0;
let lastFrameTime = performance.now();

function animate(now) {
  requestAnimationFrame(animate);

  const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now;

  fpsFrames++;
  if (now - fpsLastTime > 1000) {
    fpsEl.textContent = (fpsFrames * 1000 / (now - fpsLastTime)).toFixed(0);
    fpsFrames = 0;
    fpsLastTime = now;
  }

  if (controls.isLocked) {
    const speed = BASE_SPEED * (move.boost ? BOOST_MULT : 1) * dt;
    if (move.fwd)   controls.moveForward(speed);
    if (move.back)  controls.moveForward(-speed);
    if (move.right) controls.moveRight(speed);
    if (move.left)  controls.moveRight(-speed);
    if (move.up)    controls.getObject().position.y += speed;
  }

  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

refreshScenes();
animate(performance.now());
