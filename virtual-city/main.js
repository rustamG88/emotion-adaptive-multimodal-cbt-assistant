import * as THREE from 'https://unpkg.com/three@0.157.0/build/three.module.js';
import { PointerLockControls } from 'https://unpkg.com/three@0.157.0/examples/jsm/controls/PointerLockControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87b6e0);
scene.fog = new THREE.Fog(0x87b6e0, 200, 1400);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Lighting
const hemiLight = new THREE.HemisphereLight(0xbcd7ff, 0x4a6b3a, 0.6);
scene.add(hemiLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(100, 200, 100);
dirLight.castShadow = false;
scene.add(dirLight);

// Ground
const groundSize = 3000;
const groundGeometry = new THREE.PlaneGeometry(groundSize, groundSize);
const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x6ea76e });
const ground = new THREE.Mesh(groundGeometry, groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Generate city grid
const cellSize = 24; // meters per block
const halfBlocks = 28; // city radius in blocks
const roadEvery = 5; // every Nth row/col is a road

const roadMaterial = new THREE.MeshLambertMaterial({ color: 0x2b2b2b });
const pavementMaterial = new THREE.MeshLambertMaterial({ color: 0x6a6a6a });

const buildingMaterials = [
  new THREE.MeshLambertMaterial({ color: 0xbfd7ff }),
  new THREE.MeshLambertMaterial({ color: 0xa7c8ff }),
  new THREE.MeshLambertMaterial({ color: 0x9bd0c9 }),
  new THREE.MeshLambertMaterial({ color: 0xd8c8a7 }),
  new THREE.MeshLambertMaterial({ color: 0xdcb8b8 }),
];

const roadGeo = new THREE.BoxGeometry(cellSize, 0.1, cellSize);
const paveGeo = new THREE.BoxGeometry(cellSize, 0.2, cellSize);

const rng = mulberry32(1337);

for (let ix = -halfBlocks; ix <= halfBlocks; ix++) {
  for (let iz = -halfBlocks; iz <= halfBlocks; iz++) {
    const isRoad = ix % roadEvery === 0 || iz % roadEvery === 0;
    const x = ix * cellSize;
    const z = iz * cellSize;

    if (isRoad) {
      const road = new THREE.Mesh(roadGeo, roadMaterial);
      road.position.set(x, 0.05, z);
      scene.add(road);

      // Add dashed line for main roads
      if (ix % (roadEvery * 2) === 0 || iz % (roadEvery * 2) === 0) {
        addLaneMarkings(x, z, ix, iz);
      }
      continue;
    }

    // Pavement base under buildings
    const pave = new THREE.Mesh(paveGeo, pavementMaterial);
    pave.position.set(x, 0.1, z);
    scene.add(pave);

    // Decide lot usage: building or small park
    const parkChance = 0.12;
    if (rng() < parkChance) {
      addPark(x, z);
    } else {
      // Building
      const floors = 2 + Math.floor(rng() * 18); // 2..19 floors
      const heightPerFloor = 3.2;
      const height = floors * heightPerFloor;
      const footprint = cellSize * (0.5 + rng() * 0.35);
      const bGeo = new THREE.BoxGeometry(footprint, height, footprint);
      const mat = buildingMaterials[Math.floor(rng() * buildingMaterials.length)];
      const building = new THREE.Mesh(bGeo, mat);
      building.position.set(x, height / 2, z);
      // slight random offset within lot, but keep inside pavement
      const maxOffset = (cellSize - footprint) / 2 - 1.0;
      building.position.x += (rng() * 2 - 1) * Math.max(0, maxOffset);
      building.position.z += (rng() * 2 - 1) * Math.max(0, maxOffset);
      scene.add(building);

      // roof features
      if (rng() < 0.4) {
        const roofH = 1 + rng() * 2.5;
        const roofGeo = new THREE.BoxGeometry(footprint * (0.3 + rng() * 0.5), roofH, footprint * (0.3 + rng() * 0.5));
        const roof = new THREE.Mesh(roofGeo, mat.clone());
        roof.material.color.offsetHSL(0, -0.1, -0.1);
        roof.position.set(building.position.x, height + roofH / 2, building.position.z);
        scene.add(roof);
      }

      // windows as simple emissive boxes
      if (rng() < 0.65) {
        const windows = addWindows(building, footprint, height);
        windows.forEach(w => scene.add(w));
      }
    }
  }
}

// Add some scattered trees along pavements
addStreetTrees();

// Player controls
const controls = new PointerLockControls(camera, document.body);
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('start');
startBtn.addEventListener('click', () => {
  controls.lock();
});
controls.addEventListener('lock', () => {
  overlay.style.display = 'none';
});
controls.addEventListener('unlock', () => {
  overlay.style.display = '';
});

camera.position.set(0, 1.8, cellSize * 2);

const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false;
let canJump = false;
let isSprinting = false;
let prevTime = performance.now();

const onKeyDown = (event) => {
  switch (event.code) {
    case 'ArrowUp':
    case 'KeyW': moveForward = true; break;
    case 'ArrowLeft':
    case 'KeyA': moveLeft = true; break;
    case 'ArrowDown':
    case 'KeyS': moveBackward = true; break;
    case 'ArrowRight':
    case 'KeyD': moveRight = true; break;
    case 'Space': if (canJump) { velocity.y += 6.5; canJump = false; } break;
    case 'ShiftLeft':
    case 'ShiftRight': isSprinting = true; break;
  }
};
const onKeyUp = (event) => {
  switch (event.code) {
    case 'ArrowUp':
    case 'KeyW': moveForward = false; break;
    case 'ArrowLeft':
    case 'KeyA': moveLeft = false; break;
    case 'ArrowDown':
    case 'KeyS': moveBackward = false; break;
    case 'ArrowRight':
    case 'KeyD': moveRight = false; break;
    case 'ShiftLeft':
    case 'ShiftRight': isSprinting = false; break;
  }
};
document.addEventListener('keydown', onKeyDown);
document.addEventListener('keyup', onKeyUp);

// Simple bounds to keep player in city area
const cityExtent = cellSize * (halfBlocks + 2);

function animate() {
  requestAnimationFrame(animate);

  const time = performance.now();
  const delta = Math.min(0.05, (time - prevTime) / 1000);
  prevTime = time;

  const walkingSpeed = isSprinting ? 8.0 : 4.0; // m/s
  const damping = 8.0;

  velocity.x -= velocity.x * damping * delta;
  velocity.z -= velocity.z * damping * delta;
  velocity.y -= 9.8 * 2.5 * delta; // gravity

  direction.z = Number(moveForward) - Number(moveBackward);
  direction.x = Number(moveRight) - Number(moveLeft);
  direction.normalize();

  if (controls.isLocked === true) {
    if (moveForward || moveBackward) velocity.z -= direction.z * walkingSpeed * delta;
    if (moveLeft || moveRight) velocity.x -= direction.x * walkingSpeed * delta;

    controls.moveRight(-velocity.x * delta);
    controls.moveForward(-velocity.z * delta);

    // Vertical
    camera.position.y += velocity.y * delta;
    if (camera.position.y < 1.8) {
      velocity.y = 0;
      camera.position.y = 1.8;
      canJump = true;
    }

    // Keep within bounds
    camera.position.x = THREE.MathUtils.clamp(camera.position.x, -cityExtent, cityExtent);
    camera.position.z = THREE.MathUtils.clamp(camera.position.z, -cityExtent, cityExtent);
  }

  renderer.render(scene, camera);
}
animate();

// Helpers
function addLaneMarkings(x, z, ix, iz) {
  const isRow = iz % (roadEvery * 2) === 0;
  const count = 6;
  for (let i = -count; i <= count; i++) {
    const boxGeo = new THREE.BoxGeometry(2, 0.2, 0.6);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffaa });
    const m = new THREE.Mesh(boxGeo, mat);
    if (isRow) {
      m.position.set(x, 0.21, z + i * (cellSize / count));
      m.rotation.y = 0;
    } else {
      m.position.set(x + i * (cellSize / count), 0.21, z);
      m.rotation.y = Math.PI / 2;
    }
    scene.add(m);
  }
}

function addPark(x, z) {
  // grass patch slightly lower
  const size = cellSize * 0.9;
  const geo = new THREE.BoxGeometry(size, 0.05, size);
  const mat = new THREE.MeshLambertMaterial({ color: 0x76b576 });
  const patch = new THREE.Mesh(geo, mat);
  patch.position.set(x, 0.03, z);
  scene.add(patch);

  // few trees
  const trees = 2 + Math.floor(rng() * 5);
  for (let i = 0; i < trees; i++) {
    const ox = x + (rng() * 2 - 1) * size * 0.35;
    const oz = z + (rng() * 2 - 1) * size * 0.35;
    addTree(ox, oz, 2.5 + rng() * 4);
  }
}

function addTree(x, z, height) {
  const trunkGeo = new THREE.CylinderGeometry(0.2, 0.3, height * 0.5, 6);
  const trunkMat = new THREE.MeshLambertMaterial({ color: 0x8b5a2b });
  const trunk = new THREE.Mesh(trunkGeo, trunkMat);
  trunk.position.set(x, height * 0.25, z);
  scene.add(trunk);

  const crownGeo = new THREE.SphereGeometry(height * 0.35, 12, 10);
  const crownMat = new THREE.MeshLambertMaterial({ color: 0x2f7d32 });
  const crown = new THREE.Mesh(crownGeo, crownMat);
  crown.position.set(x, height * 0.75, z);
  scene.add(crown);
}

function addWindows(building, footprint, height) {
  const windows = [];
  const stripMat = new THREE.MeshBasicMaterial({ color: 0xfff6cc });
  const stripH = 0.15;
  const gapV = 1.1;
  const startY = 2.0;

  const perimeters = [
    { axis: 'x', fixed: 'z', sign: 1 },
    { axis: 'x', fixed: 'z', sign: -1 },
    { axis: 'z', fixed: 'x', sign: 1 },
    { axis: 'z', fixed: 'x', sign: -1 },
  ];

  const worldPos = new THREE.Vector3();
  building.getWorldPosition(worldPos);

  perimeters.forEach(side => {
    const length = footprint * 0.95;
    const strips = Math.floor((height - startY) / gapV);
    for (let i = 0; i < strips; i++) {
      const y = startY + i * gapV + stripH / 2;
      const stripGeo = side.axis === 'x'
        ? new THREE.BoxGeometry(length, stripH, 0.06)
        : new THREE.BoxGeometry(0.06, stripH, length);
      const m = new THREE.Mesh(stripGeo, stripMat);
      const fx = side.fixed === 'z' ? worldPos.x : worldPos.z;
      const offset = (footprint / 2) * side.sign;
      if (side.axis === 'x') {
        m.position.set(fx, y, worldPos.z + offset);
      } else {
        m.position.set(worldPos.x + offset, y, fx);
      }
      windows.push(m);
    }
  });
  return windows;
}

function addStreetTrees() {
  const total = 400;
  for (let i = 0; i < total; i++) {
    const ix = Math.floor((rng() * 2 - 1) * halfBlocks);
    const iz = Math.floor((rng() * 2 - 1) * halfBlocks);
    // near roads only
    if (!(ix % roadEvery === 0 || iz % roadEvery === 0)) continue;
    const x = ix * cellSize + (rng() * 2 - 1) * (cellSize * 0.45);
    const z = iz * cellSize + (rng() * 2 - 1) * (cellSize * 0.45);
    if (Math.abs(x) > groundSize * 0.48 || Math.abs(z) > groundSize * 0.48) continue;
    addTree(x, z, 2 + rng() * 3);
  }
}

// Deterministic RNG
function mulberry32(a) {
  return function() {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});