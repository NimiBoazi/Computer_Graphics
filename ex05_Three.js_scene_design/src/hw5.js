import {OrbitControls} from './OrbitControls.js'
import { createCourtLines } from './CourtLines.js';
import { createHoops } from './Hoops.js';
import { createBasketball } from './Basketball.js';


const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Set background color
scene.background = new THREE.Color(0x000000);

// === LIGHTS ===

// Ambient light: softens shadows a bit
const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
scene.add(ambientLight);

// Main directional light: acts like stadium floodlight
const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
dirLight.position.set(-15, 10, 5);
dirLight.castShadow = true;
scene.add(dirLight);


// Sharper shadows: tweak shadow map
dirLight.shadow.mapSize.width = 2048;
dirLight.shadow.mapSize.height = 2048;
dirLight.shadow.camera.near = 0.5;
dirLight.shadow.camera.far = 50;

// 3 fill point lights: from other angles to brighten dark spots
const fillLight1 = new THREE.PointLight(0xffffff, 0.3);
fillLight1.position.set(-15, 15, -10);
scene.add(fillLight1);

const fillLight2 = new THREE.PointLight(0xffffff, 0.3);
fillLight2.position.set(15, 15, -10);
scene.add(fillLight2);

const fillLight3 = new THREE.PointLight(0xffffff, 0.3);
fillLight3.position.set(0, 15, 15);
scene.add(fillLight3);

// Enable shadows
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

function degrees_to_radians(degrees) {
  var pi = Math.PI;
  return degrees * (pi/180);
}

// Create basketball court
function createBasketballCourt() {

  // Layer 1: The large yellow out-of-bounds (apron)
  const apronMaterial = new THREE.MeshStandardMaterial({
    color: 0xFDB927,
    roughness: 0.5,
  });
  const apronGeometry = new THREE.PlaneGeometry(32, 19);
  const apron = new THREE.Mesh(apronGeometry, apronMaterial);
  apron.receiveShadow = true;
  apron.rotation.x = -Math.PI / 2;
  apron.position.y = 0.00;
  scene.add(apron);

  // Layer 2: The wood-textured playing court on top
  const textureLoader = new THREE.TextureLoader();
  const woodTexture = textureLoader.load('/src/textures/WoodFloor.jpg');
  woodTexture.wrapS = THREE.RepeatWrapping;
  woodTexture.wrapT = THREE.RepeatWrapping;
  woodTexture.repeat.set(14, 8);

  // Same dimentions as real playing court
  const courtGeometry = new THREE.PlaneGeometry(28, 15);
  const courtMaterial = new THREE.MeshStandardMaterial({
    map: woodTexture,
    roughness: 0.4
  });
  const court = new THREE.Mesh(courtGeometry, courtMaterial);
  court.receiveShadow = true;
  court.rotation.x = -Math.PI / 2;
  // Position this layer slightly higher than the apron to prevent flickering
  court.position.y = 0.05;
  scene.add(court);

  // NBA logo on sidelines
  const nbaLogoTexture = new THREE.TextureLoader().load('/src/textures/nba_logo.png');
  const logoMaterial = new THREE.MeshBasicMaterial({
    map: nbaLogoTexture,
    transparent: true,
    side: THREE.DoubleSide
  });

  // Make the logo bigger
  const logoWidth = 1.5;
  const logoHeight = 0.75;
  const logoGeometry = new THREE.PlaneGeometry(logoWidth, logoHeight);

  // Shift more along the court length
  const offsetX = 5.0;

  // Push closer to the edge of the floor
  const edgeOffset = 1;
  const courtHalfWidth = 15 / 2;

  const logoLeft = new THREE.Mesh(logoGeometry, logoMaterial);
  logoLeft.position.set(offsetX, 0.06, -(courtHalfWidth + edgeOffset));
  logoLeft.rotation.x = -Math.PI / 2;
  scene.add(logoLeft);

  const logoRight = logoLeft.clone();
  logoRight.position.set(-offsetX, 0.06, +(courtHalfWidth + edgeOffset));
  scene.add(logoRight);
 
  // Create all the lines on the court
  createCourtLines(scene);

  // Create the hoops
  createHoops(scene)

  // Create the basketball
  createBasketball(scene);
}


// Create all elements
createBasketballCourt();



// cameras
// main camera
const cameraMain = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
cameraMain.position.set(0, 12, 28);
cameraMain.lookAt(0, 0, 0);

// hoop camera
const hoopX = 12.78; // Right hoop X position
const cameraBehindHoop = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
cameraBehindHoop.position.set(hoopX + 2, 2.5, 0);
cameraBehindHoop.lookAt(hoopX, 3.05, 0);


let activeCamera = cameraMain; // Start with main view
let controls = new OrbitControls(activeCamera, renderer.domElement);
controls.target.set(0, 0, 0); // Pivot: court center

let isOrbitEnabled = true;


function handleKeyDown(e) {
  if (e.key === "o") {
    isOrbitEnabled = !isOrbitEnabled;
  }
  if (e.key === "c") {
    // Swap camera
    if (activeCamera === cameraMain) {
      activeCamera = cameraBehindHoop;
      controls = new OrbitControls(activeCamera, renderer.domElement);
      controls.target.set(hoopX, 3.05, 0);
    } else {
      activeCamera = cameraMain;
      controls = new OrbitControls(activeCamera, renderer.domElement);
      controls.target.set(0, 0, 0);
    }
  }
}

document.addEventListener('keydown', handleKeyDown);

function animate() {
  requestAnimationFrame(animate);
  controls.enabled = isOrbitEnabled;
  controls.update();
  renderer.render(scene, activeCamera);
}

animate();