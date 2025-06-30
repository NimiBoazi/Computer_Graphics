// ---------------------------------------------------------------------------
// Base of the hoop  – base cylinder, pole, safety sponges
// ---------------------------------------------------------------------------
function createHoopBase(side, scene) {
    const FT = 0.3048, COURT_LEN = 94 * FT, PAD_Y = 0.11;
    const BASELINE_X = side * (COURT_LEN / 2), POST_OFFSET = 0.60;
    const baseX = BASELINE_X + side * POST_OFFSET;
    const padHeight = 0.45, padBottom = 0.90, padTop = 0.60;
    const postHeight = 2.60, postWidth = 0.35;

    const matYellowPole = new THREE.MeshStandardMaterial({ color: 0xFDB927, roughness: 0.5, metalness: 0.3 });
    const matPad = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.7, metalness: 0.1 });

    
    // the dark grey pyramid base
    const padGeo = new THREE.CylinderGeometry(padTop / 2, padBottom / 2, padHeight, 4, 1, false);
    const pad = new THREE.Mesh(padGeo, matPad);
    pad.castShadow = true; 
    pad.receiveShadow = true;
    pad.rotation.y = Math.PI / 4;
    pad.position.set(baseX, PAD_Y + padHeight / 2, 0);
    
    // the base pole
    const postGeo = new THREE.BoxGeometry(postWidth, postHeight, postWidth);
    const post = new THREE.Mesh(postGeo, matYellowPole);
    post.castShadow = true; 
    post.receiveShadow = true;
    post.position.set(baseX, PAD_Y + padHeight + postHeight / 2, 0);
    
    // Create a material for the blue safety sponges
    const matBlueSponge = new THREE.MeshStandardMaterial({ color: 0x00529b, roughness: 0.9 });
    
    // the two safety sponges on the sides of the base pole
    const spongeHeight = 1.8;
    const spongeWidth = 0.4;
    const spongeDepth = 0.6;
    const spongeGeo = new THREE.BoxGeometry(spongeDepth, spongeHeight, spongeWidth);

    // Safety sponges

    // Create the left sponge
    const spongeLeft = new THREE.Mesh(spongeGeo, matBlueSponge);
    spongeLeft.castShadow = true;
    spongeLeft.receiveShadow = true;

    // Position it on the floor, centered with the pole, and offset to the left side (negative Z)
    spongeLeft.position.set(
        baseX,
        spongeHeight / 2,
        -(postWidth / 2 + spongeWidth / 2) // Place it right next to the pole
    );

    // Create the right sponge by cloning the left one
    const spongeRight = spongeLeft.clone();
    // Move it to the right side (positive Z)
    spongeRight.position.z = (postWidth / 2 + spongeWidth / 2);

    // Create the two blue floor pads extending towards the back (like real NBA hoop)
    const rearPadHeight = 0.85;
    const rearPadDepth = 0.85;
    // We use the same width as the vertical sponges to align them perfectly.
    const rearPadGeo = new THREE.BoxGeometry(rearPadDepth, rearPadHeight, spongeWidth);

    // Create the left rear pad
    const rearPadLeft = new THREE.Mesh(rearPadGeo, matBlueSponge);
    rearPadLeft.castShadow = true;
    rearPadLeft.receiveShadow = true;

    // Position it on the floor, behind the vertical sponge
    rearPadLeft.position.set(
    baseX + side * (spongeDepth / 2 + rearPadDepth / 2),
    rearPadHeight / 2,
    spongeLeft.position.z  // Align it with the left vertical pad
    );

    // Create the right rear pad by cloning the left one
    const rearPadRight = rearPadLeft.clone();
    // Align it with the right vertical pad
    rearPadRight.position.z = spongeRight.position.z;

    // Add everything to the group
    const group = new THREE.Group();
    group.add(pad, post, spongeLeft, spongeRight, rearPadLeft, rearPadRight);
    scene.add(group);
    return group;
}

// ---------------------------------------------------------------------------
// UPPER SUPPORT  – post, diagonal arm, backboard, rim ,net
// ---------------------------------------------------------------------------
function addHoopUpperSupport(group, side = 1) {
    const FT = 0.3048, RIM_HEIGHT = 10 * FT, BASKET_FROM_BASE = 1.22;
    const POST_OFFSET = 0.60, BOARD_THICK = 0.05, BOARD_W = 1.83, BOARD_H = 1.07;
    const RIM_INNER_RADIUS = 0.2286, RIM_THICKNESS = 0.02;
    const RIM_BOARD_GAP = 0.15;
    const SQUARE_W = 24 * 0.0254, SQUARE_H = 18 * 0.0254, SQUARE_LINE_THICK = 0.02;
    const PAD_Y = 0.11, padHeight = 0.45, postHeight = 2.60, postWidth = 0.35;

    // POSITIONS
    const COURT_LEN = 94 * FT, BASELINE_X = side * (COURT_LEN / 2);
    const postTop = new THREE.Vector3(BASELINE_X + side * POST_OFFSET, PAD_Y + padHeight + postHeight, 0);
    const rimFrontPositionX = BASELINE_X - side * BASKET_FROM_BASE;
    const rimCenterX = (12.6) * side;
    const boardFrontX = rimFrontPositionX - side * RIM_BOARD_GAP;
    const boardCenterX = boardFrontX - side * (BOARD_THICK / 2);
    const boardCentre = new THREE.Vector3(boardCenterX, RIM_HEIGHT + BOARD_H / 2, 0);

    // MATERIALS
    const matOrangeConnect = new THREE.MeshStandardMaterial({ color: 0xFDB927, roughness: 0.4, metalness: 0.2 });
    const matRim = new THREE.MeshStandardMaterial({ color: 0xfd4600, roughness: 0.4, metalness: 0.2 });
    const matNet = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, roughness: 0.8, side: THREE.DoubleSide });
    const textureLoader = new THREE.TextureLoader();
    const backboardTexture = textureLoader.load('/src/textures/backboard.jpg');
    const matBackboard = new THREE.MeshPhysicalMaterial({ map: backboardTexture, color: 0xffffff, transparent: true, opacity: 0.6, roughness: 0.2, metalness: 0.1 });
    const matScoreboard = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.4 });
    const matScoreDigit = new THREE.MeshBasicMaterial({ color: 0xff4500 });


    // SCOREBOARD
    const scoreboardGroup = new THREE.Group();
    const scoreboardHeight = 0.6, scoreboardWidth = 1.5, scoreboardThick = 0.1;

    const scoreboardBox = new THREE.Mesh(
        new THREE.BoxGeometry(scoreboardThick, scoreboardHeight, scoreboardWidth),
        matScoreboard
    );
    scoreboardBox.castShadow = true;
    scoreboardBox.receiveShadow = true;

    // Position it above the backboard
    const scoreboardY = boardCentre.y + BOARD_H / 2 + scoreboardHeight / 2 + 0.1;
    scoreboardBox.position.set(boardCentre.x, scoreboardY, boardCentre.z);
    scoreboardGroup.add(scoreboardBox);

    // Support posts for the scoreboard
    const supportPost = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.1, 0.04), matOrangeConnect);
    const support1 = supportPost.clone();
    support1.castShadow = true;
    support1.receiveShadow = true;
    support1.position.set(boardCentre.x, scoreboardY - scoreboardHeight / 2 - 0.05, 0.5);
    const support2 = supportPost.clone();
    support2.position.set(boardCentre.x, scoreboardY - scoreboardHeight / 2 - 0.05, -0.5);
    scoreboardGroup.add(support1, support2);

    // Add placeholder score digits
    const digit_1 = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.2, 0.02), matScoreDigit);
    digit_1.position.set(scoreboardBox.position.x + side*0.06, scoreboardBox.position.y, -0.4);
    scoreboardGroup.add(digit_1);
    group.add(scoreboardGroup);

    // BACKBOARD & SQUARE
    const boardGeo = new THREE.BoxGeometry(BOARD_THICK, BOARD_H, BOARD_W);
    const board = new THREE.Mesh(boardGeo, matBackboard);
    board.position.copy(boardCentre);
    board.castShadow = true;
    group.add(board);

    const outlineMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const topOutline = new THREE.Mesh(new THREE.BoxGeometry(BOARD_THICK + 0.001, 0.01, BOARD_W + 0.01 * 2), outlineMaterial);
    topOutline.position.set(boardCentre.x, boardCentre.y + BOARD_H / 2 - 0.01 / 2, boardCentre.z); group.add(topOutline);

    const bottomOutline = topOutline.clone(); bottomOutline.position.y = boardCentre.y - BOARD_H / 2 + 0.01 / 2; group.add(bottomOutline);
    const sideOutline = new THREE.Mesh(new THREE.BoxGeometry(BOARD_THICK + 0.001, BOARD_H + 0.01 * 2, 0.01), outlineMaterial);
    sideOutline.position.set(boardCentre.x, boardCentre.y, boardCentre.z + BOARD_W / 2 - 0.01 / 2); group.add(sideOutline);
    
    const otherSideOutline = sideOutline.clone(); otherSideOutline.position.z = boardCentre.z - BOARD_W / 2 + 0.01 / 2; group.add(otherSideOutline);
    const squareGroup = new THREE.Group();
    const squareXPos = boardFrontX + side * 0.002;
    const squareCenterY = RIM_HEIGHT + SQUARE_H / 2;
    const topBar = new THREE.Mesh(new THREE.BoxGeometry(0.002, SQUARE_LINE_THICK, SQUARE_W), outlineMaterial);
    topBar.position.set(squareXPos, squareCenterY + SQUARE_H / 2 - SQUARE_LINE_THICK / 2, 0);
    
    const bottomBar = new THREE.Mesh(new THREE.BoxGeometry(0.002, SQUARE_LINE_THICK, SQUARE_W), outlineMaterial);
    bottomBar.position.set(squareXPos, RIM_HEIGHT + SQUARE_LINE_THICK / 2, 0);
    
    const leftBar = new THREE.Mesh(new THREE.BoxGeometry(0.002, SQUARE_H - SQUARE_LINE_THICK, SQUARE_LINE_THICK), outlineMaterial);
    leftBar.position.set(squareXPos, squareCenterY, -SQUARE_W / 2 + SQUARE_LINE_THICK / 2);
    
    const rightBar = new THREE.Mesh(new THREE.BoxGeometry(0.002, SQUARE_H - SQUARE_LINE_THICK, SQUARE_LINE_THICK), outlineMaterial);
    rightBar.position.set(squareXPos, squareCenterY, SQUARE_W / 2 - SQUARE_LINE_THICK / 2);
    squareGroup.add(topBar, bottomBar, leftBar, rightBar);
    group.add(squareGroup);

    // DIAGONAL ARM & SUPPORTS
    const armThickness = postWidth / 3;
    const connectBox = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.15, 0.3), matOrangeConnect);
    connectBox.castShadow = true;
    connectBox.position.set(boardCenterX - side * (BOARD_THICK/2 + 0.01), boardCentre.y + BOARD_H/2 - 0.15, 0);
    
    const mainArmStart = new THREE.Vector3().copy(postTop);
    const mainArmEnd = new THREE.Vector3(boardCentre.x - side * (BOARD_THICK / 2), boardCentre.y + (BOARD_H/2) - 0.15, 0);
    const deltaMainArm = new THREE.Vector3().subVectors(mainArmEnd, mainArmStart);
    const mainArmGeo = new THREE.BoxGeometry(deltaMainArm.length(), armThickness, armThickness);
    mainArmGeo.translate(deltaMainArm.length() / 2, 0, 0);
    
    const arm1 = new THREE.Mesh(mainArmGeo, matOrangeConnect);
    arm1.castShadow = true;
    arm1.receiveShadow = true;
    arm1.position.copy(mainArmStart);
    arm1.position.z += armThickness;
    arm1.rotation.z = Math.atan2(deltaMainArm.y, deltaMainArm.x);
    group.add(arm1);

    const arm2 = arm1.clone();
    arm2.position.z -= armThickness * 2;
    group.add(arm2);

    // the support beams - from the main pole to the horizontal pole
    const midArmPoint = new THREE.Vector3().lerpVectors(mainArmStart, mainArmEnd, 0.5);
    const poleSupportStart = new THREE.Vector3().copy(postTop);
    poleSupportStart.y -= postHeight * 0.4;

    const supportThickness = armThickness * 0.8;
    const beamGeo = new THREE.BoxGeometry(1, supportThickness, supportThickness);
    const beam1Start = new THREE.Vector3(poleSupportStart.x, poleSupportStart.y, poleSupportStart.z + supportThickness);
    const beam1End = new THREE.Vector3(midArmPoint.x, midArmPoint.y, midArmPoint.z + armThickness / 2);
    const deltaBeam1 = new THREE.Vector3().subVectors(beam1End, beam1Start);
    const beam1 = new THREE.Mesh(beamGeo, matOrangeConnect);
    beam1.castShadow = true;
    beam1.scale.x = deltaBeam1.length();
    beam1.position.copy(beam1Start).add(deltaBeam1.clone().multiplyScalar(0.5));
    beam1.rotation.z = Math.atan2(deltaBeam1.y, deltaBeam1.x);
    group.add(beam1);

    const beam2Start = new THREE.Vector3(poleSupportStart.x, poleSupportStart.y, poleSupportStart.z - supportThickness);
    const beam2End = new THREE.Vector3(midArmPoint.x, midArmPoint.y, midArmPoint.z - armThickness / 2);
    const deltaBeam2 = new THREE.Vector3().subVectors(beam2End, beam2Start);
    const beam2 = new THREE.Mesh(beamGeo.clone(), matOrangeConnect);
    beam2.castShadow = true;
    beam2.scale.x = deltaBeam2.length();
    beam2.position.copy(beam2Start).add(deltaBeam2.clone().multiplyScalar(0.5));
    beam2.rotation.z = Math.atan2(deltaBeam2.y, deltaBeam2.x);
    group.add(beam2);


    // RIM AND SUPPORTS
    const rimGeo = new THREE.TorusGeometry(RIM_INNER_RADIUS, RIM_THICKNESS, 30, 100);
    const rim = new THREE.Mesh(rimGeo, matRim);
    rim.castShadow = true; rim.receiveShadow = true;
    rim.rotation.x = Math.PI / 2;
    rim.position.set(rimCenterX, RIM_HEIGHT, 0);
    group.add(rim);

    const rimPlate = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.02, 0.10), matOrangeConnect);
    rimPlate.position.set(boardFrontX - side * 0.005, RIM_HEIGHT - 0.02, 0);
    group.add(rimPlate);
    
    
    // Rim Connector Box

    const plateX = rimPlate.position.x;
    const rimNearX = rim.position.x + (side * RIM_INNER_RADIUS);

    const connectorDepth = Math.abs(rimNearX - plateX);
    const connectorWidth = 0.1; // left-to-right width
    const connectorHeight = RIM_THICKNESS; // same thickness as rim

    const connectorGeo = new THREE.BoxGeometry(connectorDepth, connectorHeight, connectorWidth);

    const rimConnector = new THREE.Mesh(connectorGeo, matOrangeConnect);
    rimConnector.castShadow = true;

    rimConnector.position.set(
        (plateX + rimNearX) / 2,
        RIM_HEIGHT - RIM_THICKNESS - (connectorHeight / 2),
        0
    );

    group.add(rimConnector);


    // hoop net
    const netHookGeo = new THREE.CylinderGeometry(0.005, 0.005, RIM_THICKNESS * 2, 8);
    netHookGeo.rotateX(Math.PI / 2);
    for (let i = 0; i < 12; i++) { const angle = (i / 12) * Math.PI * 2; const hook = new THREE.Mesh(netHookGeo, matRim); hook.position.set(rim.position.x + Math.cos(angle) * RIM_INNER_RADIUS, RIM_HEIGHT, rim.position.z + Math.sin(angle) * RIM_INNER_RADIUS); hook.rotation.z = angle + Math.PI / 2; group.add(hook); }
    const netHeight = 0.45;
    const netStringGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -netHeight / 20, 0)]), 1, 0.001, 8, false);
    for (let i = 0; i < 20; i++) { const angle = (i / 20) * Math.PI * 2; const startX = rim.position.x + Math.cos(angle) * (RIM_INNER_RADIUS - RIM_THICKNESS / 2); const startZ = rim.position.z + Math.sin(angle) * (RIM_INNER_RADIUS - RIM_THICKNESS / 2); for (let j = 0; j < 20; j++) { const string = new THREE.Mesh(netStringGeo, matNet); string.position.set(startX, rim.position.y - (j * netHeight / 20), startZ); string.rotation.y = angle; group.add(string); } }
    const numNetRings = 5;
    for (let i = 1; i <= numNetRings; i++) { const currentRingRadius = RIM_INNER_RADIUS; const ringY = rim.position.y - (i / numNetRings) * netHeight; const ringGeo = new THREE.TorusGeometry(currentRingRadius, 0.001, 8, 50); const ring = new THREE.Mesh(ringGeo, matNet); ring.rotation.x = Math.PI / 2; ring.position.set(rim.position.x, ringY, rim.position.z); group.add(ring); }
}

export function createHoops(scene){
  const leftHoop  = createHoopBase(-1, scene);
  addHoopUpperSupport(leftHoop, -1);

  const rightHoop = createHoopBase(+1, scene);
  addHoopUpperSupport(rightHoop,  1);
  }