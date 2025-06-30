function createCenterCircle(scene) {
  const circleRadius = 1.8;
  const circleThickness = 0.05;
  const circleSegments = 64;

  // Create the Team Logo
  const textureLoader = new THREE.TextureLoader();
  const logoTexture = textureLoader.load('/src/textures/team_logo.png');

  // The logo will be a flat circle shape, fitting inside the white ring.
  const logoRadius = circleRadius - circleThickness;
  const logoGeometry = new THREE.CircleGeometry(logoRadius, circleSegments);
  const logoMaterial = new THREE.MeshBasicMaterial({
    map: logoTexture,
    transparent: true,
  });
  const centerLogo = new THREE.Mesh(logoGeometry, logoMaterial);

  // Position it slightly below the white line
  centerLogo.position.set(0, 0.105, 0);
  centerLogo.rotation.x = -Math.PI / 2;
  scene.add(centerLogo);


  // Create the White Circle Line
  const ringGeometry = new THREE.RingGeometry(circleRadius - circleThickness, circleRadius, circleSegments);
  const ringMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const centerCircleRing = new THREE.Mesh(ringGeometry, ringMaterial);

  centerCircleRing.position.set(0, 0.11, 0); // Position it slightly higher than the logo
  centerCircleRing.rotation.x = -Math.PI / 2;
  scene.add(centerCircleRing);
}

function createCenterLine(scene) {
  const lineLength = 15;     // same as court width
  const lineWidth = 0.05;    // 5 cm wide
  const lineHeight = 0.01;   // lays on top of the court

  const geometry = new THREE.BoxGeometry(lineWidth, lineHeight, lineLength);
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const centerLine = new THREE.Mesh(geometry, material);

  centerLine.position.set(0, 0.11, 0);
  scene.add(centerLine);
}


function createThreePointLines(scene) {
  const COURT_LENGTH = 28;                  // metres (-14 … +14 on X axis)
  const COURT_WIDTH  = 15;                  // metres (-7.5 … +7.5 on Z axis)
  const LINE_THICK   = 0.05;               // 5 cm – matches centre circle
  const BASKET_FROM_BASELINE = 1.22;       // 4 ft ≈ 1.22 m
  const STRAIGHT_INSET       = 0.90;       // 0.90 m in from each sideline
  const R_CORNER = 6.70;                   // 22 ft ≈ 6.70 m (baseline corners)
  const R_ARC    = 7.24;                   // 23 ft 9 in ≈ 7.24 m (top of key)

  const halfWidth      = COURT_WIDTH / 2;             
  const cornerZ        = halfWidth - STRAIGHT_INSET;  
  const arcHalfAngle   = Math.asin(cornerZ / R_ARC);  
  const matWhite       = new THREE.MeshBasicMaterial({ color: 0xffffff,
                                                       side: THREE.DoubleSide });

  [-1, 1].forEach(side => {                 // -1 = left basket, +1 = right
    const baselineX     = side *  COURT_LENGTH / 2;   // -14 or +14
    const hoopX         = baselineX - side * BASKET_FROM_BASELINE; // Center of the arc
    const arcDx         = Math.sqrt(R_ARC*R_ARC - cornerZ*cornerZ);
    const arcEndX       = hoopX - side * arcDx;       // where arc meets straight
    const straightLen   = Math.abs(baselineX - arcEndX);

    // Straight corner lines
    [ cornerZ, -cornerZ ].forEach(z => {
      const geom = new THREE.PlaneGeometry(straightLen, LINE_THICK);
      const mesh = new THREE.Mesh(geom, matWhite);
      
      // Position the center of the plane correctly
      mesh.position.set((baselineX + arcEndX) / 2, 0.11, z);
      
      // Rotate the plane to lay flat on the court floor
      mesh.rotation.x = -Math.PI / 2;
      
      scene.add(mesh);
    });

    // Curved arc
    const thetaStart = (side === -1 ? -arcHalfAngle     // left basket: centred on +X
                                    :  Math.PI - arcHalfAngle); // right basket: centred on –X
    const thetaLen   = arcHalfAngle * 2;

    const ring = new THREE.RingGeometry(R_ARC - LINE_THICK,
                                        R_ARC,
                                        64, 1,
                                        thetaStart,
                                        thetaLen);

    const arcMesh = new THREE.Mesh(ring, matWhite);
    arcMesh.position.set(hoopX, 0.11, 0);
    arcMesh.rotation.x = -Math.PI / 2;     // lay flat on court
    scene.add(arcMesh);
  });
}

function createFreeThrowCircle(scene) {
  const COURT_LENGTH = 28;
  const LINE_THICK   = 0.05;
  const FREE_THROW_LINE_DIST = 5.79;     // 19' ≈ 5.79m (from baseline)
  const FREE_THROW_CIRCLE_RADIUS = 1.83; // 6' ≈ 1.83m

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for both sides of the court (-1 for left, +1 for right)
  [-1, 1].forEach(side => {
    const baselineX = side * COURT_LENGTH / 2; // -14 or +14

    // The center of the semi-circle is at the midpoint of the free-throw line.
    const circleCenterX = baselineX - side * FREE_THROW_LINE_DIST;
    const thetaStart = (side === 1) ? Math.PI / 2 : -Math.PI / 2;
    
    const ringGeom = new THREE.RingGeometry(
      FREE_THROW_CIRCLE_RADIUS - LINE_THICK,
      FREE_THROW_CIRCLE_RADIUS,
      64,          // for a smooth curve
      1,
      thetaStart,  // The corrected starting angle
      Math.PI      // The length of the arc (180 degrees)
    );
    
    const ringMesh = new THREE.Mesh(ringGeom, matWhite);

    // Position the mesh at the center of the circle
    ringMesh.position.set(
      circleCenterX,
      0.11,
      0
    );

    // Rotate to lay flat on the court
    ringMesh.rotation.x = -Math.PI / 2;
    scene.add(ringMesh);
  });
}

function createLaneLines(scene) {
  const COURT_LENGTH = 28;
  const LINE_THICK   = 0.05;
  const KEY_WIDTH = 4.88;                   // 16' ≈ 4.88m
  const FREE_THROW_LINE_DIST = 5.79;     // 19' ≈ 5.79m (length of the line)

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for both sides of the court (-1 for left, +1 for right)
  [-1, 1].forEach(side => {
    const baselineX = side * COURT_LENGTH / 2;
    const laneLineLength = FREE_THROW_LINE_DIST;
    
    // Loop to draw the two parallel lines for each key
    [1, -1].forEach(zSide => {
      const geom = new THREE.PlaneGeometry(laneLineLength, LINE_THICK);
      const mesh = new THREE.Mesh(geom, matWhite);

      // Position the center of the line correctly
      const lineCenterX = baselineX - side * (laneLineLength / 2);
      const lineZ = zSide * (KEY_WIDTH / 2);

      mesh.position.set(lineCenterX, 0.11, lineZ);
      mesh.rotation.x = -Math.PI / 2; // Lay flat

      scene.add(mesh);
    });
  });
}

function createInnerKeyLines(scene) {
  const COURT_LENGTH = 28;
  const LINE_THICK   = 0.05;
  const FREE_THROW_LINE_DIST = 5.79;     // line's length
  const FREE_THROW_CIRCLE_RADIUS = 1.83; // line's position

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for both sides of the court (-1 for left, +1 for right)
  [-1, 1].forEach(side => {
    const baselineX = side * COURT_LENGTH / 2;
    const lineLength = FREE_THROW_LINE_DIST;
    
    // Loop to draw the two parallel lines for each key
    [1, -1].forEach(zSide => {
      const geom = new THREE.PlaneGeometry(lineLength, LINE_THICK);
      const mesh = new THREE.Mesh(geom, matWhite);

      // The X position is identical to the outer lane lines
      const lineCenterX = baselineX - side * (lineLength / 2);
      
      // We use the circle's radius for the Z position to align with the semi-circle
      const lineZ = zSide * FREE_THROW_CIRCLE_RADIUS;

      mesh.position.set(lineCenterX, 0.11, lineZ);
      mesh.rotation.x = -Math.PI / 2; // Lay flat

      scene.add(mesh);
    });
  });
}

function createFreeThrowLine(scene) {
  const COURT_LENGTH = 28;
  const LINE_THICK   = 0.05;
  const KEY_WIDTH    = 4.88;               // length of the free-throw line.
  const FREE_THROW_LINE_DIST = 5.79;     // position of the free-throw line.

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for both sides of the court (-1 for left, +1 for right)
  [-1, 1].forEach(side => {
    const baselineX = side * COURT_LENGTH / 2;
    const ftLineLength = KEY_WIDTH;

    const geom = new THREE.PlaneGeometry(LINE_THICK, ftLineLength);
    const mesh = new THREE.Mesh(geom, matWhite);

    // The X position is at the end of the lane lines
    const positionX = baselineX - side * FREE_THROW_LINE_DIST;

    // The line is centered on the court horizontally
    mesh.position.set(positionX, 0.11, 0);
    mesh.rotation.x = -Math.PI / 2; // Lay flat

    scene.add(mesh);
  });
}

function createDottedFreeThrowCircle(scene) {
  const COURT_LENGTH = 28;
  const FREE_THROW_LINE_DIST = 5.79;
  const FREE_THROW_CIRCLE_RADIUS = 1.83;
  const DASH_SIZE = 0.2;
  const GAP_SIZE = 0.1;

  const matDashed = new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: DASH_SIZE,
    gapSize: GAP_SIZE,
  });

  // Loop for both sides of the court
  [-1, 1].forEach(side => {
    // Define the curve at the origin
    let startAngle, endAngle;

    // We want the half of the circle that opens towards the basket.
    // The solid line opens towards the center of the court.
    if (side === 1) {
      // Right side: Basket is in +X direction. We need the arc from -PI/2 to PI/2.
      startAngle = -Math.PI / 2;
      endAngle = Math.PI / 2;
    } else { // side === -1
      // Left side: Basket is in -X direction. We need the arc from PI/2 to 3*PI/2.
      startAngle = Math.PI / 2;
      endAngle = 3 * Math.PI / 2;
    }

    const curve = new THREE.EllipseCurve(
      0, 0, // Create at the origin
      FREE_THROW_CIRCLE_RADIUS, FREE_THROW_CIRCLE_RADIUS,
      startAngle, endAngle,
      false,
      0
    );

    // Create the geometry and line object
    const points = curve.getPoints(50);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const dottedLine = new THREE.Line(geometry, matDashed);

    // Rotate the line to lay it flat
    dottedLine.rotation.x = -Math.PI / 2;

    // Move the flat line to its final position
    const baselineX = side * COURT_LENGTH / 2;
    const circleCenterX = baselineX - side * FREE_THROW_LINE_DIST;
    dottedLine.position.set(circleCenterX, 0.11, 0);

    // Compute distances for dashing
    dottedLine.computeLineDistances();

    scene.add(dottedLine);
  });
}

function createCoachBoxLines(scene) {
  const COURT_LENGTH = 28;
  const COURT_WIDTH  = 15;
  const LINE_THICK   = 0.05;
  const COACH_BOX_LINE_LENGTH = 0.91;
  const COACH_BOX_DIST_FROM_BASELINE = 8.53;

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for each sideline (top: +1, bottom: -1)
  [1, -1].forEach(sideZ => {
    // Loop for each side of the court (right: +1, left: -1)
    [1, -1].forEach(sideX => {

      const geom = new THREE.PlaneGeometry(LINE_THICK, COACH_BOX_LINE_LENGTH);
      const mesh = new THREE.Mesh(geom, matWhite);

      // Calculate the X position based on the distance from the baseline.
      const baselineX = sideX * (COURT_LENGTH / 2);
      const positionX = baselineX - sideX * COACH_BOX_DIST_FROM_BASELINE;

      // Calculate the Z position. The line starts on the sideline and goes inwards.
      // So, its center must be half its length away from the sideline.
      const sidelineZ = sideZ * (COURT_WIDTH / 2);
      const positionZ = sidelineZ - sideZ * (COACH_BOX_LINE_LENGTH / 2);

      // Set the final position and rotation
      mesh.position.set(positionX, 0.11, positionZ);
      mesh.rotation.x = -Math.PI / 2; // Lay flat

      scene.add(mesh);
    });
  });
}

// Draws the 16 hash marks on the outer lane lines of the key.
function createKeyStubs(scene) {
  const COURT_LENGTH = 28;
  const KEY_WIDTH    = 4.88;
  const LINE_THICK   = 0.05;
  const WIDE_STUB_THICKNESS = 0.305; // 1 foot thick
  const STUB_LENGTH         = 0.305; // 1 foot long
  const STUB_DISTANCES = [
    2.13, // 7 feet
    3.05, // 10 feet
    3.96, // 13 feet
    4.88  // 16 feet
  ];

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for each side of the court (right: +1, left: -1)
  [1, -1].forEach(sideX => {
    // Loop for each outer lane line (top: +1, bottom: -1)
    [1, -1].forEach(sideZ => {
      // Loop through each of the four required distances for the stubs
      STUB_DISTANCES.forEach((distance, index) => {
        
        const stubWidth = (index === 0) ? WIDE_STUB_THICKNESS : LINE_THICK;
        const geom = new THREE.PlaneGeometry(stubWidth, STUB_LENGTH);
        const mesh = new THREE.Mesh(geom, matWhite);

        // Positioning
        const baselineX = sideX * (COURT_LENGTH / 2);
        const positionX = baselineX - sideX * distance;

        const laneLineZ = sideZ * (KEY_WIDTH / 2);
        const positionZ = laneLineZ + sideZ * (STUB_LENGTH / 2);

        mesh.position.set(positionX, 0.11, positionZ);
        mesh.rotation.x = -Math.PI / 2;

        scene.add(mesh);
      });
    });
  });
}

// Draws the restricted area arc under the basket with the corrected radius.
export function createRestrictedAreaArc(scene) {
  // Constants
  const COURT_LENGTH = 28;
  const LINE_THICK   = 0.05;

  const ARC_RADIUS = 1.07; // 3.5 feet
  const HOOP_CENTER_FROM_BASELINE = 1.22; // 4 feet

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });

  // Loop for both sides of the court (right: +1, left: -1)
  [1, -1].forEach(side => {
    const baselineX = side * (COURT_LENGTH / 2);
    const arcCenterX = baselineX - side * HOOP_CENTER_FROM_BASELINE;
    const thetaStart = (side === 1) ? Math.PI / 2 : -Math.PI / 2;

    const ringGeom = new THREE.RingGeometry(
      ARC_RADIUS - LINE_THICK,
      ARC_RADIUS,
      64,
      1,
      thetaStart,
      Math.PI
    );
    
    const arcMesh = new THREE.Mesh(ringGeom, matWhite);

    arcMesh.position.set(arcCenterX, 0.11, 0);
    arcMesh.rotation.x = -Math.PI / 2;

    scene.add(arcMesh);
  });
}

export function createBoundaryLines(scene) {
  const LINE_THICK = 0.05;   // 5 cm wide line
  const LINE_HEIGHT = 0.01;  // thin height
  const COURT_LENGTH = 28;   // official playing area length
  const COURT_WIDTH  = 15;   // official playing area width

  const halfLength = COURT_LENGTH / 2;  // 14
  const halfWidth  = COURT_WIDTH  / 2;  // 7.5

  const matWhite = new THREE.MeshBasicMaterial({ color: 0xffffff });

  // Baselines
  [-1, 1].forEach(side => {
    const geometry = new THREE.BoxGeometry(LINE_THICK, LINE_HEIGHT, COURT_WIDTH);
    const baseline = new THREE.Mesh(geometry, matWhite);
    baseline.position.set(side * halfLength, 0.11, 0);
    scene.add(baseline);
  });

  // Sidelines
  [-1, 1].forEach(side => {
    const geometry = new THREE.BoxGeometry(COURT_LENGTH, LINE_HEIGHT, LINE_THICK);
    const sideline = new THREE.Mesh(geometry, matWhite);
    sideline.position.set(0, 0.11, side * halfWidth);
    scene.add(sideline);
  });
}

export function createCourtLines(scene) {
  createCenterCircle(scene);
  createCenterLine(scene);
  createThreePointLines(scene);
  createBoundaryLines(scene);
  createFreeThrowCircle(scene)
  createLaneLines(scene)
  createInnerKeyLines(scene)
  createFreeThrowLine(scene)
  createDottedFreeThrowCircle(scene)
  createCoachBoxLines(scene)
  createKeyStubs(scene)
  createRestrictedAreaArc(scene)
}