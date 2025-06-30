export function createBasketball(scene) {
    const radius = 0.12;
    const segments = 64;
    const textureLoader = new THREE.TextureLoader();
    const bumpTexture = textureLoader.load('/src/textures/rubber_basketball.jpeg');

    const material = new THREE.MeshStandardMaterial({
        color: 0xD35400,
        roughness: 0.6,
        metalness: 0.0,
        bumpMap: bumpTexture,
        bumpScale: 0.03
    });

    const geometry = new THREE.SphereGeometry(radius, segments, segments);
    const ball = new THREE.Mesh(geometry, material);
    ball.castShadow = true;

    const seamMaterial = new THREE.MeshBasicMaterial({ color: 0x222222 });
    const seamRadius = 0.0015;
    const seamOffset = 0.001;
    const currentRadius = radius + seamOffset; // Draw seams slightly outside the ball

    // In a standard basketball, there are two main vertical seams perpendicular to each other.
    const verticalSeam1Geom = new THREE.TorusGeometry(currentRadius, seamRadius, 16, 100);
    const verticalSeam1 = new THREE.Mesh(verticalSeam1Geom, seamMaterial);
    verticalSeam1.rotation.x = Math.PI / 2;
    ball.add(verticalSeam1);

    const verticalSeam2Geom = new THREE.TorusGeometry(currentRadius, seamRadius, 16, 100);
    const verticalSeam2 = new THREE.Mesh(verticalSeam2Geom, seamMaterial);
    verticalSeam2.rotation.x = Math.PI / 2; // Stand it up
    verticalSeam2.rotation.y = Math.PI / 2; // Rotate it 90 degrees
    ball.add(verticalSeam2);

    // 3. Two Ovals on opposite sides
    const ovalXRadius = currentRadius * 0.98;
    const ovalYRadius = currentRadius * 0.75;

    const curve2D = new THREE.EllipseCurve(0, 0, ovalXRadius, ovalYRadius, 0, 2 * Math.PI, false, 0);
    const points2D = curve2D.getPoints(50);
    const points3D = points2D.map(p => {
        const x = p.x;
        const y = p.y;
        const zSquared = Math.pow(currentRadius, 2) - Math.pow(x, 2) - Math.pow(y, 2);
        const z = zSquared > 0 ? Math.sqrt(zSquared) : 0;
        return new THREE.Vector3(x, y, z);
    });

    const curve3D = new THREE.CatmullRomCurve3(points3D, true);
    const tubeGeom = new THREE.TubeGeometry(curve3D, 64, seamRadius, 8, false);

    const oval1 = new THREE.Mesh(tubeGeom, seamMaterial);
    oval1.rotation.y = Math.PI / 2;
    ball.add(oval1);

    const oval2 = oval1.clone();
    oval2.rotation.y = -Math.PI / 2;
    ball.add(oval2);

    // Center court position
    ball.position.set(0, radius + 0.12, 0);
    scene.add(ball);

    return ball;
}