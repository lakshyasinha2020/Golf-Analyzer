import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export default function SwingMotionScene() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [webglReady, setWebglReady] = useState(true);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, preserveDrawingBuffer: true });
    } catch {
      setWebglReady(false);
      return undefined;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 0.6, 6.2);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const root = new THREE.Group();
    scene.add(root);

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color("rgb(28, 42, 34)"),
      metalness: 0.18,
      roughness: 0.42,
    });
    const accent = new THREE.MeshStandardMaterial({
      color: new THREE.Color("rgb(111, 219, 118)"),
      emissive: new THREE.Color("rgb(54, 139, 80)"),
      emissiveIntensity: 0.75,
      metalness: 0.2,
      roughness: 0.35,
    });
    const shadowMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color("rgb(12, 24, 19)"),
      transparent: true,
      opacity: 0.22,
    });

    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 1.05, 12, 22), material);
    torso.position.set(0, 0.78, 0);
    torso.rotation.z = -0.18;
    root.add(torso);

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 18), material);
    head.position.set(-0.13, 1.6, 0);
    root.add(head);

    const hip = new THREE.Mesh(new THREE.CapsuleGeometry(0.15, 0.58, 10, 16), material);
    hip.position.set(0.08, 0.2, 0);
    hip.rotation.z = 1.28;
    root.add(hip);

    const club = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 2.6, 12), accent);
    club.position.set(-0.75, 0.92, 0.05);
    club.rotation.z = -0.78;
    root.add(club);

    const leadArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.055, 1.06, 8, 14), material);
    leadArm.position.set(-0.42, 0.92, 0.02);
    leadArm.rotation.z = 0.88;
    root.add(leadArm);

    const trailArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.052, 0.9, 8, 14), material);
    trailArm.position.set(-0.22, 0.86, -0.02);
    trailArm.rotation.z = 0.58;
    root.add(trailArm);

    const leftLeg = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, 1.1, 8, 14), material);
    leftLeg.position.set(-0.28, -0.48, 0.02);
    leftLeg.rotation.z = -0.28;
    root.add(leftLeg);

    const rightLeg = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, 1.08, 8, 14), material);
    rightLeg.position.set(0.36, -0.5, -0.02);
    rightLeg.rotation.z = 0.24;
    root.add(rightLeg);

    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 12), accent);
    ball.position.set(-1.24, -1.06, 0.04);
    root.add(ball);

    const arc = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        new THREE.EllipseCurve(0, 0, 1.62, 1.15, 0.2, Math.PI * 1.62, false, 0.15).getPoints(92).map((point) => new THREE.Vector3(point.x - 0.08, point.y + 0.45, -0.24)),
      ),
      new THREE.LineBasicMaterial({ color: new THREE.Color("rgb(121, 235, 126)"), transparent: true, opacity: 0.78 }),
    );
    root.add(arc);

    const ground = new THREE.Mesh(new THREE.CircleGeometry(1.65, 64), shadowMaterial);
    ground.position.set(0, -1.24, -0.16);
    ground.scale.set(1.35, 0.18, 1);
    root.add(ground);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(2, 4, 5);
    scene.add(keyLight);
    scene.add(new THREE.AmbientLight(0xbceccc, 1.25));

    function resize() {
      if (!mount) return;
      const { width, height } = mount.getBoundingClientRect();
      renderer.setSize(width, height, false);
      camera.aspect = Math.max(width / Math.max(height, 1), 0.1);
      camera.updateProjectionMatrix();
    }

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    let animationFrame = 0;
    let start = performance.now();

    function animate(now: number) {
      const time = (now - start) / 1000;
      const swing = Math.sin(time * 1.15);
      const follow = Math.sin(time * 1.15 - 0.75);

      root.rotation.y = -0.18 + Math.sin(time * 0.45) * 0.16;
      torso.rotation.z = -0.18 + swing * 0.08;
      head.position.x = -0.13 + swing * 0.035;
      club.rotation.z = -0.78 + follow * 1.06;
      leadArm.rotation.z = 0.88 + follow * 0.48;
      trailArm.rotation.z = 0.58 + follow * 0.5;
      arc.rotation.z = time * 0.26;
      arc.scale.setScalar(1 + Math.max(0, follow) * 0.08);
      ball.scale.setScalar(1 + Math.max(0, follow) * 0.28);

      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    }

    animationFrame = window.requestAnimationFrame((now) => {
      start = now;
      animate(now);
    });

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          if (Array.isArray(object.material)) object.material.forEach((item) => item.dispose());
          else object.material.dispose();
        }
      });
    };
  }, []);

  return (
    <div className="swing-motion-scene" aria-label="3D animated golf swing preview" ref={mountRef}>
      {!webglReady ? <div className="swing-motion-fallback" aria-hidden="true" /> : null}
    </div>
  );
}
