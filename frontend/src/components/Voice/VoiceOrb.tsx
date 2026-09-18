import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export type OrbState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

interface VoiceOrbProps {
  state: OrbState;
  /** Live audio amplitude, 0–1. Drives radius, brightness and particle spread. */
  level: number;
  size?: number;
}

const PARTICLE_COUNT = 7000;

/**
 * Perplexity-style particle orb: a sphere made of points of light, with no
 * solid body at all.
 *
 * The "no solid body" part is the whole design. An earlier version put an
 * opaque displaced mesh at the centre with particles orbiting it — and because
 * that mesh wrote depth and sat above most of the particle altitudes, it
 * occluded nearly every particle and rendered as a plain blue blob. The look
 * comes from thousands of small additive sprites reading as one translucent
 * volume, so depth testing is off entirely and back-facing particles show
 * through the front ones.
 *
 * `state` and `level` are mirrored into refs because level updates at 60fps
 * from the audio meter; routing that through React state would re-render the
 * tree every audio frame. The scene is built once and reads the refs from the
 * RAF loop.
 */
export const VoiceOrb: React.FC<VoiceOrbProps> = ({ state, level, size = 280 }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<OrbState>(state);
  const levelRef = useRef<number>(level);

  stateRef.current = state;
  levelRef.current = level;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const CAM_Z = 3.4;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = CAM_Z;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(size, size);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    // Half the drawing-buffer height — the factor three.js itself uses for
    // sizeAttenuation. gl_PointSize is in PIXELS, so a world-space particle
    // radius has to be converted through this.
    const pixelScale = size * Math.min(window.devicePixelRatio, 2) * 0.5;

    // ── Particle field ────────────────────────────────────────
    const dirs = new Float32Array(PARTICLE_COUNT * 3);
    const seeds = new Float32Array(PARTICLE_COUNT);
    // Fibonacci spiral — the golden angle distributes points evenly. Random
    // sampling leaves visible clumps and bald patches at this density, which
    // reads as a dirty sphere rather than a field of light.
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const y = 1 - (i / (PARTICLE_COUNT - 1)) * 2;
      const ring = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * i;
      const jitter = 0.97 + Math.random() * 0.06;
      dirs[i * 3] = Math.cos(theta) * ring * jitter;
      dirs[i * 3 + 1] = y * jitter;
      dirs[i * 3 + 2] = Math.sin(theta) * ring * jitter;
      seeds[i] = Math.random();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(dirs, 3));
    geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

    const uniforms = {
      uTime: { value: 0 },
      uLevel: { value: 0 },
      uEnergy: { value: 0 },
      uScale: { value: pixelScale },
      uCamZ: { value: CAM_Z },
      // Seeded bright so the first frames aren't dark while the palette lerps in.
      uColorA: { value: new THREE.Color('#3B82FF') },
      uColorB: { value: new THREE.Color('#79ADFF') },
      uColorC: { value: new THREE.Color('#CFE2FF') },
    };

    const material = new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      // Depth testing off so the far hemisphere shows through the near one.
      // With it on, half the particles vanish and the orb reads as a hard
      // shell instead of a luminous volume.
      depthTest: false,
      blending: THREE.AdditiveBlending,
      vertexShader: `
        uniform float uTime; uniform float uLevel; uniform float uEnergy;
        uniform float uScale; uniform float uCamZ;
        attribute float aSeed;
        varying float vFront; varying float vSeed; varying float vDisp;

        vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
        vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
        vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
        vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
        float snoise(vec3 v){
          const vec2 C = vec2(1.0/6.0, 1.0/3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
          vec3 i  = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);
          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min(g.xyz, l.zxy);
          vec3 i2 = max(g.xyz, l.zxy);
          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;
          i = mod289(i);
          vec4 p = permute(permute(permute(
                     i.z + vec4(0.0, i1.z, i2.z, 1.0))
                   + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                   + i.x + vec4(0.0, i1.x, i2.x, 1.0));
          float n_ = 0.142857142857;
          vec3 ns = n_ * D.wyz - D.xzx;
          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);
          vec4 x = x_ * ns.x + ns.yyyy;
          vec4 y = y_ * ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);
          vec4 b0 = vec4(x.xy, y.xy);
          vec4 b1 = vec4(x.zw, y.zw);
          vec4 s0 = floor(b0)*2.0 + 1.0;
          vec4 s1 = floor(b1)*2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));
          vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
          vec3 p0 = vec3(a0.xy, h.x);
          vec3 p1 = vec3(a0.zw, h.y);
          vec3 p2 = vec3(a1.xy, h.z);
          vec3 p3 = vec3(a1.zw, h.w);
          vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
          p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
          vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
          m = m * m;
          return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
        }

        void main() {
          vec3 dir = normalize(position);

          // Two octaves: a slow swell that reads as breathing, plus a finer
          // band that only wakes with audio, so silence looks calm and speech
          // looks alive. Kept gentle — the silhouette should stay a sphere.
          float slow = snoise(dir * 1.25 + uTime * 0.075);
          float fast = snoise(dir * 3.20 - uTime * 0.20);
          float disp = slow * (0.05 + uEnergy * 0.04) + fast * uLevel * 0.16;

          // Each particle keeps its own shell offset so the surface has real
          // thickness instead of being an infinitely thin skin.
          float radius = 1.0 + (aSeed - 0.5) * 0.10 + disp + uLevel * 0.10;
          vec3 pos = dir * radius;

          vec4 mv = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mv;

          // Depth cue: nearer particles brighter and larger. Without this the
          // front and back hemispheres are indistinguishable and the orb looks
          // flat rather than volumetric.
          vFront = smoothstep(uCamZ + 1.2, uCamZ - 1.2, -mv.z);
          vSeed = aSeed;
          vDisp = disp;

          float worldSize = (0.014 + aSeed * 0.014 + uLevel * 0.014) * (0.65 + vFront * 0.7);
          gl_PointSize = max(1.5, worldSize * uScale / -mv.z);
        }
      `,
      fragmentShader: `
        uniform vec3 uColorA; uniform vec3 uColorB; uniform vec3 uColorC;
        uniform float uLevel; uniform float uEnergy;
        varying float vFront; varying float vSeed; varying float vDisp;
        void main() {
          // Round off the square point sprite with a soft radial falloff —
          // this is what makes each particle read as a mote of light rather
          // than a pixel block.
          vec2 uv = gl_PointCoord - 0.5;
          float d = length(uv);
          if (d > 0.5) discard;
          float sprite = smoothstep(0.5, 0.05, d);

          // Ride displacement and per-particle seed through a 3-stop gradient,
          // so the sphere is multi-hued rather than one flat tint. Biased
          // upward: at rest the displacement term is near zero, and an
          // unbiased t parked most particles on the darkest stop, which is
          // what made the whole orb read as dim dust.
          float t = clamp(vDisp * 4.0 + vSeed * 0.6 + 0.42, 0.0, 1.0);
          vec3 col = mix(uColorA, uColorB, smoothstep(0.0, 0.55, t));
          col = mix(col, uColorC, smoothstep(0.5, 1.0, t));

          float alpha = sprite * (0.32 + vFront * 0.68) * (0.85 + uEnergy * 0.25 + uLevel * 0.6);
          gl_FragColor = vec4(col, alpha);
        }
      `,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // ── State palettes: [inner, mid, accent, energy] ──────────
    // Every stop is a mid-to-bright tone. These are additively blended at
    // partial alpha over a dark background, so a "correct looking" dark navy
    // in a colour picker lands as near-invisible on screen.
    const palette: Record<OrbState, [string, string, string, number]> = {
      idle: ['#4C6FD4', '#7E9EEA', '#B9CCFF', 0.10],
      listening: ['#3B82FF', '#79ADFF', '#CFE2FF', 0.55],
      thinking: ['#9B6BFF', '#C79CFF', '#EEDDFF', 0.95],
      speaking: ['#1FCF9B', '#5FE8C0', '#C2FFEA', 0.75],
      error: ['#E85F54', '#FF9086', '#FFD2CC', 0.40],
    };

    const tA = new THREE.Color();
    const tB = new THREE.Color();
    const tC = new THREE.Color();
    let smoothed = 0;
    let frameId = 0;
    const clock = new THREE.Clock();

    let prevTime = 0;

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();
      // Clamped so a backgrounded tab doesn't resume with one enormous step.
      const dt = Math.min(0.05, t - prevTime);
      prevTime = t;
      const [a, b, c, energy] = palette[stateRef.current];

      // Asymmetric smoothing: fast attack, slow release, like a VU meter.
      // Binding raw amplitude straight to geometry strobes. Frame-rate
      // corrected so the response is identical at 60Hz and 144Hz.
      const target = levelRef.current;
      const k = target > smoothed ? 12.0 : 2.5;
      smoothed += (target - smoothed) * Math.min(1, k * dt);

      uniforms.uTime.value = t;
      uniforms.uLevel.value = smoothed;
      uniforms.uEnergy.value += (energy - uniforms.uEnergy.value) * Math.min(1, 2.0 * dt);

      tA.set(a);
      tB.set(b);
      tC.set(c);
      const colourStep = Math.min(1, 2.0 * dt);
      uniforms.uColorA.value.lerp(tA, colourStep);
      uniforms.uColorB.value.lerp(tB, colourStep);
      uniforms.uColorC.value.lerp(tC, colourStep);

      // Radians per SECOND, not per frame. The previous version incremented a
      // fixed amount each frame, so the orb spun ~2.4x faster on a 144Hz
      // display than on a 60Hz one — most of why this read as too fast.
      // Thinking is the only state with no audio to react to, so its faster
      // rotation is what tells the user it hasn't frozen.
      const spin = stateRef.current === 'thinking' ? 0.30 : 0.10;
      points.rotation.y += dt * spin;
      points.rotation.x = Math.sin(t * 0.10) * 0.20;

      points.scale.setScalar(1 + Math.sin(t * 0.6) * 0.018 + smoothed * 0.07);

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [size]);

  return (
    <div
      ref={mountRef}
      style={{ width: size, height: size, display: 'grid', placeItems: 'center' }}
      aria-hidden="true"
    />
  );
};
