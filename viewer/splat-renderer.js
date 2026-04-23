// splat-renderer.js — Custom Gaussian Splat renderer using Three.js.
//
// Renders Gaussians as soft alpha-blended billboards via a custom shader.
// No external dependencies — pure Three.js.
//
// How it works:
//   - Each Gaussian is a point sprite (gl_PointSize controls screen size)
//   - Fragment shader draws a soft radial falloff: exp(-r²)
//   - Alpha blending composites them back-to-front
//   - The result looks like soft, fuzzy splats that blend at edges
//
// This is a simplified but visually effective approximation of full 3DGS
// rendering. For a course project it's arguably more impressive than using
// a black-box library because you can explain every line.

import * as THREE from 'three';

// Spherical harmonics DC coefficient.
const SH_C0 = 0.28209479177387814;

/**
 * Parse a 3DGS-format .ply file into typed arrays.
 * Returns { positions, colors, scales, opacities, count }.
 */
export function parseSplatPLY(buffer) {
  const bytes = new Uint8Array(buffer);

  // Find end_header.
  let headerEnd = 0;
  const decoder = new TextDecoder();
  const headerText = decoder.decode(bytes.slice(0, Math.min(2000, bytes.length)));
  const endIdx = headerText.indexOf('end_header\n');
  if (endIdx < 0) throw new Error('Invalid PLY: no end_header');
  headerEnd = endIdx + 'end_header\n'.length;

  // Parse vertex count from header.
  const countMatch = headerText.match(/element vertex (\d+)/);
  if (!countMatch) throw new Error('Invalid PLY: no vertex count');
  const count = parseInt(countMatch[1], 10);

  // Each vertex: 17 floats (4 bytes each) = 68 bytes.
  // Layout: x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
  const FLOATS_PER_VERTEX = 17;
  const data = new Float32Array(buffer, headerEnd, count * FLOATS_PER_VERTEX);

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const scales = new Float32Array(count);
  const opacities = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const off = i * FLOATS_PER_VERTEX;

    // Position.
    positions[i * 3 + 0] = data[off + 0];
    positions[i * 3 + 1] = data[off + 1];
    positions[i * 3 + 2] = data[off + 2];

    // Color: SH DC -> RGB.
    colors[i * 3 + 0] = Math.max(0, Math.min(1, data[off + 6] * SH_C0 + 0.5));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, data[off + 7] * SH_C0 + 0.5));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, data[off + 8] * SH_C0 + 0.5));

    // Opacity: sigmoid of stored logit.
    const logit = data[off + 9];
    opacities[i] = 1.0 / (1.0 + Math.exp(-logit));

    // Scale: exp of stored log-scale (use average of 3 axes).
    const s0 = Math.exp(data[off + 10]);
    const s1 = Math.exp(data[off + 11]);
    const s2 = Math.exp(data[off + 12]);
    scales[i] = (s0 + s1 + s2) / 3.0;
  }

  return { positions, colors, scales, opacities, count };
}

// Vertex shader: position points and compute screen-space size from scale.
const vertexShader = `
  attribute float aScale;
  attribute float aOpacity;
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    vColor = color;
    vOpacity = aOpacity;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Scale factor: world-space scale -> screen-space pixel size.
    // Multiply by a tunable constant for visual comfort.
    float screenScale = aScale * 800.0 / (-mvPosition.z);
    gl_PointSize = clamp(screenScale, 1.0, 64.0);
  }
`;

// Fragment shader: soft Gaussian falloff per point sprite.
const fragmentShader = `
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    // gl_PointCoord is [0,1]x[0,1] within the point sprite.
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);

    // Discard pixels outside the unit circle for efficiency.
    if (r2 > 1.0) discard;

    // Gaussian falloff.
    float alpha = exp(-4.0 * r2) * vOpacity;

    gl_FragColor = vec4(vColor, alpha);
  }
`;

/**
 * Create a Three.js Points object from parsed splat data.
 */
export function createSplatPoints(splatData) {
  const { positions, colors, scales, opacities, count } = splatData;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('aScale', new THREE.Float32BufferAttribute(scales, 1));
  geometry.setAttribute('aOpacity', new THREE.Float32BufferAttribute(opacities, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    vertexColors: true,
    transparent: true,
    depthWrite: false,       // needed for proper alpha blending
    blending: THREE.NormalBlending,
  });

  const points = new THREE.Points(geometry, material);
  return points;
}
