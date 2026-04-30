import React, { useRef, useEffect, useState } from 'react';

function hexToRGB(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return[r, g, b];
}

function getTextColorForBg(hex: string) {
  const [r, g, b] = hexToRGB(hex);
  const luma = r * 0.299 + g * 0.587 + b * 0.114;
  return luma > 0.5 ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.9)';
}

const VERTEX_SHADER = `#version 300 es
precision mediump float;
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FSHADER_PASS1 = `#version 300 es
precision highp float;
uniform float u_time;
uniform vec3 u_colors[4];
uniform float u_distortion;
uniform float u_swirl;
uniform float u_meshBlur;
uniform float u_mixing;
uniform vec2 u_resolution;

uniform float u_baseType;
uniform float u_metaballsCount;

out vec4 fragColor;
in vec2 v_uv;

mat2 rotate(float th) { return mat2(cos(th), sin(th), -sin(th), cos(th)); }

vec2 getPosition(int i, float t) {
  float a = float(i) * .37;
  float b = .6 + fract(float(i) / 3.) * .9;
  float c = .8 + fract(float(i + 1) / 4.);
  return .5 + .5 * vec2(sin(t * b + a), cos(t * c + a * 1.5));
}

void main() {
  vec2 uv = v_uv;
  float minDim = min(u_resolution.x, u_resolution.y);
  vec2 aspectVec = (minDim > 0.0) ? u_resolution / minDim : vec2(1.0);
  
  vec2 uvMap = (uv - 0.5) * aspectVec + 0.5;

  float t = .5 * (u_time + 41.5);
  float radius = smoothstep(0., 1., length(uvMap - .5));
  float center = 1. - radius;
  
  for (float i = 1.; i <= 2.; i++) {
    uvMap.x += u_distortion * center / i * sin(t + i * .4 * smoothstep(.0, 1., uvMap.y)) * cos(.2 * t + i * 2.4 * smoothstep(.0, 1., uvMap.y));
    uvMap.y += u_distortion * center / i * cos(t + i * 2. * smoothstep(.0, 1., uvMap.x));
  }
  
  vec2 uvRotated = uvMap - .5;
  uvRotated = rotate(-3. * u_swirl * radius) * uvRotated + .5;

  if (u_baseType < 0.5) {
      vec3 color = vec3(0.);
      float totalWeight = 0.;
      
      float exponent = mix(6.0, 1.5, u_meshBlur);
      
      for (int i = 0; i < 4; i++) {
        vec2 pos = getPosition(i, t);
        pos = (pos - 0.5) * aspectVec + 0.5;
        
        float dist = length(uvRotated - pos);
        dist = pow(dist, exponent);
        float weight = 1. / (dist + 1e-3);
        color += u_colors[i] * weight;
        totalWeight += weight;
      }
      fragColor = vec4(color / max(1e-4, totalWeight), 1.0);
  } else {
      float tVal = 0.0;
      vec3 mColor = vec3(0.0);
      float mWeight = 0.0;
      
      vec3 bgColor = u_colors[3]; // color4 is background
      int mCount = int(u_metaballsCount);
      
      float sizes[3];
      sizes[0] = 1.0;
      sizes[1] = 1.3;
      sizes[2] = 0.7;
      
      for (int i = 0; i < 3; i++) {
          if (i >= mCount) break;
          
          vec2 pos = getPosition(i, t);
          pos = (pos - 0.5) * aspectVec + 0.5;
          
          float d = length(uvRotated - pos);
          float radius = mix(0.1, 0.6, u_meshBlur) * sizes[i]; 
          float e = exp(-(d * d) / (radius * radius));
          
          tVal += e;
          mColor += u_colors[i] * e;
          mWeight += e;
      }
      
      if (mWeight > 0.0) {
          mColor /= mWeight;
      }
      
      float edge = 0.5;
      float borderSoftness = mix(0.01, 0.8, u_mixing);
      float alpha = smoothstep(edge - borderSoftness, edge + borderSoftness, tVal);
      
      fragColor = vec4(mix(bgColor, mColor, alpha), 1.0);
  }
}
`;

const FSHADER_PASS2 = `#version 300 es
precision highp float;

uniform sampler2D u_image;
uniform float u_size; 
uniform float u_contrast; 
uniform vec2 u_resolution;
uniform float u_gridNoise;
uniform float u_softness;

in vec2 v_uv;
out vec4 fragColor;

const float cosC = 0.9659258;  const float sinC = 0.2588190;
const float cosM = 0.2588190;  const float sinM = 0.9659258;
const float cosY = 1.0;        const float sinY = 0.0;
const float cosK = 0.7071068;  const float sinK = 0.7071068;

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx+p3.yz)*p3.zy);
}

vec4 RGBAtoCMYK(vec3 rgb) {
  float k = 1. - max(max(rgb.r, rgb.g), rgb.b);
  float denom = 1. - k;
  vec3 cmy = vec3(0.);
  if (denom > 1e-5) cmy = (1. - rgb - vec3(k)) / denom;
  return vec4(cmy, k);
}

vec2 cellCenterPos(vec2 uv, vec2 cellOffset, float channelIdx) {
  vec2 cellCenter = floor(uv) + 0.5 + cellOffset;
  return cellCenter + (hash22(cellCenter + channelIdx * 50.0) - 0.5) * u_gridNoise;
}

void colorMask(vec2 pos, vec2 cellCenter, float rad, float generalComp, inout float outMask) {
  float dist = length(pos - cellCenter);
  
  float radius = max(0., rad * (1.15 + generalComp));
  
  float mask = 1. - smoothstep(0., radius, dist);
  mask = smoothstep(0.5 - 0.5 * u_softness, 0.51 + 0.49 * u_softness, mask);
  mask *= mix(1., mix(0.5, 1.0, 1.5 * radius), u_softness);
  
  outMask += mask;
}

void main() {
  vec2 uv = v_uv;
  
  float minDim = min(u_resolution.x, u_resolution.y);
  if(minDim <= 0.0) minDim = 1.0;
  
  float cellsPerSide = mix(400.0, 10.0, pow(u_size, 0.7));
  float cellSizePx = minDim / cellsPerSide; 
  
  // Convert standard uv (0 to 1) into aspect-ratio accurate grid based on pixels
  vec2 px = uv * u_resolution;
  vec2 uvGrid = px / cellSizePx;

  vec2 uvC = mat2(cosC, sinC, -sinC, cosC) * uvGrid;
  vec2 uvM = mat2(cosM, sinM, -sinM, cosM) * uvGrid;
  vec2 uvY = mat2(cosY, sinY, -sinY, cosY) * uvGrid;
  vec2 uvK = mat2(cosK, sinK, -sinK, cosK) * uvGrid;

  vec3 tex = texture(u_image, uv).rgb;
  tex = clamp((tex - 0.5) * u_contrast + 0.5, 0.0, 1.0);
  vec4 cmyk = RGBAtoCMYK(tex);
  
  float generalComp = 0.1 * u_softness + 0.1 * u_gridNoise;

  vec4 outMask = vec4(0.);
  for (float dy = -1.; dy <= 1.; dy++) {
    for (float dx = -1.; dx <= 1.; dx++) {
      vec2 offset = vec2(dx, dy);
      colorMask(uvC, cellCenterPos(uvC, offset, 0.), cmyk.x, generalComp, outMask[0]);
      colorMask(uvM, cellCenterPos(uvM, offset, 1.), cmyk.y, generalComp, outMask[1]);
      colorMask(uvY, cellCenterPos(uvY, offset, 2.), cmyk.z, generalComp, outMask[2]);
      colorMask(uvK, cellCenterPos(uvK, offset, 3.), cmyk.w, generalComp, outMask[3]);
    }
  }

  float C = clamp(outMask[0], 0.0, 1.0);
  float M = clamp(outMask[1], 0.0, 1.0);
  float Y = clamp(outMask[2], 0.0, 1.0);
  float K = clamp(outMask[3], 0.0, 1.0);

  vec3 ink = vec3(1.0);
  ink *= mix(vec3(1.), vec3(0.,0.,0.), K);
  ink *= mix(vec3(1.), vec3(0.,1.,1.), C);
  ink *= mix(vec3(1.), vec3(1.,0.,1.), M);
  ink *= mix(vec3(1.), vec3(1.,1.,0.), Y);

  fragColor = vec4(ink, 1.0);
}
`;

const FSHADER_PASS2_DOTS = `#version 300 es
precision highp float;

uniform sampler2D u_image;
uniform vec2 u_resolution;
uniform vec3 u_colorFront;
uniform vec3 u_colorBack;
uniform float u_size;
uniform float u_radius;
uniform float u_contrast;
uniform float u_grid; // 0.0 = square, 1.0 = hex
uniform float u_type; // 0.0 = classic, 1.0 = gooey

in vec2 v_uv;
out vec4 fragColor;

float sst(float edge0, float edge1, float x) { 
  return smoothstep(edge0, edge1, x); 
}

float getCircle(vec2 uv, float r, float baseR) {
  r = mix(.25 * baseR, 0., r);
  float d = length(uv - .5);
  float aa = fwidth(d);
  return 1. - smoothstep(r - aa, r + aa, d);
}

float getGooeyBall(vec2 uv, float r, float baseR) {
  float d = length(uv - .5);
  float sizeRadius = (u_grid == 1.) ? .42 : .3;
  sizeRadius = mix(sizeRadius * baseR, 0., r);
  d = 1. - sst(0., sizeRadius, d);
  return pow(d, 2. + baseR);
}

float sigmoid(float x, float k) { 
  return 1.0 / (1.0 + exp(-k * (x - 0.5))); 
}

float getLumAtPx(vec2 uv, float contrast) {
  vec4 tex = texture(u_image, uv);
  vec3 color = vec3(
    sigmoid(tex.r, contrast),
    sigmoid(tex.g, contrast),
    sigmoid(tex.b, contrast)
  );
  return dot(vec3(0.2126, 0.7152, 0.0722), color);
}

void main() {
  float stepMultiplier = (u_type == 0.) ? 2. : 6.;

  float cellsPerSide = mix(300., 7., pow(u_size, .7));
  cellsPerSide /= stepMultiplier;
  float cellSizeY = 1. / cellsPerSide;
  
  float aspect = u_resolution.x / u_resolution.y;
  vec2 pad = cellSizeY * vec2(1. / aspect, 1.);
  
  if (u_type == 1. && u_grid == 1.) {
    pad *= .7;
  }

  vec2 uv = v_uv;
  uv -= vec2(.5);
  uv /= pad;

  float contrast = mix(0., 15., pow(u_contrast, 1.5));
  float baseRadius = u_radius;

  float totalShape = 0.;
  float stepSize = 1. / stepMultiplier;
  for (float x = -0.5; x < 0.5; x += stepSize) {
    for (float y = -0.5; y < 0.5; y += stepSize) {
      vec2 offset = vec2(x, y);

      if (u_grid == 1.) {
        float rowIndex = floor((y + .5) / stepSize);
        float colIndex = floor((x + .5) / stepSize);
        if (u_type == 1.) {
          if (mod(rowIndex + colIndex, 2.) == 1.) {
            continue;
          }
        } else {
          if (mod(rowIndex, 2.) == 1.) {
            offset.x += .5 * stepSize;
          }
        }
      }

      vec2 p = uv + offset;
      vec2 uv_i = floor(p);
      vec2 uv_f = fract(p);
      vec2 samplingUV = (uv_i + .5 - offset) * pad + vec2(.5);
      
      float lum = getLumAtPx(clamp(samplingUV, 0.0, 1.0), contrast);
      float ball = 0.;
      if (u_type == 0.) {
        ball = getCircle(uv_f, lum, baseRadius);
      } else if (u_type == 1.) {
        ball = getGooeyBall(uv_f, lum, baseRadius);
      }
      
      totalShape += ball;
    }
  }

  float finalShape = 0.;
  if (u_type == 0.) {
    finalShape = min(1., totalShape);
  } else if (u_type == 1.) {
    float aa = fwidth(totalShape);
    float th = .5;
    finalShape = smoothstep(th - aa, th + aa, totalShape);
  }

  fragColor = vec4(mix(u_colorBack, u_colorFront, finalShape), 1.0);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(prog));
    return null;
  }
  return prog;
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
}

function SliderControl({ label, value, min, max, step, onChange }: SliderControlProps) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[11px] font-mono text-white/60">
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <div className="custom-slider-wrapper" style={{ "--percent": `${percent}%` } as React.CSSProperties}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="custom-slider-native"
        />
        <span className="custom-slider-thumb-slot" aria-hidden="true">
          <span className="custom-slider-thumb-visual" />
        </span>
      </div>
    </div>
  );
}

function hexToHsv(hex: string) {
  if (hex.startsWith('#')) hex = hex.slice(1);
  if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
  let r = parseInt(hex.substring(0, 2), 16) / 255;
  let g = parseInt(hex.substring(2, 4), 16) / 255;
  let b = parseInt(hex.substring(4, 6), 16) / 255;

  let max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0, v = max;

  let d = max - min;
  s = max === 0 ? 0 : d / max;

  if (max === min) {
    h = 0; // achromatic
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }

  return { h: h * 360, s: s * 100, v: v * 100 };
}

function hsvToHex(h: number, s: number, v: number) {
  let hNormalized = (h % 360) / 360;
  if (hNormalized < 0) hNormalized += 1;
  let sNormalized = Math.max(0, Math.min(100, s)) / 100;
  let vNormalized = Math.max(0, Math.min(100, v)) / 100;

  let r = 0, g = 0, b = 0;
  let i = Math.floor(hNormalized * 6);
  let f = hNormalized * 6 - i;
  let p = vNormalized * (1 - sNormalized);
  let q = vNormalized * (1 - f * sNormalized);
  let t = vNormalized * (1 - (1 - f) * sNormalized);

  switch (i % 6) {
    case 0: r = vNormalized; g = t; b = p; break;
    case 1: r = q; g = vNormalized; b = p; break;
    case 2: r = p; g = vNormalized; b = t; break;
    case 3: r = p; g = q; b = vNormalized; break;
    case 4: r = t; g = p; b = vNormalized; break;
    case 5: r = vNormalized; g = p; b = q; break;
  }

  const toHex = (x: number) => {
    const hex = Math.round(x * 255).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function HSVColorPicker({ color, onChange }: { color: string, onChange: (c: string) => void }) {
  const [hsv, setHsv] = useState(() => hexToHsv(color));

  useEffect(() => {
    const currentHex = hsvToHex(hsv.h, hsv.s, hsv.v);
    if (color.toLowerCase() !== currentHex.toLowerCase()) {
      setHsv(hexToHsv(color));
    }
  }, [color, hsv]);

  const handleH = (h: number) => {
    const newHsv = { ...hsv, h };
    setHsv(newHsv);
    onChange(hsvToHex(newHsv.h, newHsv.s, newHsv.v));
  };
  const handleS = (s: number) => {
    const newHsv = { ...hsv, s };
    setHsv(newHsv);
    onChange(hsvToHex(newHsv.h, newHsv.s, newHsv.v));
  };
  const handleV = (v: number) => {
    const newHsv = { ...hsv, v };
    setHsv(newHsv);
    onChange(hsvToHex(newHsv.h, newHsv.s, newHsv.v));
  };

  const offset = 'var(--slider-half)';
  const size = 'var(--slider-height)';

  const sColor0 = hsvToHex(hsv.h, 0, hsv.v);
  const sColor100 = hsvToHex(hsv.h, 100, hsv.v);
  const vColor0 = '#000000';
  const vColor100 = hsvToHex(hsv.h, hsv.s, 100);

  return (
    <div className="space-y-5">
      {/* H */}
      <div className="space-y-2">
        <div className="flex justify-between text-[11px] font-mono text-white/60">
            <span>Hue</span><span>{Math.round(hsv.h)}°</span>
        </div>
        <div className="custom-slider-wrapper" style={{
          background: `linear-gradient(to right, #F09081 0px, #F09081 ${offset}, #C9CA46 calc(${offset} + (100% - ${size}) * 0.1666), #74CF6D calc(${offset} + (100% - ${size}) * 0.3333), #2CC5C5 calc(${offset} + (100% - ${size}) * 0.5), #87ADFA calc(${offset} + (100% - ${size}) * 0.6666), #D792D4 calc(${offset} + (100% - ${size}) * 0.8333), #F09081 calc(100% - ${offset}), #F09081 100%)`,
          '--percent': `${(hsv.h / 360) * 100}%`
        } as React.CSSProperties}>
          <input type="range" min="0" max="360" step="1" value={hsv.h} onChange={(e) => handleH(parseFloat(e.target.value))} className="custom-slider-native" />
          <span className="custom-slider-thumb-slot" aria-hidden="true">
            <span className="custom-slider-thumb-visual" />
          </span>
        </div>
      </div>

      {/* S */}
      <div className="space-y-2">
        <div className="flex justify-between text-[11px] font-mono text-white/60">
            <span>Saturation</span><span>{Math.round(hsv.s)}%</span>
        </div>
        <div className="custom-slider-wrapper" style={{
          background: `linear-gradient(to right, ${sColor0} 0px, ${sColor0} ${offset}, ${sColor100} calc(100% - ${offset}), ${sColor100} 100%)`,
          '--percent': `${hsv.s}%`
        } as React.CSSProperties}>
          <input type="range" min="0" max="100" step="1" value={hsv.s} onChange={(e) => handleS(parseFloat(e.target.value))} className="custom-slider-native" />
          <span className="custom-slider-thumb-slot" aria-hidden="true">
            <span className="custom-slider-thumb-visual" />
          </span>
        </div>
      </div>

      {/* V */}
      <div className="space-y-2">
        <div className="flex justify-between text-[11px] font-mono text-white/60">
            <span>Value</span><span>{Math.round(hsv.v)}%</span>
        </div>
        <div className="custom-slider-wrapper" style={{
          background: `linear-gradient(to right, ${vColor0} 0px, ${vColor0} ${offset}, ${vColor100} calc(100% - ${offset}), ${vColor100} 100%)`,
          '--percent': `${hsv.v}%`
        } as React.CSSProperties}>
          <input type="range" min="0" max="100" step="1" value={hsv.v} onChange={(e) => handleV(parseFloat(e.target.value))} className="custom-slider-native" />
          <span className="custom-slider-thumb-slot" aria-hidden="true">
            <span className="custom-slider-thumb-visual" />
          </span>
        </div>
      </div>
    </div>
  );
}

function ToggleButtonGroup({ 
  label, 
  options, 
  value, 
  onChange, 
  disabled = false 
}: { 
  label: string; 
  options: { label: string | number; value: any }[]; 
  value: any; 
  onChange: (val: any) => void; 
  disabled?: boolean; 
}) {
  return (
    <div className={`space-y-2 flex-1 transition-opacity duration-300 ${disabled ? 'opacity-30 pointer-events-none' : 'opacity-100'}`}>
      <div className="text-[11px] font-mono text-white/60">{label}</div>
      <div className="flex bg-white/10 p-0 rounded-full h-8">
        {options.map((opt, i) => (
          <button 
            key={i}
            className={`flex-1 flex items-center justify-center h-full text-[10px] uppercase tracking-wider rounded-full transition-all ${value === opt.value ? 'bg-white/20 text-white shadow-sm' : 'text-white/40 hover:text-white/80'}`}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function HalftoneMeshGradient() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const[mode, setMode] = useState<'cmyk' | 'dots'>('cmyk');
  const[color1, setColor1] = useState('#6688d6');
  const [color2, setColor2] = useState('#1e1782');
  const [color3, setColor3] = useState('#291cd9');
  const [color4, setColor4] = useState('#22f73a');
  
  const[dotColorFront, setDotColorFront] = useState('#CDD6DC');
  const [dotColorBack, setDotColorBack] = useState('#000000');
  const [dotType, setDotType] = useState<number>(1);
  const[dotGrid, setDotGrid] = useState<number>(0);
  const [dotRadius, setDotRadius] = useState<number>(0.8);
  const [dotContrast, setDotContrast] = useState<number>(0.5);

  const[animationSpeed, setAnimationSpeed] = useState(0.6);
  const [meshDistortion, setMeshDistortion] = useState(0.8);
  const [meshSwirl, setMeshSwirl] = useState(0.1);
  const [meshBlur, setMeshBlur] = useState(0.5);
  const [mixing, setMixing] = useState(0.5);
  const[dotSize, setDotSize] = useState(0.4);
  const [contrast, setContrast] = useState(1.15);
  const [gridNoise, setGridNoise] = useState(0.2);
  const[softness, setSoftness] = useState(0.0);

  const [baseType, setBaseType] = useState<number>(0);
  const [metaballsCount, setMetaballsCount] = useState<number>(3);

  const [activeColorIdx, setActiveColorIdx] = useState<number | null>(null);
  const colorsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (activeColorIdx !== null && colorsRef.current && !colorsRef.current.contains(e.target as Node)) {
        setActiveColorIdx(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [activeColorIdx]);

  useEffect(() => {
    if (activeColorIdx === null || !colorsRef.current) return;

    // A small timeout to let the popup render before measuring
    const timer = setTimeout(() => {
      if (!colorsRef.current) return;
      const el = colorsRef.current;
      const isDesktop = window.matchMedia('(min-width: 768px)').matches;

      if (!isDesktop) {
        const stickyHeight = canvasRef.current?.parentElement?.offsetHeight || 0;
        const rect = el.getBoundingClientRect();
        // Leave a 16px visual gap below the sticky canvas
        const targetViewportY = stickyHeight + 16;
        const scrollDelta = rect.top - targetViewportY;
        
        window.scrollBy({ top: scrollDelta, behavior: 'smooth' });
      }
    }, 50);

    return () => clearTimeout(timer);
  }, [activeColorIdx]);

  const fpsRef = useRef<HTMLSpanElement>(null);
  const timeInfoRef = useRef<HTMLParagraphElement>(null);
  const resolutionInfoRef = useRef<HTMLSpanElement>(null);

  const controlsRef = useRef({
    color1, color2, color3, color4,
    animationSpeed, meshDistortion, meshSwirl, meshBlur, mixing, dotSize, contrast, gridNoise, softness,
    mode, dotColorFront, dotColorBack, dotType, dotGrid, dotRadius, dotContrast,
    baseType, metaballsCount
  });

  useEffect(() => {
    controlsRef.current = {
      color1, color2, color3, color4,
      animationSpeed, meshDistortion, meshSwirl, meshBlur, mixing, dotSize, contrast, gridNoise, softness,
      mode, dotColorFront, dotColorBack, dotType, dotGrid, dotRadius, dotContrast,
      baseType, metaballsCount
    };
  },[
    color1, color2, color3, color4,
    animationSpeed, meshDistortion, meshSwirl, meshBlur, mixing, dotSize, contrast, gridNoise, softness,
    mode, dotColorFront, dotColorBack, dotType, dotGrid, dotRadius, dotContrast,
    baseType, metaballsCount
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const gl = canvas.getContext('webgl2', { 
      antialias: false, 
      preserveDrawingBuffer: true
    });
    if (!gl) {
      console.error('WebGL 2 is not supported by your browser.');
      return;
    }

    const prog1 = createProgram(gl, VERTEX_SHADER, FSHADER_PASS1);
    const prog2Cmyk = createProgram(gl, VERTEX_SHADER, FSHADER_PASS2);
    const prog2Dots = createProgram(gl, VERTEX_SHADER, FSHADER_PASS2_DOTS);
    if (!prog1 || !prog2Cmyk || !prog2Dots) return;

    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);
    
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    let fboParams = { width: 0, height: 0 };
    const texture = gl.createTexture();
    const fbo = gl.createFramebuffer();

    function resizeFBO(width: number, height: number) {
      if (!texture || !fbo || !gl) return;
      if (width === fboParams.width && height === fboParams.height) return;
      
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      fboParams = { width, height };

      if (resolutionInfoRef.current) {
        resolutionInfoRef.current.innerText = `W:${width} H:${height}`;
      }
    }

    const locTime = gl.getUniformLocation(prog1, 'u_time');
    const locColors = gl.getUniformLocation(prog1, 'u_colors');
    const locDistortion = gl.getUniformLocation(prog1, 'u_distortion');
    const locSwirl = gl.getUniformLocation(prog1, 'u_swirl');
    const locMeshBlur = gl.getUniformLocation(prog1, 'u_meshBlur');
    const locMixing = gl.getUniformLocation(prog1, 'u_mixing');
    const locRes1 = gl.getUniformLocation(prog1, 'u_resolution');
    const locBaseType = gl.getUniformLocation(prog1, 'u_baseType');
    const locMetaballsCount = gl.getUniformLocation(prog1, 'u_metaballsCount');

    const locImageCmyk = gl.getUniformLocation(prog2Cmyk, 'u_image');
    const locSizeCmyk = gl.getUniformLocation(prog2Cmyk, 'u_size');
    const locContrastCmyk = gl.getUniformLocation(prog2Cmyk, 'u_contrast');
    const locRes2Cmyk = gl.getUniformLocation(prog2Cmyk, 'u_resolution');
    const locGridNoiseCmyk = gl.getUniformLocation(prog2Cmyk, 'u_gridNoise');
    const locSoftnessCmyk = gl.getUniformLocation(prog2Cmyk, 'u_softness');

    const locImageDots = gl.getUniformLocation(prog2Dots, 'u_image');
    const locRes2Dots = gl.getUniformLocation(prog2Dots, 'u_resolution');
    const locColorFrontDots = gl.getUniformLocation(prog2Dots, 'u_colorFront');
    const locColorBackDots = gl.getUniformLocation(prog2Dots, 'u_colorBack');
    const locSizeDots = gl.getUniformLocation(prog2Dots, 'u_size');
    const locRadiusDots = gl.getUniformLocation(prog2Dots, 'u_radius');
    const locContrastDots = gl.getUniformLocation(prog2Dots, 'u_contrast');
    const locGridDots = gl.getUniformLocation(prog2Dots, 'u_grid');
    const locTypeDots = gl.getUniformLocation(prog2Dots, 'u_type');

    let animationFrameId: number;
    let lastTime = performance.now();
    let timeAccumulator = 0;
    
    let frameCount = 0;
    let lastFpsTime = lastTime;

    function render() {
      if (!gl || !canvas) return;

      const now = performance.now();
      const dt = (now - lastTime) / 1000.0;
      lastTime = now;

      frameCount++;
      if (now - lastFpsTime >= 1000) {
        const fps = (frameCount * 1000) / (now - lastFpsTime);
        if (fpsRef.current) {
          fpsRef.current.innerText = `${fps.toFixed(1)} FPS`;
        }
        frameCount = 0;
        lastFpsTime = now;
      }

      const c = controlsRef.current;
      timeAccumulator += dt * c.animationSpeed;

      if (timeInfoRef.current) {
        timeInfoRef.current.innerText = `u_time: ${timeAccumulator.toFixed(3)}`;
      }

      const dpr = window.devicePixelRatio || 1;
      const displayWidth = Math.floor(canvas.clientWidth * dpr);
      const displayHeight = Math.floor(canvas.clientHeight * dpr);

      if (displayWidth === 0 || displayHeight === 0) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      let needResize = false;
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        needResize = true;
      }

      if (needResize || fboParams.width === 0) {
        resizeFBO(displayWidth, displayHeight);
      }

      gl.viewport(0, 0, canvas.width, canvas.height);

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.useProgram(prog1);
      gl.bindVertexArray(vao);

      gl.uniform1f(locTime, timeAccumulator);
      if (locRes1 !== null) gl.uniform2f(locRes1, canvas.width, canvas.height);
      
      let rgb1, rgb2, rgb3, rgb4;
      if (c.mode === 'cmyk') {
        rgb1 = hexToRGB(c.color1);
        rgb2 = hexToRGB(c.color2);
        rgb3 = hexToRGB(c.color3);
        rgb4 = hexToRGB(c.color4);
      } else {
        if (c.baseType === 1) { 
          // For Gooey base in Dots mode (Gradient B):
          rgb1 = [0.0, 0.0, 0.0];
          rgb2 = [0.0, 0.0, 0.0];
          rgb3 = [0.0, 0.0, 0.0];
          rgb4 = [1.0, 1.0, 1.0];
        } else {
          // For Mesh base in Dots mode (Gradient A):
          rgb1 = [0.0, 0.0, 0.0];
          rgb2 = [0.33, 0.33, 0.33];
          rgb3 = [0.66, 0.66, 0.66];
          rgb4 = [1.0, 1.0, 1.0];
        }
      }
      
      gl.uniform3fv(locColors, new Float32Array([...rgb1, ...rgb2, ...rgb3, ...rgb4]));
      
      gl.uniform1f(locDistortion, c.baseType === 1 ? 0 : c.meshDistortion);
      gl.uniform1f(locSwirl, c.baseType === 1 ? 0 : c.meshSwirl);
      gl.uniform1f(locMeshBlur, c.meshBlur);
      if (locMixing) gl.uniform1f(locMixing, c.mixing);
      if (locBaseType) gl.uniform1f(locBaseType, c.baseType);
      if (locMetaballsCount) gl.uniform1f(locMetaballsCount, c.metaballsCount);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);

      if (c.mode === 'cmyk') {
        gl.useProgram(prog2Cmyk);
        gl.bindVertexArray(vao);
        
        if (locImageCmyk !== null) gl.uniform1i(locImageCmyk, 0);
        gl.uniform1f(locSizeCmyk, c.dotSize);
        gl.uniform1f(locContrastCmyk, c.contrast);
        if (locGridNoiseCmyk !== null) gl.uniform1f(locGridNoiseCmyk, c.gridNoise);
        if (locSoftnessCmyk !== null) gl.uniform1f(locSoftnessCmyk, c.softness);
        gl.uniform2f(locRes2Cmyk, canvas.width, canvas.height);
        
      } else {
        gl.useProgram(prog2Dots);
        gl.bindVertexArray(vao);
        
        if (locImageDots !== null) gl.uniform1i(locImageDots, 0);
        gl.uniform2f(locRes2Dots, canvas.width, canvas.height);
        
        const frontRgb = hexToRGB(c.dotColorFront);
        const backRgb = hexToRGB(c.dotColorBack);
        gl.uniform3fv(locColorFrontDots, new Float32Array(frontRgb));
        gl.uniform3fv(locColorBackDots, new Float32Array(backRgb));
        
        gl.uniform1f(locSizeDots, c.dotSize);
        if (locRadiusDots) gl.uniform1f(locRadiusDots, c.dotRadius);
        gl.uniform1f(locContrastDots, c.dotContrast);
        if (locGridDots) gl.uniform1f(locGridDots, c.dotGrid);
        if (locTypeDots) gl.uniform1f(locTypeDots, c.dotType);
      }

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      animationFrameId = requestAnimationFrame(render);
    }

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      gl.deleteProgram(prog1);
      gl.deleteProgram(prog2Cmyk);
      gl.deleteProgram(prog2Dots);
      gl.deleteBuffer(vbo);
      gl.deleteVertexArray(vao);
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(fbo);
    };
  },[]);

  return (
    <div className="w-full flex-1 bg-[#111113] text-[#E0E0E0] font-sans flex flex-col md:flex-row md:overflow-hidden">
      <div className="w-full aspect-square max-h-[60dvh] md:max-h-none md:aspect-auto md:h-full md:flex-1 relative bg-[#000] overflow-hidden shrink-0 group sticky top-0 z-30 shadow-[0_0_20px_rgba(17,17,19,0.9)] md:shadow-none">
        <canvas 
          ref={canvasRef} 
          className="absolute inset-0 block w-full h-full z-0"
        />
      </div>

      <div className="w-full md:w-[340px] shrink-0 md:h-full bg-[#111113] md:border-l border-white/5 flex flex-col relative z-10">
        
        <div className="flex-1 p-6 pb-safe space-y-8 md:overflow-y-auto custom-scrollbar pb-12 md:pb-6">
          
          <header className="mb-6">
            <h1 className="text-3xl font-light text-white/90 leading-[1]">
              Halftone<br/>
              <span className="font-extrabold italic">Gradient</span>
            </h1>
            <div className="mt-2 flex gap-4 text-[10px] font-mono text-white/40">
              <span ref={fpsRef}>60.0 FPS</span>
              <span ref={resolutionInfoRef}>W:0 H:0</span>
            </div>
          </header>

          <div className="flex bg-white/10 p-0 rounded-full mb-6">
            <button 
              className={`flex-1 flex items-center justify-center h-9 text-[11px] uppercase tracking-wider rounded-full transition-all font-semibold ${mode === 'cmyk' ? 'bg-white/20 text-white shadow-sm' : 'text-white/40 hover:text-white/80'}`}
              onClick={() => { setMode('cmyk'); setActiveColorIdx(null); }}
            >
              CMYK
            </button>
            <button 
              className={`flex-1 flex items-center justify-center h-9 text-[11px] uppercase tracking-wider rounded-full transition-all font-semibold ${mode === 'dots' ? 'bg-white/20 text-white shadow-sm' : 'text-white/40 hover:text-white/80'}`}
              onClick={() => { setMode('dots'); setActiveColorIdx(null); }}
            >
              MONO
            </button>
          </div>

          <div className="relative mb-6" ref={colorsRef}>
            {mode === 'cmyk' ? (
              <div 
                className="grid gap-2" 
                style={{ 
                  gridTemplateColumns: `repeat(${baseType === 0 ? 4 : (1 + metaballsCount)}, minmax(0, 1fr))` 
                }}
              >
                {[
                  { c: color1, set: setColor1, visible: true },
                  { c: color2, set: setColor2, visible: baseType === 0 || metaballsCount >= 2 },
                  { c: color3, set: setColor3, visible: baseType === 0 || metaballsCount >= 3 },
                  { c: color4, set: setColor4, visible: true },
                ].map((item, i) => item.visible ? (
                  <button
                    key={i}
                    className={`h-10 w-full rounded ring-offset-0 flex flex-col cursor-pointer transition-all shadow-lg overflow-hidden relative ${
                      activeColorIdx === i ? 'ring-1 ring-white/90 scale-105 z-10 shadow-white/20' : 'ring-1 ring-white/15 hover:scale-105 hover:ring-white/40'
                    }`}
                    style={{ backgroundColor: item.c }}
                    onClick={() => setActiveColorIdx(activeColorIdx === i ? null : i)}
                  >
                    {baseType === 1 && i === 3 && (
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-[10px] uppercase font-bold" style={{ color: getTextColorForBg(item.c) }}>
                        BG
                      </div>
                    )}
                  </button>
                ) : null)}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {[
                  { c: dotColorFront, set: setDotColorFront, label: '' },
                  { c: dotColorBack, set: setDotColorBack, label: 'BG' },
                ].map((item, i) => (
                  <button
                    key={i}
                    className={`h-10 w-full flex-1 rounded ring-offset-0 flex flex-col cursor-pointer transition-all shadow-lg overflow-hidden relative ${
                      activeColorIdx === i ? 'ring-1 ring-white/90 scale-105 z-10 shadow-white/20' : 'ring-1 ring-white/15 hover:scale-105 hover:ring-white/40'
                    }`}
                    style={{ backgroundColor: item.c }}
                    onClick={() => setActiveColorIdx(activeColorIdx === i ? null : i)}
                  >
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-[9px] uppercase font-bold" style={{ color: getTextColorForBg(item.c) }}>
                      {item.label}
                    </div>
                  </button>
                ))}
              </div>
            )}

            {activeColorIdx !== null && (
              <div className="absolute top-[52px] left-0 right-0 p-5 bg-[#1A1A1D] border border-white/10 rounded-xl shadow-[0_16px_48px_rgba(0,0,0,1)] z-50">
                {mode === 'cmyk' ? (
                  <HSVColorPicker 
                     color={[color1, color2, color3, color4][activeColorIdx]}
                     onChange={(c) => {
                       if (activeColorIdx === 0) setColor1(c);
                       else if (activeColorIdx === 1) setColor2(c);
                       else if (activeColorIdx === 2) setColor3(c);
                       else if (activeColorIdx === 3) setColor4(c);
                     }}
                  />
                ) : (
                  <HSVColorPicker 
                     color={[dotColorFront, dotColorBack][activeColorIdx]}
                     onChange={(c) => {
                       if (activeColorIdx === 0) setDotColorFront(c);
                       else if (activeColorIdx === 1) setDotColorBack(c);
                     }}
                  />
                )}
              </div>
            )}
          </div>

          <section>
            <h3 className="text-[10px] font-mono uppercase tracking-widest text-white/90 mb-4 flex items-center gap-2">
              <span className={`w-1 h-3 ${mode === 'cmyk' ? 'bg-[#35D926]' : 'bg-[#CDD6DC]'}`}></span> Properties
            </h3>
            <div className="space-y-4">
              <ToggleButtonGroup
                label="Shape"
                value={baseType}
                onChange={setBaseType}
                options={[
                  { label: 'Aurora', value: 0 },
                  { label: 'Blobs', value: 1 }
                ]}
              />
              {baseType === 1 && (
                <ToggleButtonGroup
                  label="Blob Count"
                  value={metaballsCount}
                  onChange={setMetaballsCount}
                  options={[
                    { label: 1, value: 1 },
                    { label: 2, value: 2 },
                    { label: 3, value: 3 }
                  ]}
                />
              )}
              {baseType === 0 && (
                <>
                  <SliderControl label="Distortion" value={meshDistortion} min={0} max={2} step={0.01} onChange={setMeshDistortion} />
                  <SliderControl label="Swirl" value={meshSwirl} min={0} max={1} step={0.01} onChange={setMeshSwirl} />
                </>
              )}
              {baseType === 1 && (
                <SliderControl label="Blob Softness" value={mixing} min={0} max={1} step={0.01} onChange={setMixing} />
              )}
              <SliderControl label="Spread" value={meshBlur} min={0} max={1} step={0.01} onChange={setMeshBlur} />
              <SliderControl label="Animation Speed" value={animationSpeed} min={0} max={3} step={0.01} onChange={setAnimationSpeed} />
            </div>
          </section>
          
          {mode === 'cmyk' ? (
            <section>
              <h3 className="text-[10px] font-mono uppercase tracking-widest text-white/90 mb-4 flex items-center gap-2">
                <span className="w-1 h-3 bg-[#10CCE0]"></span> CMYK Engine
              </h3>
              <div className="space-y-4">
                <SliderControl label="Dot Size" value={dotSize} min={0.01} max={1} step={0.01} onChange={setDotSize} />
                <SliderControl label="Grid Noise" value={gridNoise} min={0} max={1} step={0.01} onChange={setGridNoise} />
                <SliderControl label="Dot Softness" value={softness} min={0} max={1} step={0.01} onChange={setSoftness} />
                <SliderControl label="Contrast" value={contrast} min={0} max={2} step={0.01} onChange={setContrast} />
              </div>
            </section>
          ) : (
            <section>
              <h3 className="text-[10px] font-mono uppercase tracking-widest text-white/90 mb-4 flex items-center gap-2">
                <span className="w-1 h-3 bg-white"></span> MONO Engine
              </h3>
              <div className="space-y-4">
                <div className="flex gap-4">
                  <ToggleButtonGroup
                    label="Type"
                    value={dotType}
                    onChange={setDotType}
                    options={[
                      { label: 'Classic', value: 0 },
                      { label: 'Gooey', value: 1 }
                    ]}
                  />
                  <ToggleButtonGroup
                    label="Grid"
                    value={dotGrid}
                    onChange={setDotGrid}
                    options={[
                      { label: 'Square', value: 0 },
                      { label: 'Hex', value: 1 }
                    ]}
                  />
                </div>

                <SliderControl label="Dot Size" value={dotSize} min={0.01} max={1} step={0.01} onChange={setDotSize} />
                <SliderControl label="Radius" value={dotRadius} min={0} max={2} step={0.01} onChange={setDotRadius} />
                <SliderControl label="Contrast" value={dotContrast} min={0} max={1} step={0.01} onChange={setDotContrast} />
              </div>
            </section>
          )}

          <section className="pt-8 border-t border-white/5 space-y-4">
            <button className="w-full h-12 bg-white text-black rounded-full text-xs font-semibold uppercase tracking-[0.1em] hover:bg-white/90 transition-colors" onClick={() => {
              if (canvasRef.current) {
                const link = document.createElement('a');
                link.download = 'halftone-mesh-gradient.png';
                link.href = canvasRef.current.toDataURL();
                link.click();
              }
            }}>Export Asset</button>
            <div className="flex justify-center items-center gap-3 text-[10px] text-white/50 font-sans mt-2">
              <a href="https://github.com/gamanlook/HalftoneGradient" target="_blank" rel="noopener noreferrer" className="hover:text-white/80 transition-colors decoration-white/20 underline-offset-4 hover:underline">
                View on GitHub
              </a>
              <span>•</span>
              <a href="https://shaders.paper.design" target="_blank" rel="noopener noreferrer" className="hover:text-white/80 transition-colors decoration-white/20 underline-offset-4 hover:underline">
                Shaders by Paper
              </a>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
