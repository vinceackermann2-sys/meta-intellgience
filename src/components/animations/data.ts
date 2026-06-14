import { GradientPreset, AnimationConfig } from './types';

export const GRADIENT_PRESETS: GradientPreset[] = [
  {
    id: 'app-theme',
    name: 'App Theme',
    colors: [
      '#6366f1', // indigo-500
      '#d946ef', // fuchsia-500
      '#fbbf24', // amber-400
      '#a855f7', // purple-500
      '#d946ef', // fuchsia-500
      '#6366f1', // indigo-500
    ],
    logoColors: {
      palePink: '#c084fc',
      blue: '#6366f1',
      lightBlue: '#818cf8',
      lightPink: '#d946ef',
      lightPurple: '#a855f7'
    },
    glowColor: 'rgba(99, 102, 241, 0.45)'
  },
  {
    id: 'cosmic',
    name: 'Cosmic Swirl',
    colors: [
      '#EDAFC4',
      '#BD8EC4',
      '#818AC5',
      '#90A5D9',
      '#D68BAB',
      '#EDAFC4'
    ],
    logoColors: {
      palePink: '#EDAFC4',
      blue: '#818AC5',
      lightBlue: '#90A5D9',
      lightPink: '#D68BAB',
      lightPurple: '#BD8EC4'
    },
    glowColor: 'rgba(214, 139, 171, 0.5)'
  },
  {
    id: 'aurora',
    name: 'Emerald Aurora',
    colors: [
      '#34d399', // Emerald
      '#059669', // Deep Emerald
      '#06b6d4', // Cyan
      '#3b82f6', // Indigo-blue
      '#10b981', // Mint-green
      '#34d399'
    ],
    logoColors: {
      palePink: '#34d399',
      blue: '#3b82f6',
      lightBlue: '#06b6d4',
      lightPink: '#10b981',
      lightPurple: '#059669'
    },
    glowColor: 'rgba(52, 211, 153, 0.45)'
  },
  {
    id: 'cyberpunk',
    name: 'Cyber Neon',
    colors: [
      '#ff007f', // Rose neon
      '#7928ca', // Deep purple
      '#ff007f', // Red rose
      '#ff4500', // Crimson orange
      '#9b5de5', // Bright violet
      '#ff007f'
    ],
    logoColors: {
      palePink: '#ff007f',
      blue: '#7928ca',
      lightBlue: '#ff007f',
      lightPink: '#ff4500',
      lightPurple: '#9b5de5'
    },
    glowColor: 'rgba(255, 0, 127, 0.55)'
  },
  {
    id: 'sunset',
    name: 'Solar Flare',
    colors: [
      '#F59E0B', // Amber
      '#EF4444', // Red
      '#EC4899', // Pink
      '#F59E0B', // Amber
      '#F97316', // Orange
      '#F59E0B'
    ],
    logoColors: {
      palePink: '#F59E0B',
      blue: '#EF4444',
      lightBlue: '#EC4899',
      lightPink: '#F97316',
      lightPurple: '#F59E0B'
    },
    glowColor: 'rgba(249, 115, 22, 0.45)'
  }
];

export const DEFAULT_CONFIG: AnimationConfig = {
  transitionSpeed: 2.0, // Seconds
  orbRotateSpeed: 10, // Seconds per 360deg in idle
  thinkingRotateSpeed: 2.8, // Faster during thinking
  breathIntensity: 0.06, // Scale offset
  glowRadius: 35, // Tailwind blur in px
  size: 340, // Base scale bounding box
  particleCount: 15,
  themePreset: 'app-theme'
};

export const SAMPLE_PROMPTS = [
  "Analyze the system performance and generate visual reports.",
  "Refactor the authentication middleware to include double OAuth validations.",
  "Optimize client side caching mechanisms for rapid logo transformations.",
  "Synthesize standard micro-particles orbiting around the core neural hub."
];

export const SIMULATION_TASKS = [
  { id: '1', emoji: '🔍', text: 'Initializing dynamic vector canvas...', delay: 800 },
  { id: '2', emoji: '🧬', text: 'Decompiling abstract floral vector shapes...', delay: 1000 },
  { id: '3', emoji: '✨', text: 'Merging vector coordinates into swirling fluid state...', delay: 1200 },
  { id: '4', emoji: '🧠', text: 'Activating neural core - Synthesizing parameters...', delay: 2000 },
  { id: '5', emoji: '🦾', text: 'Refining and stabilizing particle coordinates...', delay: 1000 },
  { id: '6', emoji: '💫', text: 'Completed synthesis! Restoring shape integrity...', delay: 900 }
];
