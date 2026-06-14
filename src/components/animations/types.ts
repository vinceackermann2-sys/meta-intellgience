export type AnimationState = 
  | 'logo' 
  | 'morphing-to-orb' 
  | 'orb-thinking' 
  | 'morphing-to-logo';

export interface GradientPreset {
  id: string;
  name: string;
  colors: string[]; // Conic gradient color definitions
  logoColors: {
    palePink: string;
    blue: string;
    lightBlue: string;
    lightPink: string;
    lightPurple: string;
  };
  glowColor: string;
}

export interface AnimationConfig {
  transitionSpeed: number; // Duration of morphing in seconds (e.g., 1.5)
  orbRotateSpeed: number; // Speed of swirl rotation in seconds per rot (e.g., 8)
  thinkingRotateSpeed: number; // Faster rotational speed during thinking (e.g., 3)
  breathIntensity: number; // Scale fluctuation multiplier (e.g., 0.08)
  glowRadius: number; // Drop shadow blur radius in px (e.g., 30)
  size: number; // Orb/Logo bounding box dimension in pixels (e.g., 320)
  particleCount: number; // Floating ambient particles count (e.g., 12)
  themePreset: string; // Active theme preset ID
}

export interface MockTask {
  id: string;
  emoji: string;
  text: string;
  delay: number; // delay before this sub-step completes in ms
}
