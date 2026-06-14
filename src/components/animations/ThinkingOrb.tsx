import { useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { AnimationState, GradientPreset, AnimationConfig } from './types';

interface ThinkingOrbProps {
  state: AnimationState;
  preset: GradientPreset;
  config: AnimationConfig;
}

export default function ThinkingOrb({ state, preset, config }: ThinkingOrbProps) {
  const isOrbActive = state !== 'logo';
  const isThinking = state === 'orb-thinking';
  const isMorphingToLogo = state === 'morphing-to-logo';

  // Calculate dynamic conic-gradient string
  const conicGradient = useMemo(() => {
    return `conic-gradient(from 0deg, ${preset.colors.join(', ')})`;
  }, [preset.colors]);

  // Orb base variants for morph entry/exit
  const orbVariants = {
    hidden: {
      scale: 1.5, // Start slightly larger
      rotate: 0,
      opacity: 0,
      filter: 'blur(25px) brightness(3) contrast(1.5) drop-shadow(0px 0px 50px rgba(255,255,255,1))',
      transition: {
        duration: config.transitionSpeed,
        ease: [0.65, 0, 0.35, 1], // Match perfectly with logo disappearance
      }
    },
    visible: {
      scale: 1, // Full size
      rotate: 180,
      opacity: 1,
      filter: 'blur(0px) brightness(1) contrast(1) drop-shadow(0px 10px 25px rgba(0,0,0,0.25))',
      transition: {
        duration: config.transitionSpeed,
        ease: [0.65, 0, 0.35, 1], // Match perfectly with logo appearance
      }
    }
  };

  return (
    <div 
      className="absolute flex items-center justify-center pointer-events-none select-none"
      style={{ width: config.size, height: config.size }}
      id="thinking_orb_wrapper"
    >
      <AnimatePresence>
        {isOrbActive && (
          <motion.div
            key="main-orb"
            initial="hidden"
            animate={isMorphingToLogo ? "hidden" : "visible"}
            exit="hidden"
            variants={orbVariants as any}
            className="relative w-full h-full flex items-center justify-center cursor-default"
            id="orb_animation_stage"
          >
            {/* Glowing Orb Shadow backing */}
            <motion.div
              animate={{
                boxShadow: `0 0 ${config.glowRadius}px ${config.glowRadius / 3}px ${preset.glowColor}`
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              style={{ opacity: isThinking ? 0 : 1 }}
              className="absolute inset-0 rounded-full blur-sm pointer-events-none transition-opacity duration-1000"
              id="orb_glow_backing"
            />

            {/* Main Rotating Fluid Orb body with breathing */}
            <motion.div
              animate={{ scale: 1 }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="relative w-full h-full rounded-full overflow-hidden"
              style={{
                boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255,255,255,0.2)"
              }}
              id="orb_glass_sphere"
            >
              {/* Swirling Conic Gradient Backdrop */}
              <motion.div
                animate={{
                  rotate: 360
                }}
                transition={{
                  duration: isThinking ? config.thinkingRotateSpeed : config.orbRotateSpeed,
                  repeat: Infinity,
                  ease: "linear"
                }}
                style={{ 
                  background: conicGradient,
                  filter: 'blur(14px) saturate(1.5)'
                }}
                className="absolute inset-[-40%] rounded-full opacity-90"
                id="swirling_gradient"
              />

              {/* Internal Moving Colors during Thinking */}
              <AnimatePresence>
                {isThinking && preset.colors.slice(0, 4).map((color, i) => (
                  <motion.div
                    key={`inner-color-${i}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ 
                      opacity: 0.85,
                      scale: [1, 1.3, 1.1, 1],
                      x: [0, i % 2 === 0 ? 50 : -50, i % 2 === 0 ? -30 : 30, 0],
                      y: [0, i > 1 ? 50 : -50, i > 1 ? -30 : 30, 0],
                    }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{
                      duration: 3 + i * 0.5,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="absolute w-3/4 h-3/4 rounded-full blur-[25px] mix-blend-screen pointer-events-none"
                    style={{ 
                      backgroundColor: color,
                      left: '12.5%',
                      top: '12.5%'
                    }}
                  />
                ))}
              </AnimatePresence>

              {/* Glass Glare Overlay - Adds beautiful physical dimension */}
              <div 
                className="absolute inset-0 rounded-full bg-gradient-to-tr from-white/0 via-white/5 to-white/30 mix-blend-overlay pointer-events-none"
              />
              <div 
                className="absolute inset-x-0 top-0 h-1/2 bg-gradient-to-b from-white/20 to-transparent rounded-t-full pointer-events-none"
              />
              
              {/* Glass Edge Ring to Pop Forward */}
              <div
                className="absolute inset-0 rounded-full border-[2px] border-white/30 pointer-events-none"
                style={{
                  boxShadow: "inset 0 6px 12px rgba(255, 255, 255, 0.6), inset 0 -6px 12px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(255, 255, 255, 0.2)"
                }}
              />
              
              {/* Tiny inner pulse core */}
              <motion.div 
                className="absolute inset-1/3 rounded-full bg-white/10 filter blur-md mix-blend-overlay"
                animate={{ opacity: 0.3 }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
