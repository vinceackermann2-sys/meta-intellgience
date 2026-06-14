import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, FastForward, Rewind, ZoomIn, ZoomOut, Maximize2, Volume2, Plus, Sliders, Scissors, Video, Music, Type, Check, RefreshCw, Layers, Film, Image, FileVideo, FileAudio, Save, Trash2, FolderPlus } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Timeline Clip type used across the studio
export interface TimelineClip {
  id: string;
  type: 'video' | 'audio' | 'text';
  name: string;
  start: number; // in seconds
  duration: number; // in seconds
  color: string;
  captionText?: string; // For caption/text clips
  src?: string; // asset source path
}

// Source media file entry
interface SourceMediaEntry {
  name: string;
  type: 'video' | 'audio' | 'image';
  path?: string;
  size?: number;
  meta?: string; // e.g. "1080x1920 · 15.0s"
}

const TRACK_COLORS = {
  video: 'bg-indigo-500 border-indigo-400',
  audio: 'bg-emerald-500 border-emerald-400',
  text: 'bg-amber-500 border-amber-400',
};

export const StudioApp: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(15);
  const [zoom, setZoom] = useState(10);
  const [activeClipId, setActiveClipId] = useState<string | null>(null);
  const [isProcessingAI, setIsProcessingAI] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const handleClearTimeline = () => {
    setClips([]);
    setCurrentTime(0);
  };

  const handleNewProject = () => {
    setClips([]);
    setSourceMedia([]);
    setDuration(15);
    setCurrentTime(0);
    setToastMessage("New project created.");
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleSaveProject = () => {
    setToastMessage("Project saved successfully.");
    setTimeout(() => setToastMessage(null), 3000);
  };

  // Dynamic state — populated by AI generation
  const [clips, setClips] = useState<TimelineClip[]>([
    { id: 'v1', type: 'video', name: 'Original Raw Footage.mp4', start: 0, duration: 12, color: 'bg-indigo-600/70 border-indigo-500/70' },
    { id: 'a1', type: 'audio', name: 'Default Audio Track.wav', start: 0, duration: 15, color: 'bg-emerald-600/70 border-emerald-500/70' }
  ]);

  const [sourceMedia, setSourceMedia] = useState<SourceMediaEntry[]>([
    { name: 'AI Hype Reel.mp4', type: 'video', meta: '1080x1920 · 15.0s' },
    { name: 'SynthWave Beat.wav', type: 'audio', meta: '44.1kHz · Stereo' }
  ]);

  // Core Playback State
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(Date.now());

  // Listen for AI timeline update events
  useEffect(() => {
    const handleTimelineUpdate = (e: CustomEvent) => {
      const detail = e.detail || {};
      const { timelineClips, sourceFiles, totalDuration } = detail;

      if (Array.isArray(timelineClips) && timelineClips.length > 0) {
        const processedClips: TimelineClip[] = timelineClips.map((clip: any, i: number) => ({
          id: clip.id || `clip-${i}`,
          type: clip.type || 'video',
          name: clip.name || `Clip ${i + 1}`,
          start: clip.start ?? 0,
          duration: clip.duration ?? 3,
          color: TRACK_COLORS[clip.type as keyof typeof TRACK_COLORS] || TRACK_COLORS.video,
          captionText: clip.captionText || undefined,
          src: clip.src || undefined,
        }));
        setClips(processedClips);
        setActiveClipId(processedClips[0]?.id || null);
      }

      if (Array.isArray(sourceFiles) && sourceFiles.length > 0) {
        setSourceMedia(sourceFiles.map((f: any) => ({
          name: f.name || f.filename || 'Unnamed',
          type: f.type?.startsWith('video') ? 'video' : f.type?.startsWith('audio') ? 'audio' : 'image',
          path: f.path,
          size: f.size,
          meta: f.meta || (f.type?.startsWith('video') ? 'Video Clip' : f.type?.startsWith('audio') ? 'Audio Track' : 'Image'),
        })));
      }

      if (totalDuration && totalDuration > 0) {
        setDuration(totalDuration);
      }
    };

    const handleHyperEditSuccess = () => {
      setIsProcessingAI(true);
      setTimeout(() => {
        setIsProcessingAI(false);
        setCurrentTime(0);
        setIsPlaying(true);
      }, 2500);
    };

    window.addEventListener('hyper-edit-timeline-update', handleTimelineUpdate as EventListener);
    window.addEventListener('hyper-edit-success', handleHyperEditSuccess);

    return () => {
      window.removeEventListener('hyper-edit-timeline-update', handleTimelineUpdate as EventListener);
      window.removeEventListener('hyper-edit-success', handleHyperEditSuccess);
    };
  }, []);

  // Frame Loop for Player Simulation
  useEffect(() => {
    if (isPlaying) {
      const updateFrame = () => {
        const now = Date.now();
        const delta = (now - lastTimeRef.current) / 1000;
        lastTimeRef.current = now;

        setCurrentTime((prev) => {
          let next = prev + delta;
          if (next >= duration) {
            next = 0;
          }
          return next;
        });

        animationRef.current = requestAnimationFrame(updateFrame);
      };

      lastTimeRef.current = Date.now();
      animationRef.current = requestAnimationFrame(updateFrame);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, duration]);

  // Handle Seek Click on Track
  const handleTimelineSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const progress = clickX / rect.width;
    setCurrentTime(progress * duration);
  };

  // Get the currently active caption based on playback time
  const getActiveCaption = (): string | null => {
    const captionClips = clips.filter(c => c.type === 'text');
    for (const clip of captionClips) {
      if (currentTime >= clip.start && currentTime < clip.start + clip.duration) {
        return clip.captionText || clip.name;
      }
    }
    return null;
  };

  // Get source media icon
  const getSourceIcon = (type: string) => {
    switch (type) {
      case 'video': return <Video className="w-5 h-5" />;
      case 'audio': return <Music className="w-5 h-5" />;
      default: return <Image className="w-5 h-5" />;
    }
  };

  const getSourceIconColors = (type: string) => {
    switch (type) {
      case 'video': return { bg: 'bg-indigo-500/10 border-indigo-500/20', text: 'text-indigo-400' };
      case 'audio': return { bg: 'bg-emerald-500/10 border-emerald-500/20', text: 'text-emerald-400' };
      default: return { bg: 'bg-violet-500/10 border-violet-500/20', text: 'text-violet-400' };
    }
  };

  const activeCaption = isPlaying ? getActiveCaption() : null;
  const hasContent = clips.length > 2 || clips.some(c => c.type === 'text');

  return (
    <div id="hyperframes-studio-wrap" className="w-full h-full flex flex-col bg-slate-950 text-slate-100 font-sans select-none overflow-hidden relative">
      
      {/* Top Application Bar */}
      <div className="h-14 border-b border-slate-900 bg-slate-950/80 px-6 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-indigo-600 flex items-center justify-center mr-2">
            <Film className="w-4 h-4 text-white" />
          </div>
          <span className="font-black tracking-tight text-white">Studio</span>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={handleClearTimeline}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-bold bg-slate-900 hover:bg-slate-800 text-rose-400 rounded-lg transition-colors border border-slate-800"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear Timeline
          </button>
          <button 
            onClick={handleNewProject}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-bold bg-slate-900 hover:bg-slate-800 text-emerald-400 rounded-lg transition-colors border border-slate-800"
          >
            <FolderPlus className="w-3.5 h-3.5" />
            New Project
          </button>
          <button 
            onClick={handleSaveProject}
            className="flex items-center gap-2 px-4 py-1.5 text-xs font-bold bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors shadow-lg shadow-indigo-500/20"
          >
            <Save className="w-3.5 h-3.5" />
            Save Video
          </button>
        </div>
      </div>

      {/* Toast Notification */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }} 
            animate={{ opacity: 1, y: 0 }} 
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-indigo-600 text-white px-4 py-2 rounded-lg font-bold shadow-2xl z-[100] flex items-center gap-2"
          >
            <Check className="w-4 h-4" />
            {toastMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Processing Loader Overlay */}
      <AnimatePresence>
        {isProcessingAI && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-slate-950/90 z-50 flex flex-col items-center justify-center gap-4 border border-indigo-500/30"
          >
            <div className="relative flex items-center justify-center">
              <div className="w-16 h-16 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin" />
              <Film className="w-6 h-6 text-indigo-400 absolute animate-pulse" />
            </div>
            <div className="text-center">
              <h3 className="text-lg font-bold text-white tracking-tight">Vibe-Cutting Active</h3>
              <p className="text-xs text-slate-400 mt-1 uppercase tracking-widest font-black">AI is slicing & syncing timeline blocks...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Editor Main Content: Top Screen Split */}
      <div className="flex-1 flex min-h-0">
        
        {/* Left Side: Dynamic Source/Project Assets */}
        <aside className="w-64 border-r border-slate-900 flex flex-col bg-slate-950 shrink-0 hidden md:flex">
          <div className="p-4 border-b border-slate-900 flex items-center justify-between">
            <span className="text-xs font-black uppercase tracking-widest text-slate-400 flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-indigo-400" /> Source Media
            </span>
            <span className="text-[9px] font-bold text-slate-600 bg-slate-900 px-2 py-0.5 rounded-full">
              {sourceMedia.length} files
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
            {sourceMedia.map((item, i) => {
              const colors = getSourceIconColors(item.type);
              return (
                <motion.div 
                  key={`${item.name}-${i}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80 flex items-center gap-3 hover:bg-slate-900/80 transition-colors cursor-default"
                >
                  <div className={`w-10 h-10 rounded ${colors.bg} border flex items-center justify-center ${colors.text}`}>
                    {getSourceIcon(item.type)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-semibold truncate">{item.name}</p>
                    <p className="text-[10px] text-slate-500">{item.meta || item.type}</p>
                  </div>
                </motion.div>
              );
            })}
            {sourceMedia.length === 0 && (
              <div className="p-6 text-center">
                <Layers className="w-8 h-8 text-slate-700 mx-auto mb-2" />
                <p className="text-xs text-slate-600">No media uploaded yet</p>
                <p className="text-[10px] text-slate-700 mt-1">Use Hyper-Edit to add media</p>
              </div>
            )}
          </div>
        </aside>

        {/* Middle Area: Video Canvas Monitor */}
        <div className="flex-1 bg-slate-900/30 flex flex-col items-center justify-center p-6 relative">
          
          {/* Virtual Canvas Box */}
          <div className="aspect-[9/16] h-[340px] md:h-[420px] bg-black rounded-3xl border border-slate-800 flex flex-col relative overflow-hidden shadow-2xl">
            {/* Realtime Caption overlay */}
            <AnimatePresence mode="wait">
              {activeCaption && (
                <div className="absolute top-1/2 left-0 right-0 transform -translate-y-1/2 flex items-center justify-center text-center px-4 z-20 pointer-events-none">
                  <motion.p 
                    key={activeCaption}
                    initial={{ scale: 0.8, opacity: 0, y: 10 }} 
                    animate={{ scale: 1.05, opacity: 1, y: 0 }} 
                    exit={{ scale: 0.9, opacity: 0, y: -10 }}
                    transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                    className="text-lg md:text-xl font-black bg-amber-400 text-slate-950 px-4 py-1.5 rounded-lg shadow-2xl tracking-tight uppercase leading-tight max-w-[90%]"
                  >
                    {activeCaption}
                  </motion.p>
                </div>
              )}
            </AnimatePresence>

            {/* Simulated Frame Preview with animated glow */}
            <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden bg-slate-950">
              {hasContent ? (
                <>
                  <div className="absolute inset-0 opacity-40 bg-radial-gradient from-indigo-500/20 via-transparent to-transparent animate-pulse" />
                  <div className="text-center p-4">
                    <Video className="w-12 h-12 text-indigo-400 mx-auto opacity-70 mb-2" />
                    <span className="text-xs font-black uppercase tracking-wider text-indigo-300">
                      {isPlaying ? 'Playing' : 'Ready'}
                    </span>
                    <p className="text-[10px] text-slate-500 mt-2">
                      {clips.filter(c => c.type === 'video').length} clips · {clips.filter(c => c.type === 'text').length} captions · {clips.filter(c => c.type === 'audio').length} audio
                    </p>
                    <p className="text-[10px] text-slate-600 mt-0.5 font-mono">{currentTime.toFixed(2)}s / {duration.toFixed(1)}s</p>
                  </div>
                </>
              ) : (
                <div className="text-center p-4">
                  <Film className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                  <span className="text-xs font-semibold text-slate-400">Idle Canvas</span>
                  <p className="text-[10px] text-slate-600 mt-1 font-mono">Ready for compilation</p>
                </div>
              )}
            </div>

            {/* Playback Progress Indicator Bar inside Monitor */}
            <div className="h-1 bg-slate-900 w-full relative">
              <div className="absolute top-0 bottom-0 left-0 bg-indigo-500 transition-all" style={{ width: `${(currentTime / duration) * 100}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Editor Bottom Content: Timeline & Multitrack Workspace */}
      <div className="h-72 border-t border-slate-900 bg-slate-950 flex flex-col relative shrink-0">
        
        {/* Timeline Control Rail */}
        <div className="h-12 bg-slate-950/80 border-b border-slate-900/50 px-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-xs font-mono font-bold text-slate-400 bg-slate-900/80 border border-slate-800 px-2.5 py-1 rounded-md">
              {Math.floor(currentTime / 60).toString().padStart(2, '0')}:
              {(Math.floor(currentTime) % 60).toString().padStart(2, '0')}:
              {Math.floor((currentTime % 1) * 30).toString().padStart(2, '0')}
            </span>
            <div className="flex items-center gap-2">
              <button onClick={() => setCurrentTime(0)} className="p-1 px-1.5 hover:bg-slate-900 rounded text-slate-400 hover:text-white transition-all"><Rewind className="w-4 h-4" /></button>
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 rounded-full bg-indigo-500 hover:bg-indigo-400 text-white flex items-center justify-center transition-all shadow-md active:scale-95"
              >
                {isPlaying ? <Pause className="w-4 h-4 py-0" /> : <Play className="w-4 h-4 ml-0.5" />}
              </button>
              <button onClick={() => setCurrentTime(duration)} className="p-1 px-1.5 hover:bg-slate-900 rounded text-slate-400 hover:text-white transition-all"><FastForward className="w-4 h-4" /></button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <button onClick={() => setZoom(Math.max(5, zoom - 2))} className="p-1.5 hover:bg-slate-900 rounded text-slate-400 hover:text-white"><ZoomOut className="w-4 h-4" /></button>
              <span className="text-[10px] font-bold text-slate-500">{zoom}x</span>
              <button onClick={() => setZoom(Math.min(25, zoom + 2))} className="p-1.5 hover:bg-slate-900 rounded text-slate-400 hover:text-white"><ZoomIn className="w-4 h-4" /></button>
            </div>
            <button className="p-1.5 hover:bg-slate-900 rounded text-slate-400 hover:text-white transition-all"><Maximize2 className="w-4 h-4" /></button>
          </div>
        </div>

        {/* Multi-Track Interactive Timelines Area */}
        <div 
          className="flex-1 overflow-x-auto overflow-y-hidden custom-scrollbar bg-slate-950/40 relative"
          onClick={handleTimelineSeek}
        >
          {/* Vertical Playhead Rule Indicator */}
          <div 
            className="absolute top-0 bottom-0 w-[2px] bg-red-500 pointer-events-none z-30 shadow-[0_0_12px_rgba(239,68,68,0.5)]"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />

          {/* Time ruler ticks */}
          <div className="h-6 border-b border-slate-900/30 flex items-end relative" style={{ width: '100%' }}>
            {Array.from({ length: Math.floor(duration) + 1 }).map((_, i) => (
              <span 
                key={i} 
                className="absolute text-[9px] font-mono font-bold text-slate-600 border-l border-slate-900 h-2.5 pl-1 flex items-end leading-none"
                style={{ left: `${(i / duration) * 100}%` }}
              >
                {i}s
              </span>
            ))}
          </div>

          {/* Multi-Tracks Blocks */}
          <div className="flex-1 flex flex-col py-1 space-y-1.5 px-1 relative">
            
            {/* Visual Track 1: Video Blocks */}
            <div className="h-10 border border-slate-900 bg-slate-900/20 rounded-xl relative flex items-center px-4">
              <span className="absolute left-2.5 text-[8px] font-black uppercase tracking-widest text-slate-500 bg-slate-950 px-1 rounded z-20">Video Track</span>
              {clips.filter(c => c.type === 'video').map(clip => (
                <motion.div
                  key={clip.id}
                  layout
                  initial={{ opacity: 0, scaleX: 0.5 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={{ type: 'spring', damping: 25 }}
                  onClick={(e) => { e.stopPropagation(); setActiveClipId(clip.id); }}
                  className={`absolute h-8 ${clip.color} rounded-lg border text-white text-[10px] font-bold px-3 flex items-center justify-between cursor-pointer transition-all ${activeClipId === clip.id ? 'ring-2 ring-indigo-400 scale-[1.01] shadow-2xl' : 'opacity-70 hover:opacity-100'}`}
                  style={{
                    left: `${(clip.start / duration) * 100}%`,
                    width: `${(clip.duration / duration) * 100}%`,
                  }}
                >
                  <span className="truncate max-w-[140px]">{clip.name}</span>
                  <span className="text-[8px] font-mono text-white/60">{clip.duration.toFixed(1)}s</span>
                </motion.div>
              ))}
            </div>

            {/* Audio Track 2: Music Blocks */}
            <div className="h-10 border border-slate-900 bg-slate-900/20 rounded-xl relative flex items-center px-4">
              <span className="absolute left-2.5 text-[8px] font-black uppercase tracking-widest text-slate-400 bg-slate-950 px-1 rounded z-20">Audio Track</span>
              {clips.filter(c => c.type === 'audio').map(clip => (
                <motion.div
                  key={clip.id}
                  layout
                  initial={{ opacity: 0, scaleX: 0.5 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={{ type: 'spring', damping: 25 }}
                  onClick={(e) => { e.stopPropagation(); setActiveClipId(clip.id); }}
                  className={`absolute h-8 ${clip.color} rounded-lg border text-white text-[10px] font-bold px-3 flex items-center justify-between cursor-pointer transition-all ${activeClipId === clip.id ? 'ring-2 ring-emerald-400 scale-[1.01] shadow-2xl' : 'opacity-70 hover:opacity-100'}`}
                  style={{
                    left: `${(clip.start / duration) * 100}%`,
                    width: `${(clip.duration / duration) * 100}%`,
                  }}
                >
                  <span className="truncate max-w-[140px]">{clip.name}</span>
                  <span className="text-[8px] font-mono text-white/60">{clip.duration.toFixed(1)}s</span>
                </motion.div>
              ))}
            </div>

            {/* Captions Track 3: Subtitles / Overlay */}
            <div className="h-10 border border-slate-900 bg-slate-900/20 rounded-xl relative flex items-center px-4">
              <span className="absolute left-2.5 text-[8px] font-black uppercase tracking-widest text-slate-400 bg-slate-950 px-1 rounded z-20">Caption Track</span>
              {clips.filter(c => c.type === 'text').map(clip => (
                <motion.div
                  key={clip.id}
                  layout
                  initial={{ opacity: 0, scaleX: 0.5 }}
                  animate={{ opacity: 1, scaleX: 1 }}
                  transition={{ type: 'spring', damping: 25 }}
                  onClick={(e) => { e.stopPropagation(); setActiveClipId(clip.id); }}
                  className={`absolute h-8 ${clip.color} rounded-lg border text-white text-[10px] font-bold px-3 flex items-center justify-between cursor-pointer transition-all ${activeClipId === clip.id ? 'ring-2 ring-amber-400 scale-[1.01] shadow-2xl' : 'opacity-70 hover:opacity-100'}`}
                  style={{
                    left: `${(clip.start / duration) * 100}%`,
                    width: `${(clip.duration / duration) * 100}%`,
                  }}
                >
                  <span className="truncate max-w-[140px]">{clip.captionText || clip.name}</span>
                  <span className="text-[8px] font-mono text-white/60">{clip.duration.toFixed(1)}s</span>
                </motion.div>
              ))}
            </div>

          </div>
        </div>
      </div>

    </div>
  );
};
