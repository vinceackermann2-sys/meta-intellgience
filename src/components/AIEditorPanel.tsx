import React, { useState, useRef } from 'react';
import { Sparkles, X, Upload, FileVideo, FileImage, LayoutTemplate, Music, MessageSquareText, Loader2, CheckCircle2 } from 'lucide-react';
import { twMerge } from 'tailwind-merge';
import { clsx, type ClassValue } from 'clsx';
import { motion, AnimatePresence } from 'motion/react';
import { getActiveHyperframesProjectId } from '../lib/hyperframes';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const AspectRatioCard = ({ ratio, label, icon, selected, onClick }: any) => (
  <button
    onClick={onClick}
    className={cn(
      "flex flex-col items-center justify-center gap-2 p-3 rounded-xl border-2 transition-all w-full",
      selected 
        ? "border-indigo-500 bg-indigo-50/50 text-indigo-700" 
        : "border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:bg-slate-50"
    )}
  >
    <div className={cn(
      "border-2 rounded flex items-center justify-center",
      selected ? "border-indigo-500" : "border-slate-400",
      ratio === '16:9' ? "w-8 h-4.5" : 
      ratio === '9:16' ? "w-4.5 h-8" : "w-6 h-6"
    )} />
    <span className="text-[10px] font-bold uppercase tracking-wider">{label}</span>
  </button>
);

// Parse script text and prompt into narrative beat strings for captions
function extractBeatsFromScript(scriptText: string, prompt: string): string[] {
  const source = [scriptText, prompt].filter(Boolean).join('\n').trim();
  if (!source) {
    return [
      'Open with a bold visual hook',
      'Show the strongest proof point',
      'Build momentum with fast cuts',
      'End with a crisp call to action'
    ];
  }
  const lines = source
    .split(/\r?\n|(?<=[.!?])\s+/)
    .map((line) => line.replace(/^[-*#\d.\s]+/, '').trim())
    .filter((line) => line.length > 0)
    .slice(0, 8);
  return lines.length ? lines : [source.slice(0, 120)];
}

export default function AIEditorPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [aspectRatio, setAspectRatio] = useState('9:16');
  const [prompt, setPrompt] = useState('');
  const [mediaFiles, setMediaFiles] = useState<File[]>([]);
  const [scriptFile, setScriptFile] = useState<File | null>(null);
  const [projectId, setProjectId] = useState<string>('');
  
  const [generationState, setGenerationState] = useState<'idle' | 'analyzing' | 'cutting' | 'syncing' | 'rendering' | 'success'>('idle');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const scriptInputRef = useRef<HTMLInputElement>(null);

  const handleMediaUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files as FileList);
      setMediaFiles(prev => [...prev, ...newFiles]);
      console.log(`[Hyper-Edit] Selected ${newFiles.length} files. Total: ${mediaFiles.length + newFiles.length}`);
    }
  };

  const removeFile = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setMediaFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleGenerate = async () => {
    if (mediaFiles.length === 0 && !prompt && !scriptFile) return;
    
    try {
      console.log(`[Hyper-Edit] Starting generation for project...`);
      setGenerationState('analyzing');
      
      // 1. Read script file text if attached
      const readScriptText = (): Promise<string> => {
        return new Promise((resolve) => {
          if (!scriptFile) return resolve("");
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result as string || "");
          reader.onerror = () => resolve("");
          reader.readAsText(scriptFile);
        });
      };
      
      const scriptText = await readScriptText();

      const activeProjectId = projectId || await getActiveHyperframesProjectId();
      setProjectId(activeProjectId);

      // 2. Upload media/audio files if any
      setGenerationState('cutting');
      const uploadFiles = async (): Promise<any[]> => {
        if (mediaFiles.length === 0) return [];
        console.log(`[Hyper-Edit] Uploading ${mediaFiles.length} files...`);
        const formData = new FormData();
        for (const file of mediaFiles) {
          formData.append("file", file);
        }
        try {
          const res = await fetch(`/api/hyper-edit/upload?projectId=${encodeURIComponent(activeProjectId)}`, {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            const files = data.files || data.uploaded || data.uploadedFiles || (Array.isArray(data) ? data : []);
            console.log(`[Hyper-Edit] Upload successful. Received ${files.length} server-side assets.`);
            return Array.isArray(files) ? files : [];
          } else {
            console.error(`[Hyper-Edit] Upload failed with status: ${res.status}`);
          }
        } catch (e) {
          console.error("[Hyper-Edit] Upload error", e);
        }
        return [];
      };

      const uploadedAssets = await uploadFiles();
      
      if (mediaFiles.length > 0 && uploadedAssets.length === 0) {
        console.warn("[Hyper-Edit] Media was selected but upload returned no assets. Proceeding with fallback mode.");
      }

      // 3. Trigger AI Auto-Editor generator
      setGenerationState('syncing');
      
      const payload = {
        prompt,
        aspectRatio,
        mediaAssets: uploadedAssets,
        scriptText,
        projectId: activeProjectId,
      };

      // Transition visual cue to rendering right before requesting
      setGenerationState('rendering');

      const res = await fetch("/api/hyper-edit/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("AI editor generation failed");
      }

      const result = await res.json();
      console.log(`[Hyper-Edit] Generation complete:`, result);

      // 4. The server has written the AI-generated composition to project/index.html.
      // The real Hyperframes Studio reads that file directly via /api/projects/{id}.
      // We just need to fire the success event so the StudioFrame remounts StudioApp
      // and it picks up the new composition automatically.
      
      console.log(`[Hyper-Edit] Composition written to project. Triggering Studio reload...`);
      
      window.dispatchEvent(new CustomEvent("hyper-edit-success", { detail: { projectId: activeProjectId } }));
      setGenerationState('success');

      setTimeout(() => {
        setGenerationState('idle');
        setIsOpen(false);
        setMediaFiles([]);
        setPrompt('');
        setScriptFile(null);
      }, 3000);

    } catch (e) {
      console.error("[HYPER-EDIT GENERATION FAILED]:", e);
      setGenerationState('idle');
      alert("Hyper-Edit Studio failed to process your media. Please try again with different media or direction.");
    }
  };


  return (
    <>
      <div className="absolute right-6 bottom-6 z-50">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsOpen(true)}
          className={cn(
            "group flex items-center gap-2 px-5 py-3.5 bg-slate-900 hover:bg-indigo-600 text-white rounded-full shadow-2xl transition-all duration-300",
            isOpen ? "scale-0 opacity-0 pointer-events-none" : "scale-100 opacity-100"
          )}
        >
          <Sparkles className="w-5 h-5 text-indigo-300 group-hover:text-white transition-colors" />
          <span className="font-bold tracking-wide text-sm">Hyper-Edit Studio</span>
        </motion.button>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, x: 400, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 400, scale: 0.95 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="absolute right-6 top-6 bottom-6 w-[400px] bg-white rounded-3xl shadow-2xl border border-slate-200/60 overflow-hidden flex flex-col z-[60]"
          >
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-indigo-600" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900 tracking-tight">AI Auto-Editor</h3>
                  <p className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Director Mode</p>
                </div>
              </div>
              <motion.button 
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setIsOpen(false)}
                className="p-2 hover:bg-slate-200 rounded-full text-slate-400 transition-colors"
                disabled={generationState !== 'idle' && generationState !== 'success'}
              >
                <X className="w-4 h-4" />
              </motion.button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scrollbar">
              {/* Media Upload */}
              <div className="space-y-3">
                <label className="text-[11px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
                  <FileVideo className="w-3.5 h-3.5" /> 1. Source Media
                </label>
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full border-2 border-dashed border-slate-200 rounded-2xl p-6 flex flex-col items-center justify-center gap-3 hover:border-indigo-400 hover:bg-indigo-50/30 transition-colors cursor-pointer"
                >
                  <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
                    <Upload className="w-5 h-5 text-slate-500" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-semibold text-slate-700">Drop media or click</p>
                    <p className="text-xs text-slate-500">Supports images, video clips & audio</p>
                  </div>
                </div>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleMediaUpload} 
                  multiple 
                  accept="image/*,video/*,audio/*,.mp3,.wav,.m4a,.aac,.ogg" 
                  className="hidden" 
                />
                
                {mediaFiles.length > 0 && (
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {mediaFiles.map((file, i) => (
                      <div key={i} className="w-16 h-16 rounded-lg bg-slate-100 border border-slate-200 shrink-0 flex items-center justify-center overflow-hidden relative group">
                         {file.type.startsWith('video') ? <FileVideo className="w-6 h-6 text-slate-400" /> : file.type.startsWith('audio') ? <Music className="w-6 h-6 text-slate-400" /> : <FileImage className="w-6 h-6 text-slate-400" />}
                         <div className="absolute inset-0 bg-black/5 flex flex-col justify-end p-1">
                           <span className="text-[8px] font-bold text-white truncate drop-shadow-md">{file.name}</span>
                         </div>
                         <button 
                           onClick={(e) => removeFile(i, e)}
                           className="absolute top-0.5 right-0.5 w-4 h-4 bg-black/50 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                         >
                           <X className="w-2.5 h-2.5" />
                         </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Aspect Ratio */}
              <div className="space-y-3">
                <label className="text-[11px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
                  <LayoutTemplate className="w-3.5 h-3.5" /> 2. Aspect Ratio
                </label>
                <div className="flex gap-3">
                  <AspectRatioCard ratio="9:16" label="Vertical" selected={aspectRatio === '9:16'} onClick={() => setAspectRatio('9:16')} />
                  <AspectRatioCard ratio="16:9" label="Landscape" selected={aspectRatio === '16:9'} onClick={() => setAspectRatio('16:9')} />
                  <AspectRatioCard ratio="1:1" label="Square" selected={aspectRatio === '1:1'} onClick={() => setAspectRatio('1:1')} />
                </div>
              </div>

              {/* Prompt & Script */}
              <div className="space-y-3">
                <label className="text-[11px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
                  <MessageSquareText className="w-3.5 h-3.5" /> 3. Direction
                </label>
                <div className="relative">
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe how the video should be edited... (e.g. 'Create an energetic hype reel. Sync cuts to a fast beat, use bold typography for captions...')"
                    className="w-full h-32 bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10 resize-none transition-all"
                  />
                  
                  <div className="absolute bottom-3 right-3 flex items-center gap-2">
                    <button 
                      onClick={() => scriptInputRef.current?.click()}
                      className={cn("px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase flex items-center gap-1.5 transition-colors shadow-sm", scriptFile ? "bg-indigo-100 text-indigo-700" : "bg-white border border-slate-200 text-slate-500 hover:bg-slate-50")}
                    >
                      <Upload className="w-3 h-3" />
                      {scriptFile ? scriptFile.name.slice(0, 18) : 'Attach Script'}
                    </button>
                    <input 
                      type="file" 
                      ref={scriptInputRef} 
                      onChange={(e) => { if(e.target.files) setScriptFile(e.target.files[0]) }} 
                      accept=".txt,.md,.srt,.vtt,.json,text/*" 
                      className="hidden" 
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Action Footer */}
            <div className="p-6 border-t border-slate-100 bg-slate-50/50">
               {generationState === 'idle' ? (
                 <motion.button 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleGenerate}
                  disabled={mediaFiles.length === 0 && !prompt && !scriptFile}
                  className="w-full h-12 bg-slate-900 hover:bg-indigo-600 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-xl font-bold tracking-wide transition-all duration-300 flex items-center justify-center gap-2 shadow-lg hover:shadow-indigo-500/25 disabled:shadow-none pointer-events-auto"
                 >
                   <Sparkles className="w-4 h-4" />
                   Generate Final Cut
                 </motion.button>
               ) : (
                 <div className="w-full h-12 rounded-xl flex items-center justify-center gap-3 font-bold text-sm tracking-wide bg-white border border-slate-200 shadow-inner">
                   {generationState === 'success' ? (
                     <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} className="flex items-center gap-2 text-emerald-600">
                       <CheckCircle2 className="w-5 h-5" />
                       Media applied to Studio timeline
                     </motion.div>
                   ) : (
                     <div className="flex items-center gap-3 text-slate-700">
                       <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                       {generationState === 'analyzing' && 'Analyzing script & pacing...'}
                       {generationState === 'cutting' && 'Cutting clips to media assets...'}
                       {generationState === 'syncing' && 'Syncing script beats to clips...'}
                       {generationState === 'rendering' && 'Applying visual studio style...'}
                     </div>
                   )}
                 </div>
               )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
