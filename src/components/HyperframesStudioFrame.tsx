import { useState, useEffect, useCallback } from 'react';
import { StudioApp } from '@hyperframes/studio';
import { motion, AnimatePresence } from 'motion/react';
import { Trash2, FolderPlus, Save, Check, Film } from 'lucide-react';

export default function HyperframesStudioFrame() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Force a full Studio remount when the AI Editor generates a new composition.
  // The server writes the updated index.html to the project directory, and the
  // real Hyperframes Studio will re-read it on mount via /api/projects/{id}.
  const reloadStudio = useCallback(() => {
    setRefreshKey(prev => prev + 1);
  }, []);

  useEffect(() => {
    const handleSuccess = (e: any) => {
      const { projectId = 'project' } = e.detail || {};
      // Ensure the URL hash reflects the active project so the Studio
      // connects to the correct Hyperframes project on remount.
      window.location.hash = `#project/${encodeURIComponent(projectId)}`;
      
      // Small delay to let the server finish writing the composition
      // before the Studio tries to read it.
      setTimeout(reloadStudio, 600);
    };

    window.addEventListener('hyper-edit-success', handleSuccess);
    return () => window.removeEventListener('hyper-edit-success', handleSuccess);
  }, [reloadStudio]);

  const handleClearTimeline = async () => {
    try {
      await fetch('/api/hyper-edit/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ newProject: false })
      });
      reloadStudio();
      setToastMessage("Timeline cleared.");
      setTimeout(() => setToastMessage(null), 3000);
    } catch (e) {
      console.error(e);
    }
  };

  const handleNewProject = async () => {
    try {
      await fetch('/api/hyper-edit/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ newProject: true })
      });
      reloadStudio();
      setToastMessage("New project created.");
      setTimeout(() => setToastMessage(null), 3000);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSaveProject = () => {
    setToastMessage("Project saved successfully.");
    setTimeout(() => setToastMessage(null), 3000);
  };

  return (
    <div className="w-full h-full flex flex-col bg-slate-950 overflow-hidden relative studio-theme-custom">
      
      {/* Top Application Bar */}
      <div className="h-14 border-b border-slate-900 bg-slate-950/80 px-6 flex items-center justify-between shrink-0 z-50 relative">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-indigo-600 flex items-center justify-center mr-2">
            <Film className="w-4 h-4 text-white" />
          </div>
          <span className="font-black tracking-tight text-white">Hyperframes Studio</span>
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

      <AnimatePresence mode="wait">
        <motion.div 
          key={refreshKey}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full h-full flex-1 min-h-0"
        >
          <StudioApp />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
