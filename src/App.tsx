import { Facebook, Instagram, Share2, MessageSquare, BarChart3, LogOut, Settings, User, History, Send, Cpu, Plus, Info, CheckCircle2, AlertCircle, LayoutDashboard, Video, RefreshCw, X, TrendingUp, Users, Activity, Eye, ChevronRight, Calendar, ThumbsUp, MessageCircle, Edit2, Trash2, Check, Sparkles, Search, FileCode, Download, ShoppingBag } from 'lucide-react';
import { useState, useEffect, useRef, memo, useMemo, Fragment, forwardRef } from 'react';
import { auth, loginWithGoogle, loginAnonymously, logout, db } from './lib/firebase';
import { onAuthStateChanged, User as FirebaseUser } from 'firebase/auth';
import { collection, query, orderBy, onSnapshot, addDoc, serverTimestamp, where, doc, setDoc, deleteDoc, updateDoc } from 'firebase/firestore';
import { chatWithGemini, analyzeVideoWithGemini, generateCaptionWithGemini, LogEntry } from './lib/gemini';
import { safeFetchJson } from './lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import mermaid from 'mermaid';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';



import LogoSvg from './components/animations/LogoSvg';
import ThinkingOrb from './components/animations/ThinkingOrb';
import { AnimationState } from './components/animations/types';
import { DEFAULT_CONFIG, GRADIENT_PRESETS } from './components/animations/data';
import AIEditorPanel from './components/AIEditorPanel';
import HyperframesStudioFrame from './components/HyperframesStudioFrame';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const MorphingLogo = ({ isThinking, size = 64 }: { isThinking: boolean, size?: number }) => {
  const [internalState, setInternalState] = useState<AnimationState>(isThinking ? 'orb-thinking' : 'logo');
  const config = useMemo(() => ({ ...DEFAULT_CONFIG, size }), [size]);
  const preset = useMemo(() => GRADIENT_PRESETS.find(p => p.id === 'cosmic') || GRADIENT_PRESETS[0], []);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;

    if (isThinking) {
      setInternalState('morphing-to-orb');
      timeoutId = setTimeout(() => {
        setInternalState('orb-thinking');
      }, config.transitionSpeed * 1000);
    } else {
      setInternalState('morphing-to-logo');
      timeoutId = setTimeout(() => {
        setInternalState('logo');
      }, config.transitionSpeed * 1000);
    }

    return () => clearTimeout(timeoutId);
  }, [isThinking, config.transitionSpeed]);

  return (
    <div className="relative flex items-center justify-center shrink-0 cursor-default" style={{ width: size, height: size }}>
       <LogoSvg state={internalState} preset={preset} transitionSpeed={config.transitionSpeed} size={size} />
       <ThinkingOrb state={internalState} preset={preset} config={config} />
    </div>
  );
};

const MiniLogo = memo(({ size = 40 }: { size?: number }) => {
  const preset = useMemo(() => GRADIENT_PRESETS.find(p => p.id === 'cosmic') || GRADIENT_PRESETS[0], []);
  return (
    <div className="relative flex items-center justify-center shrink-0 cursor-default bg-white border border-slate-200/50 rounded-xl shadow-sm" style={{ width: size, height: size }}>
       <LogoSvg state="logo" preset={preset} transitionSpeed={0.3} size={size - 8} />
    </div>
  );
});

// Interfaces
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: any;
  isPublishedPost?: boolean;
  postData?: any;
  researchVideos?: any[];
  productsData?: any;
}

interface ChatSession {
  id: string;
  title: string;
  updatedAt: any;
}

// Components
mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
});

const LinkPreview = ({ url }: { url: string }) => {
  const isYoutube = url.includes('youtube.com') || url.includes('youtu.be');
  const isInstagram = url.includes('instagram.com');
  const isFacebook = url.includes('facebook.com');

  let thumbnailUrl = null;
  if (isYoutube) {
    const videoId = url.match(/(?:youtu\.be\/|youtube\.com(?:\/embed\/|\/v\/|\/watch\?v=|\/user\/\S+|\/ytscreeningroom\?v=))([\w-]{11})/)?.[1];
    if (videoId) {
      thumbnailUrl = `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`;
    }
  }

  if (!isYoutube && !isInstagram && !isFacebook) return null;

  return (
    <div className="mt-4 p-4 bg-slate-50 rounded-2xl border border-slate-100 flex flex-col gap-3 group">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isInstagram && <Instagram className="w-4 h-4 text-rose-500" />}
          {isFacebook && <Facebook className="w-4 h-4 text-blue-600" />}
          {isYoutube && <Video className="w-4 h-4 text-red-600" />}
          <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest truncate max-w-[200px]">
            {isInstagram ? 'Instagram Content' : isFacebook ? 'Facebook Content' : isYoutube ? 'YouTube Video' : 'Shared Link'}
          </span>
        </div>
        <a href={url} target="_blank" rel="noopener noreferrer" className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300 transition-all">
          <Share2 className="w-3.5 h-3.5" />
        </a>
      </div>
      
      {thumbnailUrl && (
        <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-200 border border-slate-100">
           <img src={thumbnailUrl} alt="Video thumbnail" className="w-full h-full object-cover" />
           <div className="absolute inset-0 flex items-center justify-center bg-black/10 group-hover:bg-black/20 transition-colors">
              <div className="w-10 h-10 rounded-full bg-white/90 shadow-xl flex items-center justify-center">
                 <Video className="w-5 h-5 text-slate-900 ml-0.5" />
              </div>
           </div>
        </div>
      )}

      {!thumbnailUrl && (isInstagram || isFacebook) && (
        <div className="flex items-center gap-3">
           <div className="w-10 h-10 rounded-xl bg-white border border-slate-100 flex items-center justify-center shrink-0">
              {isInstagram ? <Instagram className="w-5 h-5 text-rose-500" /> : <Facebook className="w-5 h-5 text-blue-600" />}
           </div>
           <div className="flex-1 min-w-0">
              <p className="text-xs font-bold text-slate-700 truncate">{url}</p>
              <p className="text-[10px] text-slate-400 font-medium">Click to view on {isInstagram ? 'Instagram' : 'Facebook'}</p>
           </div>
        </div>
      )}
    </div>
  );
};

const MermaidChart = ({ chart }: { chart: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const renderChart = async () => {
      if (containerRef.current) {
        try {
          const sanitizedChart = chart.replace(/"/g, "'");
          const { svg } = await mermaid.render(`mermaid-${Math.random().toString(36).substring(7)}`, sanitizedChart);
          if (containerRef.current) {
            containerRef.current.innerHTML = svg;
          }
        } catch (error) {
          console.error('Mermaid render error:', error);
          if (containerRef.current) containerRef.current.innerHTML = '<span class="text-rose-500 text-xs">Error rendering chart</span>';
        }
      }
    };
    renderChart();
  }, [chart]);

  return <div ref={containerRef} className="mermaid-chart flex justify-center w-full my-6 bg-white p-4 rounded-2xl border border-slate-100 shadow-sm" />;
};

const ChartRenderer = ({ config }: { config: any }) => {
  if (!config) return null;
  const { type, data, xAxis, yAxis, title } = config;

  return (
    <div className="w-full my-6 p-8 bg-white rounded-3xl border border-slate-100 shadow-sm relative overflow-hidden">
      {title && <h4 className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-6">{title}</h4>}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'bar' ? (
            <BarChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey={xAxis} stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} dy={10} fontStyle="normal" />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip 
                cursor={{ fill: 'rgba(241, 245, 249, 0.5)' }}
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #f1f5f9', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)', fontSize: '13px', color: '#0f172a' }}
              />
              <Bar dataKey={yAxis} fill="#0f172a" radius={[6, 6, 0, 0]} barSize={24} />
            </BarChart>
          ) : type === 'line' ? (
            <LineChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey={xAxis} stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} dy={10} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #f1f5f9', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)', color: '#0f172a' }}
              />
              <Line type="monotone" dataKey={yAxis} stroke="#0f172a" strokeWidth={3} dot={{ fill: '#0f172a', strokeWidth: 2, r: 4, stroke: '#fff' }} activeDot={{ r: 6, strokeWidth: 0 }} />
            </LineChart>
          ) : type === 'pie' ? (
            <PieChart>
              <Pie
                data={data}
                innerRadius={60}
                outerRadius={80}
                paddingAngle={2}
                dataKey={yAxis}
              >
                {data.map((_: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={['#0f172a', '#334155', '#64748b', '#94a3b8'][index % 4]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #f1f5f9', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}
              />
            </PieChart>
          ) : null}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const MessageBubble = memo(forwardRef(({ message }: { message: Message }, ref: any) => {
  const isAi = message.role === 'assistant';
  
  if (message.isPublishedPost && message.postData) {
    const { message: postBody, mediaBase64, mediaType, platform, accountName, fileUrl, thumbnailUrl } = message.postData;
    const isVideo = mediaType && mediaType.startsWith('video/');

    return (
      <motion.div ref={ref} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex w-full mb-8 justify-center">
        <div className="w-full max-w-sm bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
           <div className="p-4 flex items-center gap-3">
              <div className={cn(
                "w-10 h-10 rounded-full flex items-center justify-center shrink-0",
                platform === 'instagram' ? "bg-rose-100" : "bg-indigo-100"
              )}>
                 {platform === 'instagram' ? <Instagram className="w-5 h-5 text-rose-600" /> : <Facebook className="w-5 h-5 text-indigo-600" />}
              </div>
              <div className="flex flex-col">
                 <span className="text-sm font-semibold text-slate-900">{accountName || (platform === 'instagram' ? 'Your Instagram' : 'Your Facebook Page')}</span>
                 <span className="text-[11px] text-slate-500 font-medium">Just now · Published to {platform === 'instagram' ? 'Instagram' : 'Facebook'}</span>
              </div>
           </div>
           
           {postBody && (
              <div className="px-4 pb-3 text-[14px] text-slate-800 whitespace-pre-wrap font-normal">
                {postBody}
              </div>
           )}

           {(mediaBase64 || fileUrl) && (
             <div className="w-full bg-slate-100 flex items-center justify-center relative group/media">
               {isVideo ? (
                 <video src={mediaBase64 || fileUrl} poster={thumbnailUrl} controls className="w-full h-auto max-h-96 object-contain" />
               ) : (
                 <img src={mediaBase64 || fileUrl} alt="Post media" className="w-full h-auto max-h-96 object-contain" />
               )}
               {platform === 'instagram' && (
                 <div className="absolute top-3 right-3 bg-white/20 backdrop-blur-md p-2 rounded-xl border border-white/30 shadow-xl pointer-events-none">
                    <Instagram className="w-4 h-4 text-white" />
                 </div>
               )}
             </div>
           )}

           <div className="px-4 py-3 border-t border-slate-100 flex items-center justify-between text-slate-500 text-sm font-medium">
              <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors"><ThumbsUp className="w-4 h-4"/> Like</div>
              <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors"><MessageCircle className="w-4 h-4"/> Comment</div>
              <div className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors"><Share2 className="w-4 h-4"/> Share</div>
           </div>
        </div>
      </motion.div>
    );
  }

  let chartConfig = null;
  let textContent = message.content;
  if (isAi && message.content.includes('[CHART]')) {
    const chartMatch = message.content.match(/\[CHART\]([\s\S]*?)(\[\/CHART\]|$)/);
    if (chartMatch) {
      let jsonStr = chartMatch[1].trim();
      // Clean up markdown code blocks if AI added them
      if (jsonStr.startsWith('```json')) jsonStr = jsonStr.replace(/^```json/, '');
      if (jsonStr.startsWith('```')) jsonStr = jsonStr.replace(/^```/, '');
      if (jsonStr.endsWith('```')) jsonStr = jsonStr.replace(/```$/, '');
      jsonStr = jsonStr.trim();
      
      // Extract only the JSON object in case model appended conversational text without [/CHART]
      const startIdx = jsonStr.indexOf('{');
      const endIdx = jsonStr.lastIndexOf('}');
      if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
        jsonStr = jsonStr.substring(startIdx, endIdx + 1);
      }
      
      try {
        chartConfig = JSON.parse(jsonStr);
        // Remove only the entire chart block from the message
        textContent = textContent.replace(chartMatch[0], '').trim();
      } catch (e) {
        console.error('Failed to parse chart config', e, jsonStr);
      }
    }
  }

  // Parse interactive questions to hide them from the main bubble
  if (isAi && textContent.includes('[QUESTION]')) {
    const qMatch = textContent.match(/\[QUESTION\]([\s\S]*?)\[\/QUESTION\]/);
    if (qMatch) {
      textContent = textContent.replace(qMatch[0], '').trim();
    } else {
      const qMatchFallback = textContent.match(/\[QUESTION\]([\s\S]*?(?=\[|$))/);
      if (qMatchFallback) {
        textContent = textContent.replace(qMatchFallback[0], '').trim();
      }
    }
  }

  const urls = textContent.match(/(https?:\/\/[^\s]+)/g) || [];
  const mediaBase64 = (message as any).mediaBase64;
  const mediaType = (message as any).mediaType;
  const thumbnailUrl = (message as any).thumbnailUrl;
  const isVideo = mediaType && mediaType.startsWith('video/');

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex w-full mb-8",
        isAi ? "justify-start" : "justify-end"
      )}
    >
      <div className={cn(
        "max-w-[80%] rounded-3xl px-6 py-5 leading-relaxed",
        isAi 
          ? "bg-white text-slate-800 border border-slate-100 shadow-sm" 
          : "bg-slate-100 text-slate-900 border border-slate-200/50"
      )}>
        <div className={cn(
          "flex items-center gap-3 mb-4 text-[12px] font-bold tracking-wider uppercase",
          isAi ? "text-slate-500" : "text-slate-400"
        )}>
          {isAi ? <MiniLogo size={40} /> : <div className="w-10 h-10 rounded-full flex items-center justify-center bg-slate-200/50 border border-slate-300/50"><User className="w-5 h-5 text-slate-500" /></div>}
          {isAi ? <span className="pt-1">Decision Intelligence</span> : <span className="pt-1">You</span>}
        </div>

        {mediaBase64 && (
          <div className="mb-4 rounded-2xl overflow-hidden border border-slate-200 bg-slate-50">
            {isVideo ? (
              <video src={mediaBase64} poster={thumbnailUrl} controls className="w-full max-h-80 object-contain" />
            ) : (
              <img src={mediaBase64} alt="Attached media" className="w-full max-h-80 object-contain" />
            )}
          </div>
        )}

        <div className={cn(
          "markdown-body text-base font-normal leading-[1.6] break-words overflow-hidden",
          isAi ? "prose prose-slate max-w-[100%] prose-p:mb-3 prose-pre:max-w-full prose-pre:overflow-x-auto custom-scrollbar" : "whitespace-pre-wrap"
        )}>
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '');
                if (!inline && match && match[1] === 'mermaid') {
                  return <MermaidChart chart={String(children).replace(/\n$/, '')} />;
                }
                const isCodeBlock = !inline && match;
                if (isCodeBlock) {
                  const fileExtension = match[1];
                  const codeContent = String(children).replace(/\n$/, '');
                  const downloadFile = () => {
                    const blob = new Blob([codeContent], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `file.${fileExtension}`;
                    a.click();
                    URL.revokeObjectURL(url);
                  };
                  return (
                    <div className="relative group rounded-xl overflow-hidden my-4 border border-slate-200 shadow-sm bg-white">
                      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-50 border-b border-slate-200">
                        <span className="text-xs font-semibold text-slate-600 uppercase tracking-widest">{fileExtension} File</span>
                        <button 
                          onClick={downloadFile} 
                          title="Download File"
                          className="px-2 py-1 bg-white border border-slate-200 hover:bg-slate-50 rounded-md flex items-center gap-1.5 transition-colors text-slate-600 shadow-sm active:scale-95"
                        >
                          <Download className="w-3.5 h-3.5" /> 
                          <span className="text-[10px] font-bold">Download</span>
                        </button>
                      </div>
                      <div className="p-0 overflow-x-auto bg-slate-900 custom-scrollbar">
                        <pre className="!m-0 !p-4 !bg-transparent !text-[13px] text-slate-50 flex min-w-full float-left">
                          <code className={className} {...props}>
                            {children}
                          </code>
                        </pre>
                      </div>
                    </div>
                  );
                }
                
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              }
            }}
          >
            {textContent}
          </ReactMarkdown>
        </div>
        {urls.length > 0 && urls.map((url, idx) => (
          <LinkPreview key={idx} url={url} />
        ))}
        {message.researchVideos && message.researchVideos.length > 0 && (
          <div className="mt-6 mb-2 w-full">
            <h4 className="text-xs font-black tracking-widest uppercase text-slate-500 mb-4 flex items-center gap-2">
              <Video className="w-4 h-4"/> Analyzed Media ({message.researchVideos.length})
            </h4>
            <div className="flex overflow-x-auto gap-3 pb-4 snap-x snap-mandatory custom-scrollbar w-full" style={{ scrollPaddingLeft: '0.1rem' }}>
              {message.researchVideos.map((video, idx) => (
                <div key={video.id || idx} className="snap-start shrink-0 w-[120px] bg-slate-50 rounded-xl border border-slate-200 overflow-hidden flex flex-col group hover:shadow-md transition-shadow relative">
                  <div className="aspect-[9/16] relative bg-black">
                     <img src={video.thumbnail} alt="Thumbnail preview" className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity" />
                     <div className="absolute inset-x-0 bottom-0 pt-10 pb-2 px-2 bg-gradient-to-t from-black/90 via-black/40 to-transparent pointer-events-none flex flex-col gap-1 justify-end">
                        <div className="flex items-center gap-1 text-white/90 drop-shadow-md">
                          <Eye className="w-3 h-3"/> 
                          <span className="font-bold text-[10px]">{video.metrics?.views || 'N/A'}</span>
                        </div>
                        <div className="flex items-center gap-1.5 text-white/80 drop-shadow-md">
                          <span className="flex items-center gap-0.5 text-[9px] font-medium"><ThumbsUp className="w-2 h-2"/> {video.metrics?.likes || '0'}</span>
                          <span className="flex items-center gap-0.5 text-[9px] font-medium"><MessageCircle className="w-2 h-2"/> {video.metrics?.comments || '0'}</span>
                        </div>
                     </div>
                  </div>
                  <div className="p-2 flex-1 flex flex-col">
                     <div className="text-[9px] font-bold text-slate-400 mb-1 truncate" title={`@${video.author}`}>@{video.author}</div>
                     <p className="text-[10px] font-medium text-slate-700 line-clamp-3 leading-snug flex-1" title={video.hook}>{video.hook || "Video content"}</p>
                     <a href={video.url} target="_blank" rel="noopener noreferrer" className="mt-2 py-1 bg-slate-200 hover:bg-slate-300 text-slate-700 text-[9px] font-bold rounded-lg flex items-center justify-center transition-colors">
                        View <ChevronRight className="w-2.5 h-2.5 ml-0.5" />
                     </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {message.productsData && (
          <div className="mt-6 w-full">
            <h4 className="text-[11px] font-black tracking-widest uppercase text-slate-500 mb-4 flex items-center gap-2">
              <ShoppingBag className="w-4 h-4 text-emerald-500" /> Sourced Products
            </h4>
            
            {message.productsData.alibaba && message.productsData.alibaba.length > 0 && (
              <div className="mb-6">
                <h5 className="text-[10px] font-bold uppercase text-amber-600 mb-2.5">Alibaba Matches</h5>
                <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {message.productsData.alibaba.map((product: any, idx: number) => (
                    <a key={idx} href={product.url} target="_blank" rel="noopener noreferrer" className="group flex flex-col bg-slate-50 rounded-xl border border-slate-200 overflow-hidden hover:shadow-md transition-shadow">
                      <div className="aspect-[4/3] bg-slate-200 relative overflow-hidden">
                        {product.image ? (
                          <img src={product.image} alt={product.title} className="w-full h-full object-cover transition-transform group-hover:scale-105" />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center text-slate-400"><ShoppingBag className="w-6 h-6 opacity-30" /></div>
                        )}
                      </div>
                      <div className="p-3 flex-1 flex flex-col justify-between">
                        <div>
                           <div className="font-bold text-slate-900 text-sm mb-1">{product.price}</div>
                           <p className="text-[11px] text-slate-600 leading-snug line-clamp-2 mb-2" title={product.title}>{product.title}</p>
                        </div>
                        <div className="text-[10px] font-medium text-slate-500 flex items-center gap-1.5 mt-auto">
                           <Activity className="w-3 h-3"/> MOQ: {product.minOrder}
                        </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}

            {message.productsData.aliexpress && message.productsData.aliexpress.length > 0 && (
              <div className="mb-2">
                <h5 className="text-[10px] font-bold uppercase text-red-500 mb-2.5">AliExpress Matches</h5>
                <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                  {message.productsData.aliexpress.map((product: any, idx: number) => (
                    <a key={idx} href={product.url} target="_blank" rel="noopener noreferrer" className="group flex flex-col bg-slate-50 rounded-xl border border-slate-200 overflow-hidden hover:shadow-md transition-shadow">
                      <div className="aspect-[4/3] bg-slate-200 relative overflow-hidden">
                        {product.image ? (
                          <img src={product.image} alt={product.title} className="w-full h-full object-cover transition-transform group-hover:scale-105" />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center text-slate-400"><ShoppingBag className="w-6 h-6 opacity-30" /></div>
                        )}
                      </div>
                      <div className="p-3 flex-1 flex flex-col justify-between">
                         <div>
                            <div className="font-bold text-slate-900 text-sm mb-1">{product.price}</div>
                            <p className="text-[11px] text-slate-600 leading-snug line-clamp-2 mb-2" title={product.title}>{product.title}</p>
                         </div>
                         <div className="flex items-center gap-2 text-[10px] font-medium text-slate-500 mt-auto">
                            <span className="flex items-center gap-0.5"><Sparkles className="w-3 h-3 text-amber-500" /> {product.rating}</span>
                            <span className="flex items-center gap-0.5"><History className="w-3 h-3" /> {product.orders} sold</span>
                         </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {chartConfig && <ChartRenderer config={chartConfig} />}
      </div>
    </motion.div>
  );
}));

enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
  }
}

function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
    },
    operationType,
    path
  }
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  // In a real app we might show a toast, but for debugging we'll log it clearly
}

const AccountDropdown = memo(function AccountDropdown({ fbData, igData, selectedAccount, setSelectedAccount, direction = 'down' }: any) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [dropdownRef]);

  return (
      <div className="relative z-20" ref={dropdownRef}>
          <button type="button" onClick={() => setIsOpen(!isOpen)} className="px-4 py-2 rounded-xl text-sm font-semibold transition-all bg-white text-slate-800 shadow-sm border border-slate-200/50 flex items-center gap-2 hover:bg-slate-50 active:scale-95">
            {selectedAccount === 'auto' ? 'All Accounts' : selectedAccount}
            <svg className={cn("w-4 h-4 text-slate-400 transition-transform duration-200", isOpen && "rotate-180")} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
          </button>
          <AnimatePresence>
          {isOpen && (
              <motion.div 
                initial={{ opacity: 0, y: direction === 'up' ? 10 : -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: direction === 'up' ? 10 : -10, scale: 0.95 }}
                transition={{ duration: 0.15, ease: "easeOut" }}
                className={cn("absolute right-0 w-56 bg-white border border-slate-200 shadow-lg rounded-2xl p-1.5 z-50", direction === 'up' ? "bottom-full mb-2" : "top-full mt-2")}>
                 <button key="all-accounts" type="button" onClick={() => { setSelectedAccount('auto'); setIsOpen(false); }} className={cn("w-full text-left px-3 py-2 rounded-xl text-xs font-semibold transition-colors", selectedAccount === 'auto' ? "bg-slate-100 text-slate-900" : "text-slate-600 hover:bg-slate-50")}>
                   All Accounts
                 </button>
                 <Fragment key="fb-section">
                   {fbData?.accounts ? fbData.accounts.map((acc: any) => (
                     <button key={acc.pageId || acc.id} type="button" onClick={() => { setSelectedAccount(acc.pageName || acc.name); setIsOpen(false); }} className={cn("w-full text-left px-3 py-2 rounded-xl text-xs font-semibold transition-colors flex items-center gap-2 mt-0.5", selectedAccount === (acc.pageName || acc.name) ? "bg-blue-50 text-blue-700" : "text-slate-600 hover:bg-slate-50")}>
                       <Facebook className="w-3 h-3 text-blue-600 shrink-0" /> <span className="truncate">{acc.pageName || acc.name}</span>
                     </button>
                   )) : fbData?.pageName ? (
                     <button key="fb-btn-single" type="button" onClick={() => { setSelectedAccount(fbData.pageName); setIsOpen(false); }} className={cn("w-full text-left px-3 py-2 rounded-xl text-xs font-semibold transition-colors flex items-center gap-2 mt-0.5", selectedAccount === fbData.pageName ? "bg-blue-50 text-blue-700" : "text-slate-600 hover:bg-slate-50")}>
                       <Facebook className="w-3 h-3 text-blue-600 shrink-0" /> <span className="truncate">{fbData.pageName}</span>
                     </button>
                   ) : null}
                 </Fragment>
                 <Fragment key="ig-section">
                   {igData?.accounts ? igData.accounts.map((acc: any) => (
                     <button key={acc.accountId || acc.username} type="button" onClick={() => { setSelectedAccount(acc.username); setIsOpen(false); }} className={cn("w-full text-left px-3 py-2 rounded-xl text-xs font-semibold transition-colors flex items-center gap-2 mt-0.5", selectedAccount === acc.username ? "bg-rose-50 text-rose-700" : "text-slate-600 hover:bg-slate-50")}>
                       <Instagram className="w-3 h-3 text-rose-600 shrink-0" /> <span className="truncate">@{acc.username}</span>
                     </button>
                   )) : igData?.username ? (
                     <button key="ig-btn-single" type="button" onClick={() => { setSelectedAccount(igData.username); setIsOpen(false); }} className={cn("w-full text-left px-3 py-2 rounded-xl text-xs font-semibold transition-colors flex items-center gap-2 mt-0.5", selectedAccount === igData.username ? "bg-rose-50 text-rose-700" : "text-slate-600 hover:bg-slate-50")}>
                       <Instagram className="w-3 h-3 text-rose-600 shrink-0" /> <span className="truncate">@{igData.username}</span>
                     </button>
                   ) : null}
                 </Fragment>
              </motion.div>
          )}
          </AnimatePresence>
      </div>
  );
});

function DashboardView({ fbData, igData, selectedAccount, setSelectedAccount, onRefresh, isRefreshing, onAnalyzeVideo, onStartAudit, auditState, savedSyntheses, setAuditState }: any) {
  const [selectedPost, setSelectedPost] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [sortOrder, setSortOrder] = useState<'latest' | 'oldest' | 'engagement'>('engagement');
  const [timeframe, setTimeframe] = useState<'all' | '7d' | '30d' | '90d'>('all');
  const [performance, setPerformance] = useState<'all' | 'winners' | 'losers'>('all');

  let posts: any[] = [];
  
  if (fbData?.accounts) {
    fbData.accounts.forEach((acc: any) => {
      if (selectedAccount === 'auto' || selectedAccount === (acc.pageName || acc.name)) {
        (acc.recentPosts || []).forEach((p: any) => {
           if (p.id) {
               posts.push({ 
                 ...p, 
                 platform: 'facebook', 
                 accountName: acc.pageName || acc.name, 
                 caption: p.message || p.caption || "",
                 _likes: Number(p.likes || 0), 
                 _comments: Number(p.comments || 0),
                 _shares: Number(p.shares || 0),
                 _reach: Number(p.reach || 0),
                 _engaged: Number(p.engaged || 0),
                 _views: Number(p.views || p.reach || 0),
                 _clicks: Number(p.clicks || 0),
                 timestamp: p.created_time || p.timestamp
               });
           }
        });
      }
    });
  }

  if (igData?.accounts) {
    igData.accounts.forEach((acc: any) => {
       if (selectedAccount === 'auto' || selectedAccount === acc.username) {
           (acc.recentPosts || []).forEach((p: any) => {
              if (p.id) {
                  posts.push({ 
                    ...p, 
                    platform: 'instagram', 
                    accountName: acc.username, 
                    _likes: Number(p.likes || 0), 
                    _comments: Number(p.comments || 0), 
                    _shares: Number(p.shares || 0),
                    _reach: Number(p.reach || p.impressions || 0),
                    _engaged: Number(p.likes || 0) + Number(p.comments || 0) + Number(p.shares || 0) + Number(p.saved || 0),
                    _views: Number(p.views || p.play_count || p.video_view_count || 0),
                    _clicks: Number(p.clicks || 0),
                    timestamp: p.timestamp
                  });
              }
           });
       }
    });
  }

  // Filter by Timeframe
  if (timeframe !== 'all') {
    const now = new Date();
    const days = timeframe === '7d' ? 7 : timeframe === '30d' ? 30 : 90;
    const threshold = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
    posts = posts.filter(p => new Date(p.timestamp) >= threshold);
  }

  // Calculate Performance Threshold (Average Engagement)
  const totalEngagement = posts.reduce((acc, p) => acc + (p._engaged || (p._likes + p._comments)), 0);
  const avgEngagement = posts.length > 0 ? totalEngagement / posts.length : 0;

  // Filter by Performance
  if (performance === 'winners') {
    posts = posts.filter(p => (p._engaged || (p._likes + p._comments)) > avgEngagement);
  } else if (performance === 'losers') {
    posts = posts.filter(p => (p._engaged || (p._likes + p._comments)) <= avgEngagement);
  }

  // Sort Posts
  posts.sort((a, b) => {
    if (sortOrder === 'latest') return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    if (sortOrder === 'oldest') return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    return (b._likes + b._comments) - (a._likes + a._comments);
  });

  const videoPosts = posts.filter(p => p.source || (p.media_type && p.media_type.toLowerCase().includes('video')) || (p.media_url && p.media_url.includes('video')));

  return (
    <div className="flex-1 overflow-y-auto bg-[#FAFAFA] md:p-10 p-4 custom-scrollbar">
       <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
              <div>
                <h1 className="text-3xl font-black text-slate-900 tracking-tight mb-2">Content Intelligence</h1>
                <p className="text-slate-500 font-medium">Analyze and discover performing assets</p>
              </div>
              <div className="flex items-center gap-3">
                  <button 
                    onClick={onRefresh} 
                    disabled={isRefreshing || auditState.isAuditing}
                    className="h-12 px-6 rounded-2xl text-sm font-bold transition-all bg-white text-slate-800 shadow-sm border border-slate-200 flex items-center gap-2 hover:border-slate-300 disabled:opacity-50 active:scale-[0.98]"
                  >
                    <RefreshCw className={cn("w-4 h-4", isRefreshing ? "animate-spin" : "")} />
                    Sync Assets
                  </button>
                  <AccountDropdown fbData={fbData} igData={igData} selectedAccount={selectedAccount} setSelectedAccount={setSelectedAccount} direction="down" />
              </div>
          </div>

          <div className="mb-8 flex flex-wrap items-center gap-4">
             <div className="flex items-center gap-2 bg-white p-1 rounded-2xl border border-slate-200">
                <button onClick={() => setSortOrder('engagement')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", sortOrder === 'engagement' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>Winners First</button>
                <button onClick={() => setSortOrder('latest')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", sortOrder === 'latest' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>Latest</button>
                <button onClick={() => setSortOrder('oldest')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", sortOrder === 'oldest' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>Oldest</button>
             </div>

             <div className="flex items-center gap-2 bg-white p-1 rounded-2xl border border-slate-200">
                <button onClick={() => setTimeframe('all')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", timeframe === 'all' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>All Time</button>
                <button onClick={() => setTimeframe('7d')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", timeframe === '7d' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>7D</button>
                <button onClick={() => setTimeframe('30d')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", timeframe === '30d' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>30D</button>
                <button onClick={() => setTimeframe('90d')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", timeframe === '90d' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>90D</button>
             </div>

             <div className="flex items-center gap-2 bg-white p-1 rounded-2xl border border-slate-200">
                <button onClick={() => setPerformance('all')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", performance === 'all' ? "bg-slate-900 text-white shadow-lg" : "text-slate-400 hover:text-slate-600")}>All Assets</button>
                <button onClick={() => setPerformance('winners')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", performance === 'winners' ? "bg-emerald-600 text-white shadow-lg" : "text-emerald-600/60 hover:text-emerald-600")}>Winning</button>
                <button onClick={() => setPerformance('losers')} className={cn("px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all", performance === 'losers' ? "bg-rose-600 text-white shadow-lg" : "text-rose-600/60 hover:text-rose-600")}>Losing</button>
             </div>
          </div>

          {/* Bulk Audit Control Center */}
          <div className="mb-16">
             <div className="bg-slate-900 rounded-[40px] p-8 md:p-12 relative overflow-hidden shadow-2xl shadow-slate-900/20">
                <div className="absolute top-0 right-0 w-1/2 h-full opacity-10 pointer-events-none">
                   <div className="absolute inset-0 bg-gradient-to-l from-indigo-500 to-transparent" />
                   <div className="absolute top-10 right-10 w-64 h-64 border-8 border-indigo-500 rounded-full blur-3xl" />
                </div>
                
                <div className="relative z-10">
                   <div className="flex items-center gap-4 mb-6">
                      <div className="w-14 h-14 rounded-2xl bg-indigo-500 flex items-center justify-center text-white border border-indigo-400 group relative">
                        <div className="absolute inset-0 bg-white/20 blur animate-pulse rounded-2xl" />
                        <TrendingUp className="w-7 h-7 relative z-10" />
                      </div>
                      <div>
                        <div className="flex items-center gap-4">
                           <h2 className="text-2xl font-black text-white">Full Intelligence Synthesis</h2>
                           {savedSyntheses && savedSyntheses.length > 0 && (
                             <button 
                               onClick={() => setShowHistory(!showHistory)}
                               className="px-3 py-1 bg-white/10 text-indigo-300 text-[10px] font-black uppercase rounded-lg hover:bg-white/20 transition-colors border border-white/10"
                             >
                               {showHistory ? 'Hide History' : `History (${savedSyntheses.length})`}
                             </button>
                           )}
                        </div>
                        <p className="text-indigo-300 text-xs font-bold uppercase tracking-widest leading-none mt-1">Winning & Losing Pattern Detection</p>
                      </div>
                   </div>

                   {showHistory && savedSyntheses && savedSyntheses.length > 0 && (
                      <div className="mb-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                         {savedSyntheses.slice(0, 6).map((syn: any) => (
                            <div 
                              key={syn.id} 
                              onClick={() => {
                                 setAuditState((prev: any) => ({
                                   ...prev,
                                   report: { winners: syn.winners, losers: syn.losers }
                                 }));
                                 setShowHistory(false);
                                 window.scrollTo({ top: 0, behavior: 'smooth' });
                              }}
                              className="bg-white/5 border border-white/10 rounded-2xl p-4 hover:bg-white/10 transition-all cursor-pointer group"
                            >
                               <div className="flex items-center justify-between mb-3">
                                  <div className="flex items-center gap-2">
                                     {syn.platform === 'facebook' ? <Facebook className="w-3 h-3 text-blue-400" /> : <Instagram className="w-3 h-3 text-rose-400" />}
                                     <span className="text-[10px] font-black text-white uppercase truncate">@{syn.accountName}</span>
                                  </div>
                                  <span className="text-[9px] text-slate-500 font-mono">
                                     {syn.createdAt?.toDate().toLocaleDateString()}
                                  </span>
                               </div>
                               <p className="text-[11px] text-slate-300 font-medium line-clamp-2 italic mb-3">"{syn.winners.slice(0, 60)}..."</p>
                               <div className="flex items-center gap-2">
                                  <span className="text-[9px] font-black text-indigo-400 uppercase tracking-widest">Synthesis for {syn.videoCount} Assets</span>
                               </div>
                            </div>
                         ))}
                      </div>
                   )}

                   <p className="text-slate-300 font-medium mb-10 max-w-2xl leading-relaxed">
                      Automatically process every video in your feed frame-by-frame. We'll extract visual hooks, pacing signatures, and verbal CTAs to identify exactly why your top posts succeeded.
                   </p>

                   {auditState.isAuditing ? (
                      <div className="space-y-6">
                         <div className="flex items-center justify-between mb-2">
                            <span className="text-white text-sm font-black uppercase tracking-widest">
                               Processing Feed: {auditState.progress.current}/{auditState.progress.total}
                            </span>
                            <span className="text-indigo-400 text-xs font-mono">{Math.round((auditState.progress.current / (auditState.progress.total || 1)) * 100)}%</span>
                         </div>
                         <div className="h-4 bg-white/10 rounded-full overflow-hidden border border-white/5">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${(auditState.progress.current / (auditState.progress.total || 1)) * 100}%` }}
                              className="h-full bg-indigo-500"
                            />
                         </div>
                         <div className="flex items-center gap-3">
                            <div className="flex gap-1">
                    {videoPosts.length === 0 ? null : [1,2,3].map(i => <motion.div key={`audit-dot-${i}`} animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.2 }} className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />)}
                            </div>
                            <span className="text-indigo-300 text-[11px] font-bold uppercase italic">{auditState.currentTask || "Initializing pipeline..."}</span>
                         </div>
                      </div>
                   ) : auditState.report ? (
                      <div className="space-y-8">
                         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pb-8 border-b border-white/10">
                            <div className="bg-white/5 rounded-3xl p-6 border border-white/10">
                               <div className="flex items-center gap-3 mb-6">
                                  <div className="w-8 h-8 rounded-lg bg-green-500/20 text-green-400 flex items-center justify-center">
                                     <TrendingUp className="w-4 h-4" />
                                  </div>
                                  <h4 className="text-xs font-black text-white uppercase tracking-widest">Winning Patterns</h4>
                               </div>
                               <div className="prose prose-invert prose-sm">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{auditState.report.winners}</ReactMarkdown>
                               </div>
                            </div>
                            <div className="bg-white/5 rounded-3xl p-6 border border-white/10">
                               <div className="flex items-center gap-3 mb-6">
                                  <div className="w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center">
                                     <AlertCircle className="w-4 h-4" />
                                  </div>
                                  <h4 className="text-xs font-black text-white uppercase tracking-widest">Growth Killers</h4>
                               </div>
                               <div className="prose prose-invert prose-sm">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{auditState.report.losers}</ReactMarkdown>
                               </div>
                            </div>
                         </div>
                         <div className="flex items-center justify-between">
                            <button 
                              onClick={() => onStartAudit(videoPosts, selectedAccount)}
                              className="px-8 py-4 bg-white text-slate-900 font-black rounded-2xl hover:bg-indigo-50 transition-all flex items-center gap-3"
                            >
                               <RefreshCw className="w-5 h-5" />
                               Re-Run Synthesis
                            </button>
                            <span className="text-slate-400 text-[10px] uppercase font-black tracking-widest">Synthesis Complete</span>
                         </div>
                      </div>
                   ) : (
                      <button 
                        onClick={() => onStartAudit(videoPosts, selectedAccount)}
                        disabled={videoPosts.length === 0}
                        className="h-16 px-10 bg-indigo-500 text-white font-black rounded-2xl hover:bg-indigo-600 transition-all active:scale-[0.98] shadow-xl shadow-indigo-500/20 flex items-center gap-4 group disabled:opacity-50"
                      >
                         <Cpu className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                         Start Account Synthesis ({videoPosts.length} Videos)
                      </button>
                   )}
                </div>
             </div>
          </div>

          <div className="space-y-8">
            <div className="flex items-center gap-4">
               <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400">Synced Assets</h3>
              <div className="h-px flex-1 bg-slate-200" />
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                <span className="text-[10px] font-black text-slate-500 uppercase">Live Cloud Sync</span>
              </div>
            </div>

            {posts.length === 0 ? (
              <div className="p-20 border-2 border-dashed border-slate-200 rounded-[40px] flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center text-slate-400 mb-6">
                  <Activity className="w-8 h-8" />
                </div>
                <h4 className="text-lg font-black text-slate-900 mb-2">No performing assets yet</h4>
                <p className="text-slate-500 font-medium max-w-xs mx-auto text-sm">
                  Connect your accounts or paste a profile link in the chat to discover assets.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-20">
                {posts.map((post, i) => (
                  <motion.div 
                    key={post.id} 
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    onClick={() => setSelectedPost(post)} 
                    className="bg-white border-2 border-transparent cursor-pointer hover:border-slate-200 hover:shadow-2xl hover:shadow-slate-200/50 transition-all rounded-[32px] overflow-hidden flex flex-col group p-2"
                  >
                    <div className="relative aspect-square rounded-[24px] overflow-hidden bg-slate-100 mb-6 border border-slate-50">
                       {(post.full_picture || post.thumbnail_url || post.media_url) ? (
                         <img src={post.full_picture || post.thumbnail_url || post.media_url} referrerPolicy="no-referrer" alt="Thumbnail" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110" />
                       ) : (
                        <div className="w-full h-full flex items-center justify-center text-slate-300">
                          <Video className="w-12 h-12" />
                        </div>
                       )}
                       <div className="absolute top-4 left-4 h-8 px-3 bg-white/90 backdrop-blur rounded-xl flex items-center gap-2 border border-white/50 shadow-sm">
                          {post.platform === 'facebook' ? (
                            <Facebook className="w-3.5 h-3.5 text-blue-600" />
                          ) : (
                            <Instagram className="w-3.5 h-3.5 text-rose-600" />
                          )}
                          <span className="text-[10px] font-black text-slate-900">
                            {post.platform === 'instagram' ? '@' : ''}{post.accountName}
                          </span>
                       </div>
                       <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-6">
                          <div className="flex items-center gap-2 text-white font-black text-xs">
                            <span className="h-2 w-2 rounded-full bg-rose-500" />
                            Click to Expand Focus
                          </div>
                       </div>
                    </div>
                    
                    <div className="px-5 pb-5 flex-1 flex flex-col">
                      <p className="text-sm text-slate-800 font-bold line-clamp-2 leading-relaxed mb-6 min-h-[40px]">
                        {post.caption || "Untitled Asset"}
                      </p>
                      
                      <div className="grid grid-cols-3 gap-2 mb-6 text-center">
                        <div className="bg-slate-50 rounded-2xl p-3 border border-slate-100 flex flex-col justify-center text-left">
                           <span className="block text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Engagement</span>
                           <span className="block text-slate-900 font-black text-sm">{post._engaged > 0 ? post._engaged : Number(post._likes) + Number(post._comments) + Number(post._shares)}</span>
                        </div>
                        <div className="bg-slate-50 rounded-2xl p-3 border border-slate-100 flex flex-col justify-center text-left">
                           <span className="block text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Views</span>
                           <span className={cn("block font-black text-sm", (post._views > 0 || post._reach > 0) ? "text-emerald-600" : "text-slate-400")}>
                             {post._views > 0 ? post._views : (post._reach > 0 ? post._reach : 'N/A')}
                           </span>
                        </div>
                        <div className="bg-slate-50 rounded-2xl p-3 border border-slate-100 flex flex-col justify-center text-left">
                           <span className="block text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Clicks</span>
                           <span className="block text-indigo-600 font-black text-sm">{post._clicks > 0 ? post._clicks : 'N/A'}</span>
                        </div>
                      </div>

                      <button 
                        onClick={(e) => { e.stopPropagation(); onAnalyzeVideo(post.source || post.media_url); }} 
                        className="w-full bg-slate-900 text-white font-black text-xs py-4 rounded-2xl flex items-center justify-center gap-2 hover:bg-slate-800 transition-all active:scale-[0.98]"
                      >
                        <Cpu className="w-4 h-4" />
                        Analyze Engagement
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
       </div>

       {/* Post Focus Modal */}
       <AnimatePresence>
          {selectedPost && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setSelectedPost(null)}
                className="absolute inset-0 bg-slate-900/60 backdrop-blur-md"
              />
              <motion.div 
                initial={{ scale: 0.95, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.95, opacity: 0, y: 20 }}
                className={cn(
                  "bg-white rounded-[48px] w-full overflow-hidden shadow-2xl relative z-10 flex flex-col text-left",
                  ((selectedPost.media_url && (selectedPost.media_url.includes('video') || selectedPost.media_type === 'VIDEO')) || selectedPost.source) 
                    ? "max-w-lg max-h-[85vh]" 
                    : "max-w-4xl max-h-[80vh] md:flex-row"
                )}
                onClick={e => e.stopPropagation()}
              >
                {/* Close Button Mob */}
                <button onClick={() => setSelectedPost(null)} className="absolute top-6 right-6 z-20 w-12 h-12 flex items-center justify-center rounded-2xl bg-white/20 backdrop-blur text-white md:text-slate-400 md:bg-slate-50 border border-white/30 md:border-slate-100 hover:scale-110 transition-all">
                  <X className="w-6 h-6" />
                </button>

                <div className={cn(
                  "flex-1 bg-slate-950 flex shadow-inner relative overflow-hidden group",
                  ((selectedPost.media_url && (selectedPost.media_url.includes('video') || selectedPost.media_type === 'VIDEO')) || selectedPost.source) ? "aspect-[9/16]" : ""
                )}>
                  {((selectedPost.media_url && (selectedPost.media_url.includes('video') || selectedPost.media_type === 'VIDEO')) || selectedPost.source) ? (
                      <video controls autoPlay className="w-full h-full object-cover" src={selectedPost.source || selectedPost.media_url} />
                  ) : (
                      <img src={selectedPost.full_picture || selectedPost.thumbnail_url || selectedPost.media_url} referrerPolicy="no-referrer" alt="Post thumbnail" className="w-full h-full object-contain" />
                  )}
                </div>
                
                <div className={cn(
                  "w-full flex flex-col bg-white overflow-y-auto",
                  ((selectedPost.media_url && (selectedPost.media_url.includes('video') || selectedPost.media_type === 'VIDEO')) || selectedPost.source) ? "p-6" : "md:w-[400px]"
                )}>
                  <div className={cn("p-2", !((selectedPost.media_url && (selectedPost.media_url.includes('video') || selectedPost.media_type === 'VIDEO')) || selectedPost.source) ? "p-8 pb-4" : "")}>
                      <div className="flex items-center gap-3 mb-8">
                        <div className={cn(
                          "w-10 h-10 rounded-xl flex items-center justify-center",
                          selectedPost.platform === 'facebook' ? "bg-blue-50 text-blue-600" : "bg-rose-50 text-rose-600"
                        )}>
                          {selectedPost.platform === 'facebook' ? <Facebook className="w-5 h-5" /> : <Instagram className="w-5 h-5" />}
                        </div>
                        <div>
                          <span className="block font-black text-slate-900">{selectedPost.platform === 'instagram' ? '@' : ''}{selectedPost.accountName}</span>
                          <span className="text-[10px] text-slate-400 font-black uppercase tracking-widest">Asset Focus</span>
                        </div>
                      </div>

                     <div className="bg-slate-50 rounded-3xl p-6 mb-8 border border-slate-100">
                        <p className="text-slate-800 font-bold text-lg leading-relaxed mb-4">
                          {selectedPost.caption || "Engagement content"}
                        </p>
                        <div className="flex items-center gap-2 text-slate-400">
                           <Calendar className="w-4 h-4" />
                           <span className="text-[10px] font-black uppercase tracking-widest">
                             {new Date(selectedPost.timestamp).toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}
                           </span>
                        </div>
                     </div>

                     <div className="space-y-4 mb-8">
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Metric Analysis</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden group">
                             <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <Activity className="w-12 h-12 text-rose-500" />
                             </div>
                             <div className="text-rose-500 font-black text-xl mb-1 flex items-center gap-2 relative z-10">
                               {selectedPost._likes} 
                             </div>
                             <span className="text-[9px] font-black text-slate-400 uppercase relative z-10">Likes</span>
                          </div>
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden group">
                             <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <MessageSquare className="w-12 h-12 text-blue-500" />
                             </div>
                             <div className="text-blue-500 font-black text-xl mb-1 flex items-center gap-2 relative z-10">
                               {selectedPost._comments}
                             </div>
                             <span className="text-[9px] font-black text-slate-400 uppercase relative z-10">Comments</span>
                          </div>
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden group">
                             <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <Share2 className="w-12 h-12 text-indigo-500" />
                             </div>
                             <div className="text-indigo-500 font-black text-xl mb-1 flex items-center gap-2 relative z-10">
                               {selectedPost._shares}
                             </div>
                             <span className="text-[9px] font-black text-slate-400 uppercase relative z-10">Shares</span>
                          </div>
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden group">
                             <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <Eye className="w-12 h-12 text-emerald-500" />
                             </div>
                             <div className="text-emerald-500 font-black text-xl mb-1 flex items-center gap-2 relative z-10">
                               {selectedPost._views > 0 ? selectedPost._views : (selectedPost._reach > 0 ? selectedPost._reach : '--')}
                             </div>
                             <span className="text-[9px] font-black text-slate-400 uppercase relative z-10">
                               {selectedPost._views > 0 ? 'Views' : 'Reach'}
                             </span>
                          </div>
                          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden group md:col-span-2">
                             <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <TrendingUp className="w-12 h-12 text-violet-500" />
                             </div>
                             <div className="text-violet-500 font-black text-xl mb-1 flex items-center gap-2 relative z-10">
                               {selectedPost._clicks > 0 ? selectedPost._clicks : '--'}
                             </div>
                             <span className="text-[9px] font-black text-slate-400 uppercase relative z-10">Link Clicks</span>
                          </div>
                        </div>
                     </div>

                     <button onClick={(e) => { e.stopPropagation(); setSelectedPost(null); onAnalyzeVideo(selectedPost.source || selectedPost.media_url); }} className="w-full bg-slate-900 text-white font-black py-5 text-sm rounded-[24px] flex items-center justify-center gap-3 hover:bg-slate-800 transition-all active:scale-[0.98] shadow-2xl">
                        <Cpu className="w-5 h-5 animate-pulse text-indigo-400" />
                        Run Deep Intelligence Report
                     </button>
                  </div>
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
     </div>
  );
}

export default function App() {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<string | null>(null);
  const [chatFileBase64, setChatFileBase64] = useState<string | null>(null);
  const [chatFileMimeType, setChatFileMimeType] = useState<string | null>(null);
  const [isChatPlusMenuOpen, setIsChatPlusMenuOpen] = useState(false);
  const [isChatFileMode, setIsChatFileMode] = useState(false);
  const [isProductResearchMode, setIsProductResearchMode] = useState(false);
  const chatFileInputRef = useRef<HTMLInputElement>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  const [fbConnected, setFbConnected] = useState(false);
  const [igConnected, setIgConnected] = useState(false);
  const [fbData, setFbData] = useState<any>(null);
  const [igData, setIgData] = useState<any>(null);
  const [isSidebarOpen, setSidebarOpen] = useState(() => typeof window !== 'undefined' ? window.innerWidth > 768 : true);
  const [selectedAccount, setSelectedAccount] = useState<string>('auto');
  const [currentView, setCurrentView] = useState<'chat' | 'dashboard' | 'studio'>('chat');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [auditState, setAuditState] = useState<{
    isAuditing: boolean,
    progress: { current: number, total: number },
    currentTask: string | null,
    report: { winners: string, losers: string } | null
  }>({
    isAuditing: false,
    progress: { current: 0, total: 0 },
    currentTask: null,
    report: null
  });
  const [savedAnalyses, setSavedAnalyses] = useState<Record<string, string>>({});
  const [savedSyntheses, setSavedSyntheses] = useState<any[]>([]);
  const [isAutoSyncing, setIsAutoSyncing] = useState(false);

  const [isComposerOpen, setIsComposerOpen] = useState(false);
  const [composerText, setComposerText] = useState('');
  const [composerFile, setComposerFile] = useState<File | null>(null);
  const [composerPreview, setComposerPreview] = useState<string | null>(null);
  const [composerBase64, setComposerBase64] = useState<string | null>(null);
  const [composerThumbnail, setComposerThumbnail] = useState<string | null>(null);
  const [postType, setPostType] = useState<'text' | 'image' | 'video' | 'reel' | 'story'>('text');
  const [postFormat, setPostFormat] = useState<'auto' | '1:1' | '4:5' | '9:16' | '16:9'>('auto');
  const [isPublishing, setIsPublishing] = useState(false);
  const [isGeneratingCaption, setIsGeneratingCaption] = useState(false);
  const [composerStatus, setComposerStatus] = useState<{ type: 'success' | 'error', msg: string } | null>(null);
  const [crossPostToIg, setCrossPostToIg] = useState(false);

  const [chatFileThumbnail, setChatFileThumbnail] = useState<string | null>(null);

  const getFileBase64 = (file: File): Promise<string> => new Promise((resolve) => {
     const reader = new FileReader();
     reader.onload = (e) => resolve(e.target?.result as string);
     reader.readAsDataURL(file);
  });

  const generateVideoThumbnail = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        video.currentTime = 0.5;
      };
      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(video, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.7));
        video.remove();
        URL.revokeObjectURL(video.src);
      };
      video.onerror = () => resolve('');
      video.src = URL.createObjectURL(file);
    });
  };

  const handleComposerFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
       const file = e.target.files[0];
       setComposerFile(file);
       const isVideo = file.type.startsWith('video/');
       setPostType(isVideo ? 'video' : 'image');
       if (isVideo) {
         setComposerPreview(URL.createObjectURL(file));
         getFileBase64(file).then(setComposerBase64); // Load in background
         generateVideoThumbnail(file).then(setComposerThumbnail);
       } else {
         const b64 = await getFileBase64(file);
         setComposerPreview(b64);
         setComposerBase64(b64);
         setComposerThumbnail(null);
       }
    }
  };

  const handleChatFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
       const file = e.target.files[0];
       const b64 = await getFileBase64(file);
       setChatFileBase64(b64);
       setChatFileMimeType(file.type);
       
       if (file.type.startsWith('video/')) {
         generateVideoThumbnail(file).then(setChatFileThumbnail);
       } else {
         setChatFileThumbnail(null);
       }
    }
    if (e.target) e.target.value = '';
  };

  const handleGenerateCaption = async () => {
    if (!user) return;
    setIsGeneratingCaption(true);
    try {
      setComposerStatus(null);
      const promptContext = composerText.trim() || (composerFile ? `Media attached: ${composerFile.name}` : "Something exciting");
      const mediaType = composerFile?.type || null;
      const patterns = auditState.report ? auditState.report : null;

      const isVideoPost = composerFile?.type.startsWith('video/');
      const mediaToUse = isVideoPost ? (composerThumbnail || null) : composerBase64;
      const mediaTypeToUse = isVideoPost ? (composerThumbnail ? 'image/jpeg' : null) : (composerFile?.type || null);

      const generated = await generateCaptionWithGemini(
        promptContext,
        mediaToUse,
        mediaTypeToUse,
        patterns,
        (log) => setLogs(prev => [...prev, log])
      );
      setComposerText(generated);
    } catch (e: any) {
      const errorMsg = e.message || "";
      if (errorMsg.includes('quota') || errorMsg.includes('limit') || errorMsg.includes('429')) {
        setComposerStatus({ type: 'error', msg: "AI Quota Exceeded. Please try again in a few minutes or write your own caption." });
      } else {
        setComposerStatus({ type: 'error', msg: `Could not generate caption: ${errorMsg}` });
      }
    } finally {
      setIsGeneratingCaption(false);
    }
  };

  const processImageForInstagram = (base64: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;
        
        // 1. Resize if too large (IG limit is 1440px wide usually, let's use 1200 for safety)
        const maxDim = 1200;
        if (width > maxDim || height > maxDim) {
          if (width > height) {
            height *= maxDim / width;
            width = maxDim;
          } else {
            width *= maxDim / height;
            height = maxDim;
          }
        }
        
        const aspectRatio = width / height;
        
        // 2. Adjust Aspect Ratio (0.8 to 1.91)
        let finalWidth = width;
        let finalHeight = height;
        let dx = 0;
        let dy = 0;
        
        if (aspectRatio < 0.8) {
          // Too tall (e.g. 9:16) -> we make it 4:5 by adding white padding on sides
          finalWidth = height * 0.8;
          dx = (finalWidth - width) / 2;
        } else if (aspectRatio > 1.91) {
          // Too wide -> we make it 1.91:1 by adding white padding top/bottom
          finalHeight = width / 1.91;
          dy = (finalHeight - height) / 2;
        }
        
        canvas.width = finalWidth;
        canvas.height = finalHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, finalWidth, finalHeight);
          ctx.drawImage(img, dx, dy, width, height);
          resolve(canvas.toDataURL('image/jpeg', 0.85));
        } else {
          resolve(base64);
        }
      };
      img.onerror = () => resolve(base64);
      img.src = base64;
    });
  };

  const handlePublish = async () => {
    if (selectedAccount === 'auto') {
      setComposerStatus({ type: 'error', msg: "Please select a specific account to post." });
      return;
    }

    // Try finding in Facebook accounts
    let acc = fbData?.accounts?.find((a: any) => a.pageName === selectedAccount || a.name === selectedAccount);
    let platformUsed = 'facebook';

    if (!acc) {
      // Try finding in Instagram accounts
      acc = igData?.accounts?.find((a: any) => a.username === selectedAccount);
      platformUsed = 'instagram';
    }

    if (!acc) {
      setComposerStatus({ type: 'error', msg: "Selected account not found or lacks publish permissions." });
      return;
    }

    setIsPublishing(true);
    setComposerStatus(null);
    try {
        let finalMediaBase64 = composerBase64;
        if (!finalMediaBase64 && composerFile) {
           setComposerStatus({ type: 'success', msg: "Processing media..." });
           finalMediaBase64 = await getFileBase64(composerFile);
        }
        
        let uploadId = undefined;
        
        // Optimization: For Instagram images, ensure aspect ratio and size are within limits
        if (platformUsed === 'instagram' && composerBase64 && postType === 'image') {
          try {
            finalMediaBase64 = await processImageForInstagram(composerBase64);
          } catch (e) {
            console.warn("Failed to process image for IG optimization, using original", e);
          }
        }

        if (finalMediaBase64 && finalMediaBase64.length > 500_000) {
           setComposerStatus({ type: 'success', msg: "Uploading media in chunks..." });
           const startRes = await fetch('/api/upload/start', { method: 'POST' }).then(r => r.json());
           uploadId = startRes.uploadId;
           
           const chunkSize = 500_000;
           for (let i = 0; i < finalMediaBase64.length; i += chunkSize) {
              const chunk = finalMediaBase64.slice(i, i + chunkSize);
              await fetch('/api/upload/chunk', {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({ uploadId, chunk })
              });
           }
           setComposerStatus({ type: 'success', msg: "Publishing post..." });
        }

        const data = await safeFetchJson('/api/publish', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify({
              platform: platformUsed,
              accountId: platformUsed === 'facebook' ? (acc.pageId || acc.id) : acc.accountId,
              message: composerText,
              mediaBase64: uploadId ? null : finalMediaBase64,
              uploadId: uploadId,
              postType: postType,
              accessToken: platformUsed === 'facebook' ? (acc.pageToken || fbData?.accessToken) : (acc.pageToken || igData?.accessToken),
              origin: window.location.origin,
           })
        });

        let crossPostData = null;
        let linkedIgId = null;
        
        if (platformUsed === 'facebook' && fbData?.accounts) {
           linkedIgId = fbData.accounts.find((a: any) => a.pageName === selectedAccount || a.name === selectedAccount)?.linkedInstagramAccountId;
        }

        if (crossPostToIg && platformUsed === 'facebook' && linkedIgId) {
             setComposerStatus({ type: 'success', msg: "Publishing to Instagram (Cross-post)..." });
             crossPostData = await safeFetchJson('/api/publish', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({
                  platform: 'instagram',
                  accountId: linkedIgId,
                  message: composerText,
                  mediaBase64: uploadId ? null : finalMediaBase64,
                  uploadId: uploadId,
                  postType: postType,
                  accessToken: acc.pageToken || fbData?.accessToken,
                  origin: window.location.origin,
               })
             });
        }

        const jobsToWait = [];
        if (data.jobId) jobsToWait.push({ name: 'Facebook', jobId: data.jobId, url: platformUsed, target: selectedAccount });
        if (crossPostData?.jobId) jobsToWait.push({ name: 'Instagram', jobId: crossPostData.jobId, url: 'instagram', target: 'Linked IG Account' });

        if (jobsToWait.length > 0) {
          // Poll for status of all jobs
          let allCompleted = false;
          let jobResultId = null;
          let failures = [];
          
          let pollAttempts = 0;
          const maxPolls = 120; // 10 minutes at 5s interval

          while (!allCompleted && pollAttempts < maxPolls) {
            await new Promise(r => setTimeout(r, 5000));
            allCompleted = true;
            for (let j of jobsToWait) {
               if ((j as any).completed || (j as any).failed) continue;
               try {
                 const statusData = await safeFetchJson(`/api/publish/status/${j.jobId}`);
                 if (statusData.status === 'completed') {
                    (j as any).completed = true;
                    if (!jobResultId) jobResultId = statusData.resultId;
                 } else if (statusData.status === 'error') {
                    (j as any).failed = true;
                    failures.push(`${j.name}: ${statusData.error}`);
                 } else {
                    allCompleted = false;
                 }
               } catch (e) {
                 console.warn("Poll error:", e);
               }
            }
            pollAttempts++;
          }

          if (failures.length > 0 && jobsToWait.every(j => (j as any).failed)) {
             throw new Error(failures.join(" | "));
          } else if (failures.length > 0) {
             setComposerStatus({ type: 'error', msg: `Partial success. ${failures.join(" | ")}` });
          } else {
             setComposerStatus({ type: 'success', msg: `Published successfully${crossPostData ? ' to both FB and IG' : ''}! You can now switch accounts and post again, or click Finish.` });
          }

          if (!jobsToWait.every(j => (j as any).failed) && currentSessionId) {
             try {
                await addDoc(collection(db, `chats/${currentSessionId}/messages`), {
                  content: `Successfully published content to ${selectedAccount}${crossPostToIg ? ' and linked Instagram' : ''}.`,
                  role: 'assistant',
                  timestamp: serverTimestamp(),
                  isPublishedPost: true,
                  postData: {
                     id: jobResultId,
                     platform: platformUsed,
                     accountName: selectedAccount,
                     message: composerText,
                     mediaBase64: composerFile?.type.startsWith('video/') ? null : composerBase64,
                     fileUrl: composerFile?.type.startsWith('video/') ? composerPreview : null, 
                     mediaType: composerFile?.type,
                     thumbnailUrl: composerThumbnail
                  }
                });
                await setDoc(doc(db, 'chats', currentSessionId), { updatedAt: serverTimestamp() }, { merge: true });
             } catch (e) {
                console.error("Failed to add post message to chat", e);
             }
          }
        } else {
           setComposerStatus({ type: 'success', msg: "Published successfully!" });
        }
    } catch(e: any) {
       setComposerStatus({ type: 'error', msg: e.message });
    } finally {
       setIsPublishing(false);
    }
  };

  const handleResetComposer = () => {
    setComposerText('');
    setComposerFile(null);
    setComposerPreview(null);
    setComposerBase64(null);
    setComposerThumbnail(null);
    setComposerStatus(null);
    setCrossPostToIg(false);
    setIsComposerOpen(false);
  };

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setSidebarOpen(true);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Sync data on login only
  const hasSyncedRef = useRef(false);
  useEffect(() => {
    if (!user || (!fbConnected && !igConnected) || hasSyncedRef.current) return;
    
    console.log("[INITIAL-SYNC] User logged in and accounts connected. Syncing...");
    hasSyncedRef.current = true;
    handleRefreshData();
  }, [user, fbConnected, igConnected]);

  // Scheduled refresh (every 4 hours)
  useEffect(() => {
    if (!user || (!fbConnected && !igConnected)) return;

    const interval = setInterval(() => {
      console.log("[AUTO-SYNC] Triggering scheduled refresh...");
      handleRefreshData();
    }, 4 * 60 * 60 * 1000); // 4 hours
    
    return () => clearInterval(interval);
  }, [user, fbConnected, igConnected]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      setLoading(false);
      
      if (u) {
        // Ensure user profile exists and load state
        try {
          const userDoc = doc(db, 'users', u.uid);
          await setDoc(userDoc, {
            uid: u.uid,
            email: u.email,
            displayName: u.displayName,
            photoURL: u.photoURL,
            updatedAt: serverTimestamp()
          }, { merge: true });

          // Load profile state
          onSnapshot(userDoc, (snap) => {
            if (snap.exists()) {
              const data = snap.data();
              setFbConnected(!!data.fbConnected);
              setIgConnected(!!data.igConnected);
              setFbData(data.fbData || null);
              setIgData(data.igData || null);
            }
          });

          // Load Saved Intelligence
          const analysesCol = collection(db, `users/${u.uid}/analyses`);
          onSnapshot(analysesCol, (snap) => {
            const analyses: Record<string, string> = {};
            snap.docs.forEach(doc => {
              const d = doc.data();
              if (d.videoId) analyses[d.videoId] = d.analysis;
              if (d.videoUrl) analyses[d.videoUrl] = d.analysis;
            });
            setSavedAnalyses(analyses);
          });

          const synthesesCol = collection(db, `users/${u.uid}/syntheses`);
          onSnapshot(query(synthesesCol, orderBy('createdAt', 'desc')), (snap) => {
            setSavedSyntheses(snap.docs.map(d => ({ id: d.id, ...d.data() })));
          });

        } catch (error) {
          handleFirestoreError(error, OperationType.WRITE, `users/${u.uid}`);
        }
      }
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!user) return;
    const q = query(
      collection(db, 'chats'),
      where('userId', '==', user.uid)
    );
    return onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map(d => ({ id: d.id, ...d.data() } as ChatSession));
      
      // Sort locally to avoid needing a Firestore compound index on new databases
      docs.sort((a, b) => {
        const timeA = a.updatedAt?.toMillis ? a.updatedAt.toMillis() : 0;
        const timeB = b.updatedAt?.toMillis ? b.updatedAt.toMillis() : 0;
        return timeB - timeA;
      });

      setSessions(docs);
      if (docs.length > 0 && !currentSessionId) {
        setCurrentSessionId(docs[0].id);
      }
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'chats');
    });
  }, [user]);

  useEffect(() => {
    if (!currentSessionId) {
      setMessages([]);
      return;
    }
    const q = query(
      collection(db, `chats/${currentSessionId}/messages`),
      orderBy('timestamp', 'asc')
    );
    return onSnapshot(q, (snapshot) => {
      setMessages(snapshot.docs.map(d => ({ id: d.id, ...d.data() } as Message)));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, `chats/${currentSessionId}/messages`);
    });
  }, [currentSessionId]);

  useEffect(() => {
    if (scrollRef.current) {
      const container = scrollRef.current;
      const threshold = 150;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
      
      if (isNearBottom || messages.length <= 1) {
        container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
      }
    }
  }, [messages, logs]);

  useEffect(() => {
    if (scrollRef.current && streamingMessage) {
      const container = scrollRef.current;
      const threshold = 150;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
      
      if (isNearBottom) {
        container.scrollTo({ top: container.scrollHeight, behavior: 'auto' });
      }
    }
  }, [streamingMessage]);

  const runTikTokResearch = async () => {
    if (sending || !user) return;
    setIsChatPlusMenuOpen(false);
    
    setInput('');
    setSending(true);
    setLogs([]);
    
    let sessionId = currentSessionId;
    if (!sessionId) {
      try {
        const sessionDoc = await addDoc(collection(db, 'chats'), {
          title: "TikTok Competitor Research",
          userId: user.uid,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        });
        sessionId = sessionDoc.id;
        setCurrentSessionId(sessionId);
      } catch (error: any) {
        handleFirestoreError(error, OperationType.WRITE, 'chats');
        setSending(false);
        return;
      }
    }

    try {
      const userMessageRef = await addDoc(collection(db, `chats/${sessionId}/messages`), {
        role: 'user',
        content: "Please conduct TikTok Competitor Research for my currently focused account and tell me what is winning for them.",
        timestamp: serverTimestamp(),
      });

      const addLog = (log: LogEntry) => setLogs(prev => [...prev, log]);
      
      let filteredFb = fbConnected ? fbData : null;
      let filteredIg = igConnected ? igData : null;
      if (selectedAccount !== 'auto') {
         if (filteredFb?.accounts) {
             filteredFb = { ...filteredFb, accounts: filteredFb.accounts.filter((a: any) => (a.pageName || a.name) === selectedAccount) };
         }
         if (filteredIg?.accounts) {
             filteredIg = { ...filteredIg, accounts: filteredIg.accounts.filter((a: any) => a.username === selectedAccount) };
         }
      }

      const metaData = {
        fb: filteredFb || null,
        ig: filteredIg || null,
        focusAccount: selectedAccount,
        syntheses: savedSyntheses || []
      };

      addLog({ timestamp: new Date(), level: 'info', message: 'Generating niche keywords and searching TikTok...' });
      
      const { fetchTikTokResearch } = await import('./lib/tiktok');
      const researchResponse = await fetchTikTokResearch(selectedAccount, metaData) || {};
      const researchVideos = researchResponse.videos || [];
      const searchContext = researchResponse.searchContext || `Searched TikTok API for competitor videos near the niche.`;
      
      addLog({ timestamp: new Date(), level: 'info', message: `Found ${researchVideos.length} competitor videos. Extracing patterns...` });
      
      // Deeply analyze only the top 4 videos to prevent hitting exact rate limits (15 RPM)
      const videosToDeepAnalyze = Math.min(4, researchVideos.length);
      for (let i = 0; i < researchVideos.length; i++) {
         const video = researchVideos[i];
         
         if (i >= videosToDeepAnalyze) {
             video.aiAnalysis = "Visual analysis skipped to preserve API quota. Insights based on caption, metrics, and metadata.";
             continue;
         }

         const videoToAnalyze = video.playUrl; 
         if (!videoToAnalyze) {
             video.aiAnalysis = "Visual analysis skipped. No direct MP4 URL found. Analyzing based on caption and metrics.";
             continue; // Skip trying to proxy a web page.
         }
         try {
             addLog({ timestamp: new Date(), level: 'info', message: `Visually analyzing Top Competitor Video ${i+1}/${videosToDeepAnalyze}...` });
             const analysis = await analyzeVideoWithGemini(videoToAnalyze, () => {}); // Silence individual logs
             video.aiAnalysis = analysis;
             if (i < videosToDeepAnalyze - 1) {
                 addLog({ timestamp: new Date(), level: 'info', message: `Waiting 12s to preserve API quota...` });
                 await new Promise(r => setTimeout(r, 12000));
             }
         } catch(e: any) {
             console.error("Video AI analysis failed for", videoToAnalyze, e.message);
             video.aiAnalysis = "Visual analysis failed or platform restricted. Use provided metrics and hook.";
             if (i < videosToDeepAnalyze - 1) await new Promise(r => setTimeout(r, 8000));
         }
      }

      const researchDataStr = JSON.stringify(researchResponse, null, 2);
      const analysisText = "\n\n[TIKTOK RESEARCH CONTEXT]:\n" + researchDataStr;
      
      await setDoc(userMessageRef, { content: "Please conduct TikTok Competitor Research for my currently focused account and tell me what is winning for them.\n\n_" + searchContext + "_" }, { merge: true });

      setStreamingMessage("");
      const messagePayload: any = { role: 'user', content: `Please conduct TikTok Competitor Research for my currently focused account and tell me what is winning for them.\n\n${searchContext}\n\nAnalyze the provided competitor videos (which are provided in JSON format below) and tell me what's winning in terms of metrics, hooks, and pacing. Mention the competitor brands/creators by name. Format similarly to the Dashboard Synthesis (winning patterns and growth killers metrics).` + analysisText };

      const aiResponse = await chatWithGemini(
        [...messages, messagePayload] as any,
        metaData,
        addLog,
        (chunk) => setStreamingMessage(prev => (prev || "") + chunk)
      );

      setStreamingMessage(null);

      await addDoc(collection(db, `chats/${sessionId}/messages`), {
        role: 'assistant',
        content: aiResponse || 'No response',
        researchVideos: researchVideos,
        timestamp: serverTimestamp()
      });

      await setDoc(doc(db, 'chats', sessionId), { updatedAt: serverTimestamp() }, { merge: true });

    } catch (error: any) {
      console.error("Chat error:", error);
      const addLog = (log: LogEntry) => setLogs(prev => [...prev, log]);
      addLog({ timestamp: new Date(), level: 'error', message: error.message || 'Failed to complete research' });
    } finally {
      setSending(false);
    }
  };

  const runProductResearch = async (query?: string) => {
    if (sending || !user) return;
    setIsChatPlusMenuOpen(false);
    
    // We will ask the user what product they want to research
    const productQuery = query || window.prompt("Enter product to research (e.g., 'smart watch'):");
    if (!productQuery || !productQuery.trim()) return;

    setInput('');
    setSending(true);
    setLogs([]);
    
    let sessionId = currentSessionId;
    if (!sessionId) {
      try {
        const sessionDoc = await addDoc(collection(db, 'chats'), {
          title: `Product Research: ${productQuery}`,
          userId: user.uid,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        });
        sessionId = sessionDoc.id;
        setCurrentSessionId(sessionId);
      } catch (error: any) {
        handleFirestoreError(error, OperationType.WRITE, 'chats');
        setSending(false);
        return;
      }
    }

    try {
      const userMessageRef = await addDoc(collection(db, `chats/${sessionId}/messages`), {
        role: 'user',
        content: `Please conduct Product Research for: "${productQuery}" on Alibaba and AliExpress and give me a summary of findings.`,
        timestamp: serverTimestamp(),
      });

      const addLog = (log: LogEntry) => setLogs(prev => [...prev, log]);
      
      addLog({ timestamp: new Date(), level: 'info', message: `Querying Alibaba and AliExpress for "${productQuery}"...` });
      
      const { fetchProductResearch } = await import('./lib/ecommerce');
      const researchResponse = await fetchProductResearch(productQuery) || {};
      
      const alibabaCount = researchResponse.alibaba?.length || 0;
      const aliexpressCount = researchResponse.aliexpress?.length || 0;
      
      if (researchResponse.generatedKeyword && researchResponse.generatedKeyword !== productQuery) {
          addLog({ timestamp: new Date(), level: 'info', message: `Analyzed prompt. Extracted search keyword: "${researchResponse.generatedKeyword}"` });
      }
      
      addLog({ timestamp: new Date(), level: 'info', message: `Found ${alibabaCount} items on Alibaba and ${aliexpressCount} on AliExpress. Analyzing...` });
      
      let filteredFb = fbConnected ? fbData : null;
      let filteredIg = igConnected ? igData : null;
      if (selectedAccount !== 'auto') {
         if (filteredFb?.accounts) {
             filteredFb = { ...filteredFb, accounts: filteredFb.accounts.filter((a: any) => a.name === selectedAccount) };
         }
         if (filteredIg?.accounts) {
             filteredIg = { ...filteredIg, accounts: filteredIg.accounts.filter((a: any) => a.username === selectedAccount) };
         }
      }

      const metaData = {
        fb: filteredFb || null,
        ig: filteredIg || null,
        focusAccount: selectedAccount,
        syntheses: savedSyntheses || []
      };

      const researchDataStr = JSON.stringify({
        ...researchResponse,
        // Make sure to remove large raw structures if any, keep it clean
      }, null, 2);
      const analysisText = `\n\n[ECOMMERCE RESEARCH CONTEXT (Live Alibaba & AliExpress Data)]:\nExtracted Keyword Used for Search: "${researchResponse.generatedKeyword || productQuery}"\n` + researchDataStr;
      
      await setDoc(userMessageRef, { content: productQuery }, { merge: true });

      setStreamingMessage("");
      const messagePayload: any = { role: 'user', content: `Here is my product research request: "${productQuery}".\n\nPlease address my specific request directly using the live competitor product data provided below from Alibaba and AliExpress. Optimize your response to provide exactly what I need based on my prompt. If I asked a specific question, answer it. If I just gave a product name, give me a clear snapshot of prices, trends, and marketing angles.\n\n${analysisText}` };

      if (isChatFileMode) {
         setStreamingMessage("Generating file...");
         addLog({ timestamp: new Date(), level: 'info', message: 'Generating requested file via assistant...' });
         
         const payload = { prompt: messagePayload.content, type: 'md', metaData };
         
         try {
           const fileRes = await safeFetchJson('/api/gemini/generate-file', {
             method: 'POST',
             headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify(payload)
           });
           
           setStreamingMessage(null);
           const aiResponse = `I generated the file **${fileRes.fileName}** as requested:\n\n\`\`\`md\n${fileRes.content}\n\`\`\``;
           
           await addDoc(collection(db, `chats/${sessionId}/messages`), {
             role: 'assistant',
             content: aiResponse,
             timestamp: serverTimestamp()
           });
           setIsChatFileMode(false);
         } catch (e: any) {
           addLog({ timestamp: new Date(), level: 'error', message: `File generation failed: ${e.message}` });
           setStreamingMessage(null);
         }
      } else {
        const aiResponse = await chatWithGemini(
          [...messages, messagePayload] as any,
          metaData,
          addLog,
          (chunk) => setStreamingMessage(prev => (prev || "") + chunk)
        );

        setStreamingMessage(null);

        await addDoc(collection(db, `chats/${sessionId}/messages`), {
          role: 'assistant',
          content: aiResponse || 'No response',
          productsData: researchResponse,
          timestamp: serverTimestamp()
        });
      }

      await setDoc(doc(db, 'chats', sessionId), { updatedAt: serverTimestamp() }, { merge: true });

    } catch (error: any) {
      console.error("Chat error:", error);
      const addLog = (log: LogEntry) => setLogs(prev => [...prev, log]);
      addLog({ timestamp: new Date(), level: 'error', message: error.message || 'Failed to complete product research' });
    } finally {
      setSending(false);
    }
  };

  const handleSendMessage = async (e?: React.FormEvent, overrideText?: string) => {
    e?.preventDefault();
    const userText = overrideText || input;
    if ((!userText.trim() && !chatFileBase64) || sending || !user) return;

    if (isProductResearchMode && !chatFileBase64 && !overrideText) {
       setIsProductResearchMode(false);
       return runProductResearch(userText);
    }

    setInput('');
    setSending(true);
    setLogs([]);

    let sessionId = currentSessionId;
    if (!sessionId) {
      const sessionDoc = await addDoc(collection(db, 'chats'), {
        userId: user.uid,
        title: userText.trim() ? userText.slice(0, 30) + '...' : 'Analysis context',
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      });
      sessionId = sessionDoc.id;
      setCurrentSessionId(sessionId);
    }

    try {
      const userMessageRef = await addDoc(collection(db, `chats/${sessionId}/messages`), {
        role: 'user',
        content: userText,
        timestamp: serverTimestamp(),
        mediaBase64: chatFileBase64,
        mediaType: chatFileMimeType,
        thumbnailUrl: chatFileThumbnail
      });

      const addLog = (log: LogEntry) => setLogs(prev => [...prev, log]);
      
      let filteredFb = fbConnected ? fbData : null;
      let filteredIg = igConnected ? igData : null;
      
      if (selectedAccount !== 'auto') {
         if (filteredFb?.accounts) {
             filteredFb = { ...filteredFb, accounts: filteredFb.accounts.filter((a: any) => (a.pageName || a.name) === selectedAccount) };
         }
         if (filteredIg?.accounts) {
             filteredIg = { ...filteredIg, accounts: filteredIg.accounts.filter((a: any) => a.username === selectedAccount) };
         }
      }

      const metaData = {
        fb: filteredFb || null,
        ig: filteredIg || null,
        focusAccount: selectedAccount,
        syntheses: savedSyntheses || []
      };

      let analysisText = "";
      const urlMatch = userText.match(/(https?:\/\/[^\s]+)/);
      const isVideoRequest = userText.toLowerCase().includes("video") || userText.toLowerCase().includes("reel") || userText.toLowerCase().includes("analyze");

      if (urlMatch) {
         const url = urlMatch[0];
         const isDiscoveryCandidate = url.includes('facebook.com') || url.includes('instagram.com');
         
          if (isVideoRequest && (url.includes('/reels/') || url.includes('/reel/') || url.includes('/videos/') || url.includes('/p/'))) {
            try {
              const analysis = await analyzeVideoWithGemini(url, addLog);
              analysisText = "\n\n[VIDEO ANALYSIS CONTEXT]:\n" + analysis;
              await setDoc(userMessageRef, { content: userText + analysisText }, { merge: true });
            } catch(e: any) {
                 addLog({ timestamp: new Date(), level: 'error', message: `Sync error: ${e.message}` });
            }
         } else if (isDiscoveryCandidate && (userText.toLowerCase().includes("discover") || userText.toLowerCase().includes("find") || userText.toLowerCase().includes("profile"))) {
            addLog({ timestamp: new Date(), level: 'info', message: `Connecting to discovery engine...` });
            setTimeout(() => addLog({ timestamp: new Date(), level: 'info', message: `Parsing public DOM for asset signatures...` }), 1200);
            
            try {
               const data = await safeFetchJson('/api/discover-videos', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ url })
               });
               if (data.links && data.links.length > 0) {
                  analysisText = `\n\n[DISCOVERY CONTEXT]: I found ${data.links.length} public video links on this profile:\n${data.links.slice(0, 5).join('\n')}${data.links.length > 5 ? '\n...and more.' : ''}`;
                  addLog({ timestamp: new Date(), level: 'success', message: `Discovered intelligence: Found ${data.links.length} assets.` });
                  await setDoc(userMessageRef, { content: userText + analysisText }, { merge: true });
               } else {
                  addLog({ timestamp: new Date(), level: 'warning', message: `Discovery scan complete: 0 public assets found.` });
               }
            } catch(e: any) {
               addLog({ timestamp: new Date(), level: 'error', message: `Discovery engine failed: ${e.message}` });
            }
         }
      }

      setStreamingMessage("");

      const messagePayload: any = { role: 'user', content: userText + analysisText };
      if (chatFileBase64 && chatFileMimeType) {
         let data = chatFileBase64;
         if (chatFileBase64.startsWith('data:')) {
           const arr = chatFileBase64.split(',');
           if (arr.length > 1) data = arr[1];
         }
         messagePayload.inlineData = { data, mimeType: chatFileMimeType };
      }
      setChatFileBase64(null);
      setChatFileMimeType(null);
      setChatFileThumbnail(null);

      if (isChatFileMode) {
         setStreamingMessage("Generating file...");
         addLog({ timestamp: new Date(), level: 'info', message: 'Generating requested file via assistant...' });
         
         const payload = { prompt: userText + analysisText, type: 'md', metaData };
         if (userText.toLowerCase().includes('html')) payload.type = 'html';
         if (userText.toLowerCase().includes('json')) payload.type = 'json';
         if (userText.toLowerCase().includes('csv')) payload.type = 'csv';
         
         try {
           const fileRes = await safeFetchJson('/api/gemini/generate-file', {
             method: 'POST',
             headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify(payload)
           });
           
           setStreamingMessage(null);
           const aiResponse = `I generated the file **${fileRes.fileName}** as requested:\n\n\`\`\`${payload.type}\n${fileRes.content}\n\`\`\``;
           
           await addDoc(collection(db, `chats/${sessionId}/messages`), {
             role: 'assistant',
             content: aiResponse,
             timestamp: serverTimestamp()
           });
           setIsChatFileMode(false);
         } catch (e: any) {
           addLog({ timestamp: new Date(), level: 'error', message: `File generation failed: ${e.message}` });
           setStreamingMessage(null);
         }
      } else {
        const aiResponse = await chatWithGemini(
          [...messages, messagePayload] as any,
          metaData,
          addLog,
          (chunk) => setStreamingMessage(prev => (prev || "") + chunk)
        );

        setStreamingMessage(null);

        await addDoc(collection(db, `chats/${sessionId}/messages`), {
          role: 'assistant',
          content: aiResponse || 'No response',
          timestamp: serverTimestamp()
        });
      }

      await setDoc(doc(db, 'chats', sessionId), { updatedAt: serverTimestamp() }, { merge: true });

    } catch (error: any) {
      if (error && error.message && (error.message.includes("quota") || error.message.includes("429"))) {
        console.warn("Chat Error (Quota):", error.message);
        setLogs(prev => [...prev, { timestamp: new Date(), level: 'error', message: "Error: You have exceeded your Gemini API quota. Please check your billing details." }]);
      } else if (error && error.code === 'permission-denied') {
        console.error("Chat Error:", error);
        handleFirestoreError(error, OperationType.WRITE, `chats/${sessionId}`);
      } else {
        console.error("Chat Error:", error);
        setLogs(prev => [...prev, { timestamp: new Date(), level: 'error', message: `Error: ${error instanceof Error ? error.message : String(error)}` }]);
      }
    } finally {
      setSending(false);
    }
  };

  const handleRenameChat = async (sessionId: string) => {
    if (!editingTitle.trim() || !user) {
      setEditingSessionId(null);
      return;
    }
    try {
      await updateDoc(doc(db, 'chats', sessionId), { title: editingTitle.trim(), updatedAt: serverTimestamp() });
      setEditingSessionId(null);
    } catch (error) {
      handleFirestoreError(error, OperationType.UPDATE, `chats/${sessionId}`);
    }
  };

  const handleDeleteChat = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!user) return;
    try {
      await deleteDoc(doc(db, 'chats', sessionId));
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
      }
    } catch (error) {
      handleFirestoreError(error, OperationType.DELETE, `chats/${sessionId}`);
    }
  };

  const handleRefreshData = async () => {
    if (!user) return;
    setIsRefreshing(true);
    setAuthStatus("Refreshing data...");
    
    try {
      const refreshPlatform = async (platform: 'facebook' | 'instagram', token: string) => {
        const data = await safeFetchJson('/api/auth/meta/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ platform, accessToken: token })
        });
        const { data: fetchedData } = data;
        await setDoc(doc(db, 'users', user.uid), {
          [`${platform === 'facebook' ? 'fb' : 'ig'}Data`]: fetchedData,
          updatedAt: serverTimestamp()
        }, { merge: true });
      };

      const promises = [];
      if (fbConnected && fbData?.accessToken) promises.push(refreshPlatform('facebook', fbData.accessToken));
      if (igConnected && igData?.accessToken) promises.push(refreshPlatform('instagram', igData.accessToken));
      
      await Promise.all(promises);
      setAuthStatus("Data refreshed successfully!");
      
      setTimeout(() => setAuthStatus(null), 3000);
    } catch (e: any) {
      setAuthStatus(`Refresh error: ${e.message}`);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleConnect = async (platform: 'facebook' | 'instagram') => {
    if (!user) return;
    setAuthStatus(`Requesting ${platform} auth...`);
    try {
      const originUrl = encodeURIComponent(window.location.origin);
      const data = await safeFetchJson(`/api/auth/${platform}/url?redirectBase=${originUrl}`);
      const { url } = data;
      setAuthStatus(`Waiting for ${platform} authorization...`);
      const authWindow = window.open(url, `${platform}_auth`, 'width=600,height=700');
      
      const handleMessage = async (event: MessageEvent) => {
        if (event.data?.type === 'AUTH_ERROR' && event.data?.platform === platform) {
          setAuthStatus(`Authentication Error: ${event.data.error}`);
          window.removeEventListener('message', handleMessage);
        } else if (event.data?.type === 'AUTH_SUCCESS' && event.data?.platform === platform) {
          setAuthStatus('Exchanging token... This might take a few seconds.');
          try {
            // Un-encode the base url for the redirectUri
            const originBase = window.location.origin;
            const data = await safeFetchJson('/api/auth/meta/exchange', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                 code: event.data.code,
                 platform,
                 redirectUri: `${originBase}/auth/${platform}/callback`
              })
            });
            
            const { data: fetchedData } = data;
            setAuthStatus(`Saving ${platform} data...`);
            await setDoc(doc(db, 'users', user.uid), {
              [`${platform === 'facebook' ? 'fb' : 'ig'}Connected`]: true,
              [`${platform === 'facebook' ? 'fb' : 'ig'}Data`]: fetchedData,
              updatedAt: serverTimestamp()
            }, { merge: true });
            setAuthStatus(`Successfully connected ${platform}!`);
            setTimeout(() => setAuthStatus(null), 3000);
          } catch (err: any) {
            setAuthStatus(`Error: ${err.message}`);
            handleFirestoreError(err, OperationType.WRITE, `users/${user.uid}`);
          }
          window.removeEventListener('message', handleMessage);
        }
      };
      window.addEventListener('message', handleMessage);
    } catch (e: any) {
      console.error(e);
      setAuthStatus(`Error: ${e.message}`);
    }
  };

  const handleStartAudit = async (videoPosts: any[], accountScope: string, isAuto = false) => {
    if (auditState.isAuditing || videoPosts.length === 0 || !user) return;
    
    setAuditState(prev => ({
      ...prev,
      isAuditing: true,
      progress: { current: 0, total: videoPosts.length },
      currentTask: isAuto ? "Auto-Sync: Processing new assets..." : "Initializing Intelligence Pipeline..."
    }));

    const analyses: string[] = [];

    try {
      for (let i = 0; i < videoPosts.length; i++) {
        const post = videoPosts[i];
        const url = post.source || post.media_url || post.videoUrl;
        const id = post.id;
        
        // CHECK IF ALREADY ANALYZED (By ID or URL)
        const cachedAnalysis = (id && savedAnalyses[id]) || (url && savedAnalyses[url]);
        
        if (cachedAnalysis) {
          analyses.push(`[VIDEO ${i+1} METRICS: ${post._likes} likes, ${post._comments} comments, ${post._shares || 0} shares, ${post._views || 0} views, ${post._reach || 0} reach, ${post._engaged || 0} engaged, ${post._clicks || 0} clicks]\n${cachedAnalysis}`);
          setAuditState(prev => ({
            ...prev,
            progress: { current: i + 1, total: videoPosts.length },
            currentTask: `Retrieved analyzed context for ${post.platform} Asset ${i+1}`
          }));
          continue;
        }

        if (!url) continue;

        // Space out requests to avoid hitting rate limits
        if (i > 0) {
           setAuditState(prev => ({
              ...prev,
              currentTask: `Sleeping 12s to preserve API quota...`
           }));
           await new Promise(r => setTimeout(r, 12000));
        }

        setAuditState(prev => ({
          ...prev,
          progress: { current: i + 1, total: videoPosts.length },
          currentTask: `Analyzing Video ${i+1}: "${post.caption?.slice(0, 30) || 'Untitled'}..."`
        }));

        try {
          const analysis = await analyzeVideoWithGemini(url, (log) => setAuditState(prev => ({ ...prev, currentTask: log.message })));
          
          if (analysis) {
            analyses.push(`[VIDEO ${i+1} METRICS: ${post._likes} likes, ${post._comments} comments, ${post._shares || 0} shares, ${post._views || 0} views, ${post._reach || 0} reach, ${post._engaged || 0} engaged, ${post._clicks || 0} clicks]\n${analysis}`);
            
            // PERSIST ANALYSIS
            await addDoc(collection(db, `users/${user.uid}/analyses`), {
               videoId: id || null,
               videoUrl: url,
               analysis: analysis,
               processedAt: serverTimestamp(),
               metrics: { likes: post._likes, comments: post._comments, shares: post._shares || 0, reach: post._reach || 0, engaged: post._engaged || 0, views: post._views || 0, clicks: post._clicks || 0 },
               platform: post.platform
            });
          }
        } catch (err: any) {
          let msg = err.message || String(err);
          // Parse JSON if possible
          try {
             const parsed = JSON.parse(msg);
             if (parsed.error && parsed.error.message) msg = parsed.error.message;
          } catch(e) {}
          
          if (msg.includes("quota") || msg.includes("RESOURCE_EXHAUSTED") || msg.includes("429")) {
             console.warn("Audit item failed due to quota:", msg);
             if (analyses.length > 0) {
                 console.warn("API quota exceeded, breaking loop to synthesize early.");
                 setAuditState(prev => ({
                    ...prev,
                    currentTask: `Quota exceeded. Synthesizing ${analyses.length} early...`
                 }));
                 await new Promise(r => setTimeout(r, 2000));
                 break; // Proceed to synthesize what we have
             } else {
                 throw new Error("You have exceeded your Gemini API quota. Please check your billing details.");
             }
          } else if (msg.includes("API key not valid") || msg.includes("API_KEY_INVALID")) {
             console.warn("Audit item failed due to invalid API key:", msg);
             throw new Error("API key not valid. Please pass a valid API key.");
          } else {
             console.warn("Audit item failed:", msg);
          }
        }
      }

      setAuditState(prev => ({ ...prev, currentTask: "Synthesizing Global Performance Patterns..." }));

      // Prompt Gemini to synthesize the patterns
      const synthesisPrompt = `
        I have analyzed ${analyses.length} videos from a social media account. 
        Each analysis includes its engagement metrics (likes, comments, reach, views, clicks, shares, total engaged) and a frame-by-frame breakdown of its content.
        
        Please identify the "Winning DNA" and "Engagement Killers" for this account.
        Focus on:
        1. Visual Hooks: What visual starts lead to higher reach or likes?
        2. Pacing: Does fast pacing correlate with higher engagement / comments?
        3. CTAs: Which calls-to-action generate the most shares or comments?
        4. Verbal Patterns: Are there specific topics or styles of talking that win overall reach?
        
        FORMAT YOUR RESPONSE AS TWO PARTS:
        ### WINNING PATTERNS
        (Bullet points of what works)
        
        ### GROWTH KILLERS
        (Bullet points of what to avoid)

        DATA FOR SYNTHESIS:
        ${analyses.join('\n\n---\n\n')}
      `;

      const aiResponse = await chatWithGemini(
        [{ role: 'user', content: synthesisPrompt }] as any,
        {},
        (log) => console.log("[AUDIT LOG]", log.message)
      );

      const splitStr = '### GROWTH KILLERS';
      const [winnersPart, losersPart] = (aiResponse || "").split(splitStr);
      const winners = winnersPart.replace('### WINNING PATTERNS', '').trim();
      const losers = losersPart ? losersPart.trim() : "No clear losing patterns identified yet.";
      
      const report = { winners, losers };

      // PERSIST SYNTHESIS
      await addDoc(collection(db, `users/${user.uid}/syntheses`), {
        accountName: accountScope === 'auto' ? 'Global' : accountScope,
        platform: accountScope === 'auto' ? 'Global' : videoPosts[0].platform,
        winners: winners,
        losers: losers,
        videoCount: videoPosts.length,
        createdAt: serverTimestamp()
      });

      setAuditState(prev => ({
        ...prev,
        isAuditing: false,
        currentTask: null,
        report
      }));

    } catch (err: any) {
      if (err && err.message && (err.message.includes("quota") || err.message.includes("429"))) {
         console.warn("Master Audit Failed (Quota):", err.message);
      } else {
         console.error("Master Audit Failed:", err);
      }
      setAuditState(prev => ({ ...prev, isAuditing: false, currentTask: `Error: ${err?.message || "Failed during audit."}` }));
    }
  };

  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [authStatus, setAuthStatus] = useState<string | null>(null);

  const handleLogin = async () => {
    if (isLoggingIn) return;
    setIsLoggingIn(true);
    try {
      await loginWithGoogle();
    } catch (error: any) {
      if (error?.code === 'auth/cancelled-popup-request') {
        console.log('Login popup request was superseded or cancelled.');
      } else if (error?.code === 'auth/popup-closed-by-user') {
        console.log('Login popup was closed by user.');
      } else {
        console.error('Login error:', error);
      }
    } finally {
      setIsLoggingIn(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#FAFAFA]">
        <motion.div 
          animate={{ rotate: 360 }} 
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-8 h-8 border-t-2 border-r-2 border-slate-900 rounded-full"
        />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-[#FAFAFA] text-slate-900 p-6">
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md p-10 bg-white rounded-3xl border border-slate-100 shadow-sm text-center"
        >
          <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Cpu className="w-8 h-8 text-slate-900" />
          </div>
          <h1 className="text-2xl font-semibold mb-3 tracking-tight">MetaLink</h1>
          <p className="text-slate-500 mb-8 px-4 text-sm font-medium">
            Universal social analytics powered by Gemini.
          </p>
          <button
            onClick={handleLogin}
            disabled={isLoggingIn}
            className="w-full py-4 px-6 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 text-white font-medium rounded-xl transition-colors flex items-center justify-center gap-3"
          >
            {isLoggingIn ? (
              <motion.div 
                animate={{ rotate: 360 }} 
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-5 h-5 border-t-2 border-r-2 border-white rounded-full"
              />
            ) : (
              <User className="w-5 h-5" />
            )}
            {isLoggingIn ? "Signing in..." : "Continue with Google"}
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#FAFAFA] text-slate-900 overflow-hidden font-sans selection:bg-slate-200 selection:text-slate-900 relative">
      {/* Sidebar Backdrop */}
      <AnimatePresence>
        {isSidebarOpen && (
           <motion.div 
             initial={{ opacity: 0 }}
             animate={{ opacity: 1 }}
             exit={{ opacity: 0 }}
             onClick={() => setSidebarOpen(false)}
             className="fixed inset-0 bg-slate-900/20 z-40 md:hidden"
           />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.aside
            initial={{ x: -260 }}
            animate={{ x: 0 }}
            exit={{ x: -260 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-[260px] border-r border-slate-200 bg-[#FAFAFA] flex flex-col z-50 absolute md:relative h-full"
          >
            <div className="p-6 flex items-center justify-between">
              <div className="flex items-center gap-3 font-semibold text-[17px] tracking-tight text-slate-900">
                <div className="w-8 h-8 flex items-center justify-center border border-slate-200 rounded-xl bg-white shadow-sm">
                  <Cpu className="w-4 h-4 text-slate-900" />
                </div>
                MetaLink
              </div>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setCurrentSessionId(null)} 
                className="p-2 hover:bg-slate-200/50 rounded-xl text-slate-400 hover:text-slate-600 transition-colors"
              >
                <Plus className="w-5 h-5" />
              </motion.button>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1">
              
              <div className="mb-6 space-y-1">
                <div className="px-3 pb-2 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Views</div>
                <button
                  onClick={() => {
                      setCurrentView('chat');
                      if (window.innerWidth <= 768) setSidebarOpen(false);
                  }}
                  className={cn("w-full text-left px-3 py-2 text-sm font-medium rounded-xl transition-colors flex items-center gap-3 relative", currentView === 'chat' ? "text-slate-900" : "text-slate-600 hover:bg-slate-200/50")}
                >
                  {currentView === 'chat' && (
                      <motion.div layoutId="nav-pill" className="absolute inset-0 bg-white border border-slate-200 shadow-sm rounded-xl z-0" transition={{ type: "spring", stiffness: 300, damping: 25 }} />
                  )}
                  <MessageSquare className="w-4 h-4 shrink-0 z-10 relative" />
                  <span className="z-10 relative">Chat</span>
                </button>
                <button
                  onClick={() => {
                      setCurrentView('dashboard');
                      if (window.innerWidth <= 768) setSidebarOpen(false);
                  }}
                  className={cn("w-full text-left px-3 py-2 text-sm font-medium rounded-xl transition-colors flex items-center gap-3 relative", currentView === 'dashboard' ? "text-slate-900" : "text-slate-600 hover:bg-slate-200/50")}
                >
                  {currentView === 'dashboard' && (
                      <motion.div layoutId="nav-pill" className="absolute inset-0 bg-white border border-slate-200 shadow-sm rounded-xl z-0" transition={{ type: "spring", stiffness: 300, damping: 25 }} />
                  )}
                  <LayoutDashboard className="w-4 h-4 shrink-0 z-10 relative" />
                  <span className="z-10 relative">Dashboard</span>
                </button>
                <button
                  onClick={() => {
                      setCurrentView('studio');
                      if (window.innerWidth <= 768) setSidebarOpen(false);
                  }}
                  className={cn("w-full text-left px-3 py-2 text-sm font-medium rounded-xl transition-colors flex items-center gap-3 relative", currentView === 'studio' ? "text-slate-900" : "text-slate-600 hover:bg-slate-200/50")}
                  id="nav_studio_btn"
                >
                  {currentView === 'studio' && (
                      <motion.div layoutId="nav-pill" className="absolute inset-0 bg-white border border-slate-200 shadow-sm rounded-xl z-0" transition={{ type: "spring", stiffness: 300, damping: 25 }} />
                  )}
                  <Video className="w-4 h-4 shrink-0 z-10 relative" />
                  <span className="z-10 relative">Studio</span>
                </button>
              </div>

              <div className="px-3 mb-4 text-[10px] font-semibold uppercase tracking-wider text-slate-400">Workspace Library</div>
              {sessions.length === 0 ? (
                <div className="px-4 py-6 text-center text-xs font-medium text-slate-400">
                  No active chats
                </div>
              ) : (
                sessions.map(s => (
                  <div key={s.id} className={cn(
                    "w-full text-left px-3 py-2.5 rounded-xl transition-colors text-sm flex items-center justify-between font-medium group cursor-pointer relative",
                    currentSessionId === s.id 
                      ? "text-slate-900" 
                      : "text-slate-500 hover:bg-slate-200/50 hover:text-slate-900"
                  )} onClick={() => {
                      if (editingSessionId !== s.id) {
                          setCurrentSessionId(s.id);
                          if (window.innerWidth <= 768) setSidebarOpen(false);
                      }
                  }}>
                    {currentSessionId === s.id && (
                        <motion.div layoutId="session-pill" className="absolute inset-0 bg-white border border-slate-200 shadow-sm rounded-xl z-0" transition={{ type: "spring", stiffness: 300, damping: 25 }} />
                    )}
                    {editingSessionId === s.id ? (
                        <div className="flex items-center w-full gap-2 z-10 relative" onClick={e => e.stopPropagation()}>
                            <MessageSquare className="w-4 h-4 shrink-0 opacity-50" />
                            <input
                              autoFocus
                              type="text"
                              className="flex-1 min-w-0 bg-transparent border-none text-slate-900 text-sm focus:ring-0 p-0"
                              value={editingTitle}
                              onChange={(e) => setEditingTitle(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRenameChat(s.id);
                                if (e.key === 'Escape') setEditingSessionId(null);
                              }}
                            />
                            <button onClick={() => handleRenameChat(s.id)} className="p-1 hover:bg-slate-200 rounded text-green-600"><Check className="w-3.5 h-3.5" /></button>
                            <button onClick={() => setEditingSessionId(null)} className="p-1 hover:bg-slate-200 rounded text-slate-400"><X className="w-3.5 h-3.5" /></button>
                        </div>
                    ) : (
                        <>
                            <div className="flex items-center gap-3 truncate min-w-0 flex-1 pr-2 relative z-10">
                              <MessageSquare className="w-4 h-4 shrink-0 opacity-50" />
                              <span className="truncate">{s.title}</span>
                            </div>
                            <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity relative z-10">
                               <button 
                                 onClick={(e) => { e.stopPropagation(); setEditingTitle(s.title); setEditingSessionId(s.id); }}
                                 className="p-1.5 text-slate-400 hover:text-slate-900 hover:bg-slate-100 rounded-md"
                               >
                                  <Edit2 className="w-3.5 h-3.5" />
                               </button>
                               <button 
                                 onClick={(e) => handleDeleteChat(e, s.id)}
                                 className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-md ml-1"
                               >
                                  <Trash2 className="w-3.5 h-3.5" />
                               </button>
                            </div>
                        </>
                    )}
                  </div>
                ))
              )}
            </div>

            <div className="px-4 pb-4 space-y-4">
              {authStatus && (
                <div className="p-3 text-sm rounded-xl bg-orange-50 border border-orange-200 text-orange-700 font-medium">
                  <div>{authStatus}</div>
                  {authStatus.includes("configured as a desktop app") && (
                     <div className="mt-2 text-xs opacity-90 p-2 bg-white/50 rounded-lg">
                        <strong>Fix:</strong> Go to Meta Developer Dashboard &rarr; Facebook Login &rarr; Settings &rarr; Set "Native or desktop app?" to <strong>No</strong>. Save changes and try again.
                     </div>
                  )}
                </div>
              )}
              <div className="grid grid-cols-2 gap-2">
                <button 
                  onClick={() => {
                    if (fbConnected) {
                       if (confirm("Disconnect Facebook?")) {
                          setDoc(doc(db, 'users', user.uid), { fbConnected: false, fbData: null }, { merge: true });
                       }
                    } else {
                       handleConnect('facebook');
                    }
                  }}
                  className={cn(
                    "py-3 px-3 rounded-xl border transition-colors flex flex-col items-center gap-2",
                    fbConnected ? "bg-white border-blue-200 text-slate-900 shadow-sm" : "bg-transparent border-slate-200 text-slate-500 hover:bg-white hover:shadow-sm"
                  )}
                >
                  <Facebook className={cn("w-4 h-4", fbConnected ? "text-blue-600" : "text-slate-400")} />
                  <span className="text-[10px] uppercase font-semibold tracking-wider">Facebook</span>
                </button>
                <button 
                  onClick={() => {
                    if (igConnected) {
                       if (confirm("Disconnect Instagram?")) {
                          setDoc(doc(db, 'users', user.uid), { igConnected: false, igData: null }, { merge: true });
                       }
                    } else {
                       handleConnect('instagram');
                    }
                  }}
                  className={cn(
                    "py-3 px-3 rounded-xl border transition-colors flex flex-col items-center gap-2",
                    igConnected ? "bg-white border-rose-200 text-slate-900 shadow-sm" : "bg-transparent border-slate-200 text-slate-500 hover:bg-white hover:shadow-sm"
                  )}
                >
                  <Instagram className={cn("w-4 h-4", igConnected ? "text-rose-600" : "text-slate-400")} />
                  <span className="text-[10px] uppercase font-semibold tracking-wider">Instagram</span>
                </button>
              </div>
              <div className="flex items-center gap-3 p-3 bg-white rounded-xl border border-slate-200 shadow-sm">
                <img src={user.photoURL || ''} className="w-9 h-9 rounded-lg bg-slate-100 object-cover border border-slate-100" />
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] font-semibold truncate text-slate-900 leading-tight">{user.displayName}</p>
                  <p className="text-[11px] text-slate-500 font-medium truncate lowercase">{user.email}</p>
                </div>
                <button onClick={logout} className="p-2 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-slate-600 transition-colors">
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Chat / Dashboard */}
      <main className="flex-1 flex flex-col relative min-w-0 bg-white md:shadow-[-8px_0_24px_-8px_rgba(0,0,0,0.05)] z-10 md:m-2 md:ml-0 md:rounded-2xl border md:border-slate-200 overflow-hidden">
        {currentView === 'dashboard' ? (
           <DashboardView 
              fbData={fbData} 
              igData={igData} 
              selectedAccount={selectedAccount} 
              setSelectedAccount={setSelectedAccount}
              onRefresh={handleRefreshData}
              isRefreshing={isRefreshing}
              onAnalyzeVideo={(url: string) => {
                 setCurrentView('chat');
                 handleSendMessage(undefined, `Analyze this video: ${url}`);
              }} 
              onStartAudit={handleStartAudit}
              auditState={auditState}
              savedSyntheses={savedSyntheses}
              setAuditState={setAuditState}
           />
        ) : currentView === 'studio' ? (
           <div className="w-full h-full flex flex-col relative overflow-hidden bg-white">
              <HyperframesStudioFrame />
              <AIEditorPanel />
           </div>
        ) : (
          <>
            <header className="h-16 flex items-center justify-between px-6 border-b border-slate-100 bg-white/80 backdrop-blur z-10">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setSidebarOpen(!isSidebarOpen)} 
              className="w-8 h-8 flex items-center justify-center hover:bg-slate-100 rounded-lg text-slate-500 transition-colors"
            >
              <History className="w-4 h-4" />
            </button>
            <div className="flex flex-col">
              <h2 className="font-semibold text-[15px] tracking-tight text-slate-900 leading-none mb-1">
                {currentSessionId ? sessions.find(s => s.id === currentSessionId)?.title : "Universal Intelligence"}
              </h2>
              <div className="flex items-center gap-1.5">
                 <div className={cn("w-1.5 h-1.5 rounded-full", (fbConnected || igConnected) ? "bg-green-500" : "bg-slate-300")} />
                 <span className="text-[10px] font-medium text-slate-500">
                   {(fbConnected || igConnected) ? "Live Sync Active" : "Local Mode"}
                 </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
             <div className={cn("w-8 h-8 rounded-full flex items-center justify-center border transition-colors", fbConnected ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-transparent border-slate-200 text-slate-300")}>
               <Facebook className="w-3.5 h-3.5" />
             </div>
             <div className={cn("w-8 h-8 rounded-full flex items-center justify-center border transition-colors", igConnected ? "bg-rose-50 border-rose-200 text-rose-600" : "bg-transparent border-slate-200 text-slate-300")}>
               <Instagram className="w-3.5 h-3.5" />
             </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 md:px-8 pt-8 pb-32 scroll-smooth" ref={scrollRef}>
          {messages.length === 0 && !sending && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col items-center justify-center h-[70vh] text-center max-w-sm mx-auto"
            >
               <MorphingLogo isThinking={sending} size={240} />
               <h3 className="text-2xl font-semibold tracking-tight text-slate-900 mb-2 mt-8">Decision Intelligence</h3>
               <p className="text-slate-500 text-sm mb-8">
                 Analyze audience behavior across platforms.
               </p>
               <div className="grid grid-cols-2 gap-3 w-full mb-3">
                 <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 text-left hover:border-slate-200 transition-colors cursor-pointer" onClick={() => handleSendMessage(undefined, "Can you analyze my recent posts regarding engagement vs reach?")}>
                    <p className="text-[13px] font-semibold text-slate-900 mb-0.5">Post Analysis</p>
                    <p className="text-[11px] text-slate-500 font-medium">Identify resonant sets</p>
                 </div>
                 <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 text-left hover:border-slate-200 transition-colors cursor-pointer" onClick={() => handleSendMessage(undefined, "Based on my reach history, what can I forecast for next month?")}>
                    <p className="text-[13px] font-semibold text-slate-900 mb-0.5">Reach Forecasting</p>
                    <p className="text-[11px] text-slate-500 font-medium">Predict future growth</p>
                 </div>
               </div>
               <div className="w-full flex">
                 <div className="flex-1 p-4 bg-indigo-50 rounded-2xl border border-indigo-100 text-center hover:bg-indigo-100 transition-colors cursor-pointer group flex items-center justify-center gap-3" onClick={() => setIsComposerOpen(true)}>
                    <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shadow-md group-hover:scale-105 transition-transform">
                      <Plus className="w-4 h-4 text-white" />
                    </div>
                    <div className="text-left">
                      <p className="text-[13px] font-semibold text-indigo-900 mb-0.5">Create New Post</p>
                      <p className="text-[11px] text-indigo-700/70 font-medium">Open AI Composer</p>
                    </div>
                 </div>
               </div>
            </motion.div>
          )}
          
          <AnimatePresence mode="popLayout">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </AnimatePresence>

          {/* Live Request Indicator & Streaming Message */}
          <AnimatePresence>
             { (sending || streamingMessage !== null) && (
               <motion.div 
                 initial={{ opacity: 0, y: 10 }} 
                 animate={{ opacity: 1, y: 0 }} 
                 exit={{ opacity: 0, scale: 0.95 }}
                 className="flex w-full mb-8 justify-start origin-bottom-left"
               >
                 <div className="max-w-[80%] rounded-3xl px-6 py-5 leading-relaxed bg-white text-slate-800 border border-slate-100 shadow-sm">
                   <div className="flex items-center gap-3 mb-4 text-[12px] font-bold tracking-wider uppercase text-slate-500">
                     <MorphingLogo isThinking={true} size={64} />
                     <span className="pt-1">
                     {streamingMessage !== null ? "Generating Answer" : "Processing"}
                     {streamingMessage === null && (
                        <span className="inline-flex items-center gap-0.5 ml-1">
                           <motion.span animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.4, repeat: Infinity, delay: 0 }} className="w-1 h-1 bg-slate-400 rounded-full" />
                           <motion.span animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.4, repeat: Infinity, delay: 0.2 }} className="w-1 h-1 bg-slate-400 rounded-full" />
                           <motion.span animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.4, repeat: Infinity, delay: 0.4 }} className="w-1 h-1 bg-slate-400 rounded-full" />
                        </span>
                     )}
                     </span>
                   </div>
                   <div className="markdown-body text-[15px] font-normal leading-[1.6] break-words overflow-hidden prose prose-slate max-w-[100%] prose-p:mb-3">
                     {streamingMessage !== null ? (
                       <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingMessage}</ReactMarkdown>
                     ) : (
                       <div className="text-slate-400 flex flex-col gap-1.5 h-6 overflow-hidden relative">
                          <AnimatePresence mode="popLayout">
                             {logs.slice(-1).map((log, i) => (
                                <motion.div 
                                  key={log.timestamp.getTime()}
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -10 }}
                                  className="text-[13px] truncate"
                                >
                                   {log.message}
                                </motion.div>
                             ))}
                          </AnimatePresence>
                       </div>
                     )}
                   </div>
                 </div>
               </motion.div>
             )}
          </AnimatePresence>

        </div>

        {/* Input Bar - Restrained */}
        <div className="absolute bottom-4 md:bottom-6 left-0 right-0 px-4 md:px-8 flex flex-col items-center justify-end pointer-events-none z-20">
          
          <div className="w-full max-w-xl flex items-center gap-2 mb-2 pointer-events-auto">
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setIsComposerOpen(!isComposerOpen)} 
              className={cn("flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 border border-indigo-100 text-indigo-600 rounded-xl text-xs font-bold shadow-sm hover:bg-indigo-100 transition-all cursor-pointer text-shadow-sm", isComposerOpen && "bg-indigo-600 text-white hover:bg-indigo-700 font-bold border-indigo-700")}
            >
               <Plus className={cn("w-3.5 h-3.5 transition-transform", isComposerOpen && "rotate-45")} /> Create Post
            </motion.button>
          </div>

          <div className="w-full max-w-xl relative pointer-events-auto bg-white rounded-[32px] border border-slate-200 shadow-sm flex flex-col transition-shadow focus-within:border-slate-300 focus-within:shadow-md">
            <AnimatePresence>
              {isComposerOpen && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="w-full border-b border-slate-100 flex flex-col bg-slate-50/50 max-h-[60vh] overflow-y-auto custom-scrollbar rounded-t-[32px]"
                >
                  <div className="p-3 md:p-5 pb-3">
                    <div className="flex items-center justify-between mb-4">
                       <h3 className="text-slate-900 font-bold flex items-center gap-2 text-sm">
                         <Plus className="w-4 h-4 text-indigo-500" /> Create Post
                       </h3>
                    </div>

                    {composerStatus && (
                       <div className={cn("p-3 rounded-xl mb-4 text-sm font-medium border flex items-center justify-between gap-4", composerStatus.type === 'success' ? "bg-green-50 text-green-700 border-green-200" : "bg-rose-50 text-rose-700 border-rose-200")}>
                         <span>{composerStatus.msg}</span>
                         {composerStatus.type === 'success' && (
                           <button 
                             onClick={handleResetComposer}
                             type="button"
                             className="px-3 py-1 bg-green-600 text-white rounded-lg text-[10px] font-black uppercase hover:bg-green-700 transition-colors shadow-sm cursor-pointer"
                           >
                             Finish
                           </button>
                         )}
                       </div>
                    )}

                    <div className="flex flex-col md:flex-row gap-2 mb-3 mt-1">
                       <select 
                         value={postType} 
                         onChange={(e) => setPostType(e.target.value as any)}
                         className="bg-white border border-slate-200 rounded-xl px-3 py-2 text-xs font-bold w-full md:w-auto text-slate-700 outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400/20"
                       >
                          <option value="text">Text / Status</option>
                          <option value="image">Image Post</option>
                          <option value="video">Feed Video</option>
                          <option value="reel">Reel / Short</option>
                          <option value="story">Story</option>
                       </select>

                       <select 
                         value={postFormat} 
                         onChange={(e) => setPostFormat(e.target.value as any)}
                         className="bg-white border border-slate-200 rounded-xl px-3 py-2 text-xs font-bold w-full md:w-auto text-slate-700 outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400/20"
                       >
                          <option value="auto">Original Format</option>
                          <option value="1:1">1:1 (Square)</option>
                          <option value="4:5">4:5 (Portrait)</option>
                          <option value="9:16">9:16 (Vertical)</option>
                          <option value="16:9">16:9 (Landscape)</option>
                       </select>
                    </div>

                    <div className={cn(
                      "grid gap-4 md:gap-6",
                      "grid-cols-1 md:grid-cols-2"
                    )}>
                       {/* Left Side: Upload & Media */}
                       <div className="flex flex-col gap-2">
                          <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Media Preview</label>
                          <div 
                             className="w-full max-w-[180px] mx-auto border-2 border-dashed border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/50 bg-slate-50 transition-all rounded-[28px] flex flex-col items-center justify-center cursor-pointer overflow-hidden relative shadow-inner"
                             style={{
                               minHeight: '120px',
                               maxHeight: '200px',
                               aspectRatio: postFormat === 'auto' ? undefined : postFormat.replace(':', '/')
                             }}
                          >
                             {composerPreview ? (
                                <>
                                  {composerPreview.startsWith('blob:') || composerPreview.startsWith('data:video/') ? (
                                    <video 
                                      src={composerPreview} 
                                      poster={composerThumbnail || undefined} 
                                      controls 
                                      className="w-full h-full object-cover absolute z-20 inset-0 bg-black pointer-events-auto" 
                                    />
                                  ) : (
                                    <img src={composerPreview} alt="Preview" className="w-full h-full object-cover absolute inset-0" />
                                  )}
                                  <div className="absolute top-4 right-4 z-40">
                                    <button type="button" onClick={() => { setComposerPreview(null); setComposerBase64(null); setComposerFile(null); }} className="bg-black/60 hover:bg-black text-white p-2 rounded-full backdrop-blur-md transition-all shadow-xl hover:scale-110 active:scale-95">
                                      <X className="w-5 h-5"/>
                                    </button>
                                  </div>
                                </>
                             ) : (
                                <label className="w-full h-full absolute inset-0 flex flex-col items-center justify-center p-8 cursor-pointer z-10 hover:bg-slate-100/50 transition-colors">
                                  <div className="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm mb-4">
                                    <Video className="w-8 h-8 text-indigo-500" />
                                  </div>
                                  <span className="text-sm font-bold text-slate-900 mb-1">Upload Creative</span>
                                  <span className="text-[11px] font-medium text-slate-400 text-center uppercase tracking-wider">Drag and drop or click to browse</span>
                                  <input type="file" accept="image/*,video/*" className="hidden" onChange={handleComposerFile} />
                                </label>
                             )}
                          </div>
                          {(postType === 'reel' || postFormat === '9:16') && (
                            <p className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest text-center mt-2 flex items-center justify-center gap-2">
                              <CheckCircle2 className="w-3 h-3" /> Immersive vertical format active
                            </p>
                          )}
                       </div>

                       {/* Right Side: Caption */}
                       <div className="flex flex-col gap-2 min-h-[120px]">
                          <div className="flex items-center justify-between">
                             <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Caption</label>
                             {isGeneratingCaption ? (
                               <div className="flex items-center gap-2">
                                 <div className="w-24 h-1.5 bg-indigo-100 rounded-full overflow-hidden">
                                    <motion.div className="h-full bg-indigo-500 rounded-full" initial={{ width: "0%" }} animate={{ width: "100%" }} transition={{ duration: 15, ease: "linear" }} />
                                 </div>
                                 <span className="text-[10px] font-bold text-indigo-500 animate-pulse uppercase">Analyzing...</span>
                               </div>
                             ) : (
                               <button onClick={handleGenerateCaption} className="text-[9px] font-black uppercase text-indigo-500 hover:text-indigo-600 tracking-wider flex items-center gap-1 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded-md transition-colors cursor-pointer relative z-10 pointer-events-auto">
                                 <Cpu className="w-3 h-3" /> Gen AI
                               </button>
                             )}
                          </div>
                          <textarea 
                            value={composerText}
                            onChange={(e) => setComposerText(e.target.value)}
                            placeholder="Write a caption or generate one..."
                            className="flex-1 bg-white border border-slate-200 rounded-2xl p-3 text-xs resize-none focus:outline-none focus:border-indigo-400 transition-colors custom-scrollbar z-10 relative"
                          />
                       </div>
                    </div>

                    <div className="mt-4 flex items-center justify-between">
                       <div className="flex flex-col gap-2">
                         <div className="flex items-center gap-2">
                            <div className="text-[9px] font-black uppercase tracking-widest text-slate-400">Target</div>
                            <AccountDropdown fbData={fbData} igData={igData} selectedAccount={selectedAccount} setSelectedAccount={setSelectedAccount} direction="up" />
                         </div>
                         {fbData?.accounts?.find((a: any) => a.pageName === selectedAccount || a.name === selectedAccount)?.linkedInstagramAccountId && (
                            <label className={cn("flex items-center gap-2 mt-1 pl-[48px]", (!composerBase64 && !composerFile) ? "opacity-50 cursor-not-allowed" : "cursor-pointer")} title={(!composerBase64 && !composerFile) ? "Instagram requires media to cross-post." : "Post to linked Instagram account"}>
                               <input type="checkbox" disabled={!composerBase64 && !composerFile} checked={crossPostToIg} onChange={(e) => setCrossPostToIg(e.target.checked)} className="rounded text-indigo-500 focus:ring-indigo-500/20 bg-slate-100 border-slate-300 w-3 h-3 disabled:cursor-not-allowed" />
                               <span className="text-[11px] font-medium text-slate-500 uppercase tracking-widest flex items-center gap-1"><Instagram className="w-3 h-3"/> Cross-post to linked IG</span>
                            </label>
                         )}
                       </div>
                       <button 
                         onClick={handlePublish}
                         disabled={isPublishing || (!composerText && !composerFile)}
                         className="px-6 py-2 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white font-bold rounded-xl transition-all shadow-md shadow-indigo-500/20 active:scale-[0.98] flex items-center gap-2 text-sm z-10 relative cursor-pointer"
                       >
                         {isPublishing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                         Publish
                       </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <form 
              onSubmit={handleSendMessage}
              className="flex items-center p-1.5 w-full bg-white relative z-20 rounded-[32px]"
            >
            <input type="file" accept="image/*,video/*" className="hidden" ref={chatFileInputRef} onChange={handleChatFileUpload} />
            {isChatFileMode && (
               <div className="flex items-center gap-1.5 ml-2 px-2.5 py-1 bg-indigo-50 border border-indigo-100 rounded-full shrink-0">
                  <FileCode className="w-3.5 h-3.5 text-indigo-500" />
                  <span className="text-[10px] uppercase font-bold text-indigo-600 tracking-wider">File Mode</span>
                  <button type="button" onClick={() => setIsChatFileMode(false)} className="hover:text-indigo-800 ml-1">
                    <X className="w-3 h-3" />
                  </button>
               </div>
            )}
            {isProductResearchMode && (
               <div className="flex items-center gap-1.5 ml-2 px-2.5 py-1 bg-emerald-50 border border-emerald-100 rounded-full shrink-0">
                  <ShoppingBag className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-[10px] uppercase font-bold text-emerald-600 tracking-wider">Product Mode</span>
                  <button type="button" onClick={() => setIsProductResearchMode(false)} className="hover:text-emerald-800 ml-1">
                    <X className="w-3 h-3" />
                  </button>
               </div>
            )}
            {chatFileBase64 ? (
              <div className="relative w-10 h-10 ml-1 rounded-xl overflow-hidden shrink-0 group border border-slate-200">
                {chatFileMimeType?.startsWith('video/') ? (
                  chatFileThumbnail ? (
                    <img src={chatFileThumbnail} alt="Video thumbnail" className="w-full h-full object-cover" />
                  ) : (
                    <video src={chatFileBase64} className="w-full h-full object-cover" />
                  )
                ) : (
                  <img src={chatFileBase64} alt="Upload preview" className="w-full h-full object-cover" />
                )}
                <button type="button" onClick={() => { setChatFileBase64(null); setChatFileMimeType(null); setChatFileThumbnail(null); }} className="absolute inset-0 bg-black/50 hidden group-hover:flex items-center justify-center text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="relative ml-1">
                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  type="button"
                  onClick={() => setIsChatPlusMenuOpen(!isChatPlusMenuOpen)}
                  className={cn("w-10 h-10 rounded-2xl flex items-center justify-center transition-colors shrink-0 bg-slate-50 text-slate-400 hover:bg-slate-100 hover:text-slate-600 focus:outline-none", isChatPlusMenuOpen && "bg-slate-200 text-slate-900")}
                >
                  <Plus className={cn("w-5 h-5 transition-transform", isChatPlusMenuOpen && "rotate-45")} />
                </motion.button>
                <AnimatePresence>
                  {isChatPlusMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-30" onClick={() => setIsChatPlusMenuOpen(false)} />
                      <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        className="absolute bottom-full left-0 mb-2 w-48 bg-white border border-slate-200 shadow-xl rounded-2xl overflow-hidden z-40 py-1"
                      >
                         <button
                           type="button"
                           onClick={() => { chatFileInputRef.current?.click(); setIsChatPlusMenuOpen(false); }}
                           className="w-full text-left px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-3 transition-colors"
                         >
                            <Video className="w-4 h-4 text-pink-500" />
                            Upload Media
                         </button>
                         <button
                           type="button"
                           onClick={runTikTokResearch}
                           className="w-full text-left px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-3 transition-colors border-t border-slate-100"
                         >
                            <Search className="w-4 h-4 text-blue-500" />
                            Research Mode
                         </button>
                         <button
                           type="button"
                           onClick={() => { 
                             setIsProductResearchMode(!isProductResearchMode); 
                             if (!isProductResearchMode) setIsChatFileMode(false);
                             setIsChatPlusMenuOpen(false); 
                           }}
                           className="w-full text-left px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-3 transition-colors border-t border-slate-100"
                         >
                            <ShoppingBag className={`w-4 h-4 ${isProductResearchMode ? 'text-indigo-500' : 'text-emerald-500'}`} />
                            {isProductResearchMode ? 'Product Mode: ON' : 'Product Mode: OFF'}
                         </button>
                         <button
                           type="button"
                           onClick={() => { 
                             setIsChatFileMode(!isChatFileMode); 
                             if (!isChatFileMode) setIsProductResearchMode(false);
                             setIsChatPlusMenuOpen(false); 
                           }}
                           className="w-full text-left px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 flex items-center gap-3 transition-colors border-t border-slate-100"
                         >
                            <FileCode className={`w-4 h-4 ${isChatFileMode ? 'text-indigo-500' : 'text-amber-500'}`} />
                            {isChatFileMode ? 'File Mode: ON' : 'File Mode: OFF'}
                         </button>
                      </motion.div>
                    </>
                  )}
                </AnimatePresence>
              </div>
            )}
            <div className="bg-slate-50 p-1 rounded-2xl ml-1 border border-slate-100 shrink-0 select-none hidden md:block">
              <AccountDropdown fbData={fbData} igData={igData} selectedAccount={selectedAccount} setSelectedAccount={setSelectedAccount} direction="up" />
            </div>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isChatFileMode ? "Describe the file to generate..." : isProductResearchMode ? "Enter product name to research..." : (fbConnected || igConnected) ? "Ask about performance..." : "Type your query..."}
              disabled={sending}
              className="flex-1 bg-transparent py-3 pl-3 md:pl-4 pr-4 focus:outline-none disabled:opacity-50 text-slate-900 font-medium placeholder:text-slate-400 text-[16px] md:text-[15px]"
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              disabled={(!input.trim() && !chatFileBase64) || sending}
              className="w-10 h-10 flex-shrink-0 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-100 disabled:text-slate-400 text-white rounded-2xl flex items-center justify-center transition-colors shrink-0"
            >
              <Send className="w-4 h-4 ml-0.5" />
            </motion.button>
          </form>
          </div>
        </div>
        </>
        )}
      </main>
    </div>
  );
}
