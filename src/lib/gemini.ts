import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { safeFetchJson } from "./api";

let _aiInstance: GoogleGenAI | null = null;

function getAi() {
  if (_aiInstance) return _aiInstance;
  const apiKey = process.env.GEMINI_API_KEY;
  _aiInstance = new GoogleGenAI({ apiKey });
  return _aiInstance;
}

export interface LogEntry {
  timestamp: Date;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

export async function generateCaptionWithGemini(
  context: string,
  mediaBase64: string | null,
  mediaType: string | null,
  patterns: any | null,
  onLog: (log: LogEntry) => void
) {
  onLog({ timestamp: new Date(), level: 'info', message: 'Generating caption with AI...' });
  
  try {
    let textPrompt = `Write a highly engaging social media caption for a Facebook post. Context: ${context || 'Something exciting'}. Make it catchy, use emojis and hashtags.`;
    if (patterns && (patterns.winners || patterns.losers)) {
      textPrompt += `\nConsider these successful patterns: ${JSON.stringify(patterns.winners)}. Avoid these: ${JSON.stringify(patterns.losers)}.`;
    }

    let payload: any = { prompt: textPrompt };
    let endpoint = '/api/gemini/chat';

    if (mediaBase64 && mediaType) {
       endpoint = '/api/gemini/vision';
       let data = mediaBase64;
       if (mediaBase64.startsWith('data:')) {
          const arr = mediaBase64.split(',');
          if (arr.length > 1) data = arr[1];
       }
       payload = { 
         data, 
         mimeType: mediaType, 
         prompt: textPrompt 
       };
    } else {
       payload = {
          contents: [{ role: 'user', parts: [{ text: textPrompt }] }]
       };
    }

    const data = await safeFetchJson(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    onLog({ timestamp: new Date(), level: 'success', message: 'Caption generated successfully.' });
    return data.text || "Failed to generate caption.";
  } catch (error: any) {
    onLog({ timestamp: new Date(), level: 'error', message: `Generation error: ${error.message}` });
    throw error;
  }
}

export async function analyzeVideoWithGemini(videoUrl: string, onLog: (log: LogEntry) => void): Promise<string> {
  onLog({ timestamp: new Date(), level: 'info', message: 'Downloading video for analysis...' });
  
  try {
    const { base64, mimeType } = await safeFetchJson('/api/proxy-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoUrl })
    });
    
    onLog({ timestamp: new Date(), level: 'info', message: 'Analyzing downloaded video with Gemini...' });
    
    const data = await safeFetchJson('/api/gemini/vision', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: base64,
        mimeType,
        prompt: `Analyze this video frame by frame. Everything visual and verbal has to be analyzed pixel by pixel. Make sure to identify every detail, visual element, text hook, spoken word, and pacing choice.`
      })
    });

    onLog({ timestamp: new Date(), level: 'success', message: 'Visual intelligence analysis complete.' });
    return data.text || "Analysis generated but returned empty response.";
  } catch (error: any) {
    onLog({ timestamp: new Date(), level: 'error', message: `Vision Error: ${error.message}` });
    throw error;
  }
}

export async function chatWithGemini(
  messages: { role: string; content: string, inlineData?: { data: string, mimeType: string } }[],
  metaData: any,
  onLog: (log: LogEntry) => void,
  onChunk?: (chunk: string) => void
) {
  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
  onLog({ timestamp: new Date(), level: 'info', message: 'Initializing Meta Intelligence engine...' });
  await sleep(600);
  
  try {
    onLog({ timestamp: new Date(), level: 'info', message: 'Checking Meta integration status...' });
    await sleep(400);
    if (!metaData) {
      onLog({ timestamp: new Date(), level: 'warning', message: 'Meta account not connected. Results will be general.' });
    } else {
      onLog({ timestamp: new Date(), level: 'success', message: 'Meta data successfully retrieved.' });
    }
    await sleep(300);

    onLog({ timestamp: new Date(), level: 'info', message: 'Processing your request with Gemini...' });
    
    const isEcommerceQuery = messages[messages.length - 1]?.content?.includes('[ECOMMERCE RESEARCH CONTEXT');

    const systemInstruction = isEcommerceQuery ? `
      You are an elite E-commerce and Product Sourcing Analyst.
      The user is asking about a product or analyzing RapidAPI data from Alibaba/AliExpress.
      
      FOCUS DIRECTIVE:
      - Answer the user's explicit question directly.
      - If they provided ecommerce research context, it contains product data (prices, names, MOQs, ratings).
      - Summarize pricing, identify trends, suggest marketing angles, and give a profitability breakdown.
      - EXTREMELY IMPORTANT: DO NOT talk about "creatives", "Instagram", "Facebook", "visual hooks", or "social media metrics" unless explicitly asked to generate marketing content for this product. Stay completely focused on the physical product, pricing, sourcing, margins, and market viability.
      - Treat the user as a dropshipper or brand owner looking to source a product.
      - Ensure your response is highly formatted, professional, and directly matches what they asked.
      
      OUTPUT FORMAT:
      - Use headers, bullet points, and markdown tables.
      - Be concise, direct, and actionable. Avoid generic fluff.
    ` : `
      You are Meta Intelligence AI, a specialist in Facebook and Instagram data analysis.
      You help users understand their social media performance, audience engagement, and profile data.
      
      The user's connected data:
      - Facebook: ${JSON.stringify(metaData?.fb || "Not connected")}
      - Instagram: ${JSON.stringify(metaData?.ig || "Not connected")}
      - Dashboard Syntheses: ${JSON.stringify(metaData?.syntheses || [])}
      
      FOCUS DIRECTIVE:
      The user explicitly selected to focus their query on the account: ${metaData.focusAccount === 'auto' ? 'All Accounts (Global)' : String(metaData.focusAccount).toUpperCase()}.
      If a specific account is selected, ONLY prioritize insights, data, and answers regarding that account. Ignore metrics from other accounts unless comparing.

      When the user asks about their data:
      1. Reference specific platforms (Facebook or Instagram) clearly.
      2. If a platform is disconnected, guide them to connect it in the sidebar.
      3. If a [VIDEO ANALYSIS CONTEXT] is provided, it contains a frame-by-frame breakdown from a vision model. Present this clearly, interpret choices, and give advice.
      4. If a [DISCOVERY CONTEXT] is provided, it contains public video links on a profile. Offer to analyze them.
      5. If a [TIKTOK RESEARCH CONTEXT] is provided, it contains RapidAPI TikTok API data about competitor videos. Summarize what is winning for them based on hooks, pacing, and visual style.
      6. If Dashboard Syntheses are provided, use them to discuss overall winning patterns and growth killers based on past video analyses.
      7. DO NOT ask questions at the end of your response. Just provide the insights and analysis.
      8. OUTPUT FORMAT:
         - Professional, insightful, readable text.
         - HIGHLY VISUAL: Use lots of emojis, icons, and formatting!
         - SPREADSHEETS & GRAPHS: Use Markdown tables ('|Col 1|Col 2|') extensively, treat them like mini-spreadsheets!
         - If explaining flows, you MUST use Mermaid charts in a code block with correct newlines:
           \`\`\`mermaid
           graph TD
           A[Step 1] --> B[Step 2]
           \`\`\`
           (IMPORTANT: DO NOT use double quotes " inside Mermaid node labels or it will break the parser)
         - CHART GENERATION: Output JSON block inside [CHART] and [/CHART] tags to render interactive graphs.
            Example for Bar Chart:
            [CHART]
            {"type": "bar", "data": [{"name": "Likes", "value": 150}, {"name": "Comments", "value": 45}], "xAxis": "name", "yAxis": "value"}
            [/CHART]
            Example for Pie Chart:
            [CHART]
            {"type": "pie", "data": [{"name": "IG", "value": 500}, {"name": "FB", "value": 300}], "dataKey": "value"}
            [/CHART]
            Other valid types: "line", "area". Always provide clean JSON.
    `;

    const contents = messages.map(m => {
      const parts: any[] = [{ text: m.content || "" }];
      if (m.inlineData) {
        parts.unshift({ inlineData: m.inlineData });
      }
      return { role: m.role === 'user' ? 'user' : 'model', parts };
    });

    // NOTE: Streaming is not used here to ensure backend key safety 
    // and simplicity, but could be added via server-sent events if needed.
    const data = await safeFetchJson('/api/gemini/chat', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ contents, systemInstruction })
    });

    const fullText = data.text || "";
    if (onChunk) onChunk(fullText); // Fake stream if needed by UI

    onLog({ timestamp: new Date(), level: 'success', message: 'Response generated.' });
    return fullText;
  } catch (error: any) {
    onLog({ timestamp: new Date(), level: 'error', message: `Error: ${error.message}` });
    throw error;
  }
}
