import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
dotenv.config();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function checkModel(name: string) {
    try {
        await ai.models.generateContent({
            model: name,
            contents: ["hi"]
        });
        console.log(`Success: ${name}`);
    } catch(e: any) {
        console.log(`Error ${name}: ${e.message}`);
    }
}

async function run() {
    await checkModel("gemini-1.5-flash");
    await checkModel("gemini-2.5-flash");
    await checkModel("gemini-2.5-pro");
    await checkModel("gemini-3.0-flash-preview");
    await checkModel("gemini-3.1-flash-preview");
    await checkModel("gemini-3.1-pro-preview");
}
run();
