export async function safeFetchJson(url: string, options?: RequestInit) {
  const res = await fetch(url, options);
  const text = await res.text();
  
  if (!res.ok) {
    let errorMessage = `Error ${res.status}`;
    let isRateLimit = text.includes("Rate exceeded") || res.status === 429;
    
    try {
      const json = JSON.parse(text);
      errorMessage = json.error || json.message || errorMessage;
    } catch (e) {
      errorMessage = text || errorMessage;
    }

    if (isRateLimit) {
      throw new Error("API Rate limit exceeded. Please wait a moment and try again.");
    }
    
    throw new Error(errorMessage);
  }

  try {
    return JSON.parse(text);
  } catch (e) {
    if (text.includes("<title>Starting Server...</title>")) {
       throw new Error("Server is still starting or restarting. Please try again in a few seconds.");
    }
    console.error("Failed to parse JSON response:", text);
    throw new Error("Received invalid response from server.");
  }
}
