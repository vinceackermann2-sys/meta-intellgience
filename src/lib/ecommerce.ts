import { safeFetchJson } from './api';

export async function fetchProductResearch(query: string) {
   try {
      return await safeFetchJson('/api/product/research', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ query })
      });
   } catch (error) {
       console.error("Product research err:", error);
       return { alibaba: [], aliexpress: [], error: String(error) };
   }
}
