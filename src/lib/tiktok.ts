import { safeFetchJson } from "./api";

export async function fetchTikTokResearch(accountName: string, metaData: any) {
  const res = await safeFetchJson('/api/tiktok/research', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ accountName, metaData })
  });
  return res;
}
