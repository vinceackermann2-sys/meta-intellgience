export interface HyperframesProject {
  id: string;
  title?: string;
  dir?: string;
}

function parseProjectIdFromHash(hash: string): string | null {
  const match = hash.match(/^#project\/([^?]+)/);
  if (!match?.[1]) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return match[1];
  }
}

export async function getActiveHyperframesProjectId(): Promise<string> {
  const hashProjectId = parseProjectIdFromHash(window.location.hash);
  if (hashProjectId) return hashProjectId;

  const response = await fetch('/api/hyper-edit/project');
  if (!response.ok) throw new Error('Unable to resolve Hyperframes project');
  const data = await response.json();
  return data.projectId || 'project';
}

export function buildHyperframesStudioUrl(projectId: string, refreshKey = 0): string {
  const search = refreshKey ? `?r=${refreshKey}` : '';
  return `/hyperframes-studio/${search}#project/${encodeURIComponent(projectId)}`;
}
