import { accessToken, apiBase } from "./auth";
import type { Me, ModelSummary, ModelVersions, RunSummary, SessionSummary } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// Set by AuthContext: a 401 anywhere means the session ended — go sign in again.
let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

async function send<T>(path: string, init: RequestInit = {}, json = true): Promise<T> {
  const headers: Record<string, string> = { Authorization: `Bearer ${await accessToken()}` };
  if (json) headers["Content-Type"] = "application/json";
  let res: Response;
  try {
    res = await fetch(apiBase() + path, { ...init, headers: { ...headers, ...(init.headers as object) } });
  } catch {
    throw new ApiError(0, "Can't reach the gateway — check your connection.");
  }
  if (res.status === 401) onUnauthorized();
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, typeof body.detail === "string" ? body.detail : `${res.status} ${res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  me: () => send<Me>("/api/v1/me"),

  listSessions: () => send<{ sessions: SessionSummary[] }>("/api/v1/sessions").then((r) => r.sessions),

  getSession: (id: string) => send<SessionSummary>(`/api/v1/sessions/${id}`),

  startSession: (message: string, filePath?: string, project?: string, autopilot = false) =>
    send<SessionSummary>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ message, file_path: filePath, project: project || undefined, autopilot }),
    }),

  sendMessage: (id: string, message: string, autopilot: boolean) =>
    send<SessionSummary>(`/api/v1/sessions/${id}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, autopilot }),
    }),

  cancel: (id: string) => send<SessionSummary>(`/api/v1/sessions/${id}/cancel`, { method: "POST" }),

  deleteSession: (id: string) => send<void>(`/api/v1/sessions/${id}`, { method: "DELETE" }),

  // No Content-Type: the browser sets the multipart boundary itself.
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return send<{ file_path: string; name: string; size_bytes: number }>(
      "/api/v1/upload", { method: "POST", body: form }, false);
  },

  runs: () => send<{ runs: RunSummary[] }>("/api/v1/runs").then((r) => r.runs ?? []),

  deleteRun: (id: string) => send<void>(`/api/v1/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),

  deleteModel: (name: string) => send<void>(`/api/v1/models/${encodeURIComponent(name)}`, { method: "DELETE" }),

  models: () => send<{ models?: ModelSummary[]; error?: string }>("/api/v1/models"),

  modelVersions: (name: string) => send<ModelVersions>(`/api/v1/models/${encodeURIComponent(name)}`),
};
