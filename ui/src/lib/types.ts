// Mirrors gateway/app.py's responses. The agent is one continuous chat thread
// with human-in-the-loop pauses, not a typed pipeline: text turns + a status.

export interface ChatTurn {
  role: "user" | "agent";
  text: string;
  attachment?: string; // the uploaded file's own name
}

// A2A task states the gateway passes through. "working" means the turn is
// running in the background (the UI polls); "input-required" means an agent
// asked a question and is waiting for the user's next message.
export type SessionStatus = "working" | "input-required" | "completed" | "failed" | "canceled" | "rejected";

export interface SessionSummary {
  session_id: string;
  status: SessionStatus;
  title: string;
  project: string | null;
  autopilot: boolean; // agents decide instead of asking
  reply: string;
  history: ChatTurn[];
  started_at: number | null; // epoch seconds, while working
  updated_at: string;
}

export interface Me {
  user: string;
  name?: string;
  email?: string;
  roles: string[];
  admin: boolean;
}

export interface RunSummary {
  run_id: string;
  phase: string;
  suggested_next: string;
  target: string | null;
  model: string | null;
  domain: string | null;
  owner: string | null;
  has_report: boolean;
  last_touched: string;
  registry?: { name?: string; version?: number } | null;
}

export interface ModelSummary {
  name: string;
  latest_version: number | null;
  aliases: Record<string, number>;
  updated: number | null;
}

export interface ModelVersion {
  version: number;
  aliases: string[];
  model: string | null;
  target: string | null;
  readiness: string | null;
  metrics: Record<string, number>;
  version_tags: Record<string, string>;
  created: number | null;
}

export interface ModelVersions {
  name?: string;
  versions?: ModelVersion[];
  error?: string;
}
