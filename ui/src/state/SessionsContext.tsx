import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import { fileLabel } from "../lib/files";
import type { ChatTurn, SessionSummary } from "../lib/types";
import { useAuth } from "./AuthContext";

const POLL_MS = 2000;

interface PendingSend {
  message: string;
  filePath?: string;
}

interface SessionsState {
  sessions: SessionSummary[];
  activeId: string | null;
  active: SessionSummary | null;
  /** history + the optimistic user turn not yet confirmed by the server. */
  displayedHistory: ChatTurn[];
  loadingList: boolean;
  isSending: boolean;
  /** the agents are working on the active chat's last message (polled). */
  isWorking: boolean;
  error: string | null;
  canRetry: boolean;
  /** project for the NEXT new chat (a chat's project is fixed once started). */
  project: string;
  setProject: (p: string) => void;
  /** sent with every message, so switching it applies from the next one. */
  autopilot: boolean;
  setAutopilot: (on: boolean) => void;
  draft: string;
  setDraft: (d: string) => void;
  newChat: (draft?: string) => void;
  selectSession: (id: string) => void;
  send: (message: string, filePath?: string) => Promise<void>;
  cancel: () => Promise<void>;
  retry: () => void;
  removeSession: (id: string) => Promise<void>;
}

const SessionsContext = createContext<SessionsState | null>(null);

const describe = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function SessionsProvider({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [optimistic, setOptimistic] = useState<ChatTurn[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFailed, setLastFailed] = useState<PendingSend | null>(null);
  const [project, setProject] = useState(() => localStorage.getItem("project") ?? "");
  const [autopilot, setAutopilot] = useState(() => localStorage.getItem("autopilot") === "1");
  const [draft, setDraft] = useState("");

  const upsert = useCallback((s: SessionSummary) => {
    setSessions((prev) => [s, ...prev.filter((x) => x.session_id !== s.session_id)]);
  }, []);

  useEffect(() => {
    if (status !== "signed-in") return;
    api
      .listSessions()
      .then(setSessions)
      .catch((e) => setError(describe(e)))
      .finally(() => setLoadingList(false));
  }, [status]);

  useEffect(() => {
    localStorage.setItem("project", project);
  }, [project]);

  useEffect(() => {
    localStorage.setItem("autopilot", autopilot ? "1" : "0");
  }, [autopilot]);

  const active = sessions.find((s) => s.session_id === activeId) ?? null;
  const isWorking = active?.status === "working";
  const displayedHistory = [...(active?.history ?? []), ...optimistic];

  // Poll every chat that is working — the active one, and any other still
  // running in the background, so its sidebar dot updates when it finishes.
  const workingIds = sessions.filter((s) => s.status === "working").map((s) => s.session_id).join(",");
  useEffect(() => {
    if (!workingIds) return;
    const timer = setInterval(() => {
      for (const id of workingIds.split(",")) {
        api.getSession(id).then(upsert).catch(() => {});
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [workingIds, upsert]);

  const newChat = useCallback((withDraft = "") => {
    setActiveId(null);
    setOptimistic([]);
    setError(null);
    setDraft(withDraft);
  }, []);

  const selectSession = useCallback((id: string) => {
    setActiveId(id);
    setOptimistic([]);
    setError(null);
  }, []);

  const send = useCallback(
    async (message: string, filePath?: string) => {
      setError(null);
      setLastFailed(null);
      setOptimistic([{ role: "user", text: message, attachment: filePath && fileLabel(filePath) }]);
      setIsSending(true);
      try {
        const result = activeId
          ? await api.sendMessage(activeId, message, autopilot)
          : await api.startSession(message, filePath, project.trim(), autopilot);
        upsert(result);
        setActiveId(result.session_id);
      } catch (e) {
        setError(describe(e));
        setLastFailed({ message, filePath });
      } finally {
        setOptimistic([]);
        setIsSending(false);
      }
    },
    [activeId, project, autopilot, upsert],
  );

  const cancel = useCallback(async () => {
    if (!activeId) return;
    try {
      upsert(await api.cancel(activeId));
    } catch (e) {
      setError(describe(e));
    }
  }, [activeId, upsert]);

  const retry = useCallback(() => {
    if (lastFailed) void send(lastFailed.message, lastFailed.filePath);
  }, [lastFailed, send]);

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id);
        setSessions((prev) => prev.filter((s) => s.session_id !== id));
        if (activeId === id) newChat();
      } catch (e) {
        setError(describe(e));
      }
    },
    [activeId, newChat],
  );

  const value: SessionsState = {
    sessions,
    activeId,
    active,
    displayedHistory,
    loadingList,
    isSending,
    isWorking,
    error,
    canRetry: lastFailed !== null,
    project,
    setProject,
    autopilot,
    setAutopilot,
    draft,
    setDraft,
    newChat,
    selectSession,
    send,
    cancel,
    retry,
    removeSession,
  };
  return <SessionsContext.Provider value={value}>{children}</SessionsContext.Provider>;
}

export function useSessions(): SessionsState {
  const ctx = useContext(SessionsContext);
  if (!ctx) throw new Error("useSessions must be used within SessionsProvider");
  return ctx;
}
