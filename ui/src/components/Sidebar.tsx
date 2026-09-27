import { useState } from "react";
import { deleteEach, useSelection } from "../lib/selection";
import type { SessionStatus } from "../lib/types";
import { useAuth } from "../state/AuthContext";
import { useSessions } from "../state/SessionsContext";
import { SelectionBar } from "./SelectionBar";

export type View = "chat" | "runs" | "models";

const STATUS_HINT: Partial<Record<SessionStatus, string>> = {
  working: "agents are working",
  "input-required": "waiting on your reply",
  failed: "the last turn failed",
  canceled: "stopped",
};

export function Sidebar({
  open,
  onClose,
  view,
  onView,
}: {
  open: boolean;
  onClose: () => void;
  view: View;
  onView: (v: View) => void;
}) {
  const { me, signOut } = useAuth();
  const { sessions, activeId, loadingList, newChat, selectSession, removeSession } = useSessions();

  const go = (v: View) => {
    onView(v);
    onClose();
  };

  const sel = useSelection(sessions.map((s) => s.session_id));
  const [busy, setBusy] = useState(false);
  const deleteSelected = async () => {
    if (!window.confirm(`Delete ${sel.selected.length} chat(s)? This cannot be undone.`)) return;
    setBusy(true);
    await deleteEach(sel.selected, removeSession); // removeSession reports its own errors
    setBusy(false);
    sel.clear();
  };

  return (
    <nav className={`sidebar ${open ? "open" : ""}`} aria-label="Navigation">
      <div className="sidebar-header">
        <div className="brand">
          <span className="brand-mark">◆</span> Holdout
        </div>
        <button
          type="button"
          className="new-chat-btn primary"
          onClick={() => {
            newChat();
            go("chat");
          }}
        >
          ＋ New chat
        </button>
        <div className="nav-tabs" role="tablist">
          {(["chat", "runs", "models"] as View[]).map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={view === v}
              className={`nav-tab ${view === v ? "active" : ""}`}
              onClick={() => go(v)}
            >
              {v === "chat" ? "Chats" : v === "runs" ? "Runs" : "Models"}
            </button>
          ))}
        </div>
      </div>

      <div className={`session-list ${sel.selected.length ? "selecting" : ""}`}>
        {sel.selected.length > 0 && (
          <SelectionBar
            what="chats"
            total={sessions.length}
            count={sel.selected.length}
            allSelected={sel.allSelected}
            onToggleAll={sel.toggleAll}
            onDelete={() => void deleteSelected()}
            busy={busy}
          />
        )}
        {loadingList && sessions.length === 0 && <div className="session-list-empty">Loading…</div>}
        {!loadingList && sessions.length === 0 && <div className="session-list-empty">No chats yet.</div>}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            role="button"
            tabIndex={0}
            className={`session-item ${s.session_id === activeId && view === "chat" ? "active" : ""}`}
            onClick={() => {
              selectSession(s.session_id);
              go("chat");
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                selectSession(s.session_id);
                go("chat");
              }
            }}
          >
            <input
              type="checkbox"
              className="check"
              checked={sel.isSelected(s.session_id)}
              onChange={() => sel.toggle(s.session_id)}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
              aria-label={`Select chat ${s.title}`}
            />
            <span className={`dot ${s.status}`} title={STATUS_HINT[s.status] ?? "done"} />
            <span className="session-title">
              {s.title}
              {s.project && <span className="session-project">{s.project}</span>}
            </span>
            <button
              type="button"
              className="session-delete"
              title="Delete chat"
              aria-label={`Delete chat ${s.title}`}
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm(`Delete “${s.title}”? This cannot be undone.`)) void removeSession(s.session_id);
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="subject" title={me?.email ?? me?.user}>
          {me?.name || me?.user}
          {me?.admin && <span className="badge">admin</span>}
        </div>
        <button type="button" className="ghost" onClick={signOut}>
          Sign out
        </button>
      </div>
    </nav>
  );
}
