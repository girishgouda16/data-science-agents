import { useEffect, useRef, useState } from "react";
import { useSessions } from "../state/SessionsContext";
import { Composer } from "./Composer";
import { MessageBubble, WorkingBubble } from "./MessageBubble";

// [agent that handles it, prompt]
const EXAMPLES: [string, string][] = [
  ["Classification", "Predict churn in my customer file — explain what drives it"],
  ["Forecasting", "Forecast next month's call volume from the attached time series"],
  ["Anomaly detection", "Find anomalous calling patterns (possible Wangiri fraud)"],
  ["Clustering", "Segment these customers into behavioural groups"],
  ["Drift", "Has this month's data drifted from what the champion model was trained on?"],
];

function elapsed(since: number | null, now: number): string {
  if (!since) return "";
  const s = Math.max(0, Math.round(now / 1000 - since));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

export function ChatView({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  const { active, displayedHistory, isSending, isWorking, error, canRetry, retry, send, cancel, setDraft } =
    useSessions();
  const logRef = useRef<HTMLDivElement>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [displayedHistory.length, isSending, isWorking]);

  useEffect(() => {
    if (!isWorking) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [isWorking]);

  const waiting = active?.status === "input-required" && !isSending;
  const busy = isSending || isWorking;
  const statusText = isSending
    ? "Sending…"
    : isWorking
      ? `Agents working · ${elapsed(active?.started_at ?? null, now)}`
      : waiting
        ? "Waiting on your answer"
        : active?.status === "failed"
          ? "Last turn failed"
          : active?.status === "canceled"
            ? "Stopped"
            : "Ready";

  const empty = displayedHistory.length === 0 && !busy;

  return (
    <div className={`main ${empty ? "is-empty" : ""}`}>
      <div className="topbar">
        <button type="button" className="ghost sidebar-toggle" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          ☰
        </button>
        <div className="session-title-bar">
          {active?.title ?? "New chat"}
          {active?.project && <span className="chip">project: {active.project}</span>}
          {active?.autopilot && <span className="chip">autopilot</span>}
        </div>
        <div className="status-bar" aria-live="polite">
          <span className={`dot ${busy ? "working" : active?.status ?? ""}`} />
          {statusText}
        </div>
        {isWorking && (
          <button type="button" className="danger small" onClick={() => void cancel()}>
            Stop
          </button>
        )}
      </div>

      <div className="log-wrap" ref={logRef}>
        <div className="log">
          {displayedHistory.length === 0 && (
            <div className="empty-state">
              <h2>What should we model today?</h2>
              <p>Attach a CSV (drag it here or use the paperclip) and describe the goal. The agents ask before any
                data-changing step, and nothing is promoted without your say-so.</p>
              <div className="examples">
                {EXAMPLES.map(([agent, text]) => (
                  <button key={text} type="button" className="example" onClick={() => setDraft(text)}>
                    <span className="example-agent">{agent}</span>
                    {text}
                  </button>
                ))}
              </div>
            </div>
          )}
          {displayedHistory.map((turn, i) => (
            <MessageBubble key={i} turn={turn} />
          ))}
          {busy && <WorkingBubble sending={isSending} />}
        </div>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <span>⚠ {error}</span>
          {canRetry && (
            <button type="button" onClick={retry}>
              Retry
            </button>
          )}
        </div>
      )}

      <Composer
        onSend={send}
        disabled={busy}
        isNewChat={!active}
        placeholder={
          waiting
            ? "The agent is waiting for your answer…"
            : isWorking
              ? "The agents are working — you can Stop them, or wait for the reply"
              : "Describe what you want, e.g. “predict churn and explain the drivers”"
        }
      />
    </div>
  );
}
