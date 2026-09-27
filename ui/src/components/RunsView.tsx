import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { deleteEach, useSelection } from "../lib/selection";
import type { RunSummary } from "../lib/types";
import { useSessions } from "../state/SessionsContext";
import { SelectionBar } from "./SelectionBar";

const PHASE_LABEL: Record<string, string> = {
  prepared: "prepared",
  trained: "trained",
  tuned: "tuned",
  threshold_tuned: "threshold tuned",
  exported: "exported",
  exported_blocked: "exported (gate BLOCKED)",
  unknown: "unknown",
};

export function RunsView({ onToggleSidebar, onOpenChat }: { onToggleSidebar: () => void; onOpenChat: () => void }) {
  const { newChat } = useSessions();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    api
      .runs()
      .then((r) => {
        setRuns(r);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };
  useEffect(load, []);

  const sel = useSelection(runs?.map((r) => r.run_id) ?? []);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const deleteSelected = async () => {
    if (!window.confirm(`Delete ${sel.selected.length} run(s) and their files? This cannot be undone.`)) return;
    setBusy(true);
    const failed = await deleteEach(sel.selected, api.deleteRun);
    setBusy(false);
    sel.clear();
    setNotice(failed.length ? `Not deleted: ${failed.join(" · ")}` : null);
    load();
  };

  return (
    <div className="main">
      <div className="topbar">
        <button type="button" className="ghost sidebar-toggle" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          ☰
        </button>
        <div className="session-title-bar">Runs</div>
        <button type="button" className="ghost small" onClick={load}>
          Refresh
        </button>
      </div>
      <div className="log-wrap">
        <div className="page">
          <p className="muted">
            Your training runs, read from disk — where each got to and the next step. Runs are working state and are
            swept after the retention window; registered models live on under Models. A run behind a registered model
            can only be deleted after that model.
          </p>
          {error && <div className="connect-error">{error}</div>}
          {notice && <div className="connect-error">{notice}</div>}
          {runs === null && !error && <div className="muted">Loading…</div>}
          {runs?.length === 0 && <div className="muted">No runs yet — start a chat and train something.</div>}
          <SelectionBar
            what="runs"
            total={runs?.length ?? 0}
            count={sel.selected.length}
            allSelected={sel.allSelected}
            onToggleAll={sel.toggleAll}
            onDelete={() => void deleteSelected()}
            busy={busy}
          />
          {runs && runs.length > 0 && (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th />
                    <th>Run</th>
                    <th>Phase</th>
                    <th>Target</th>
                    <th>Model</th>
                    <th>Owner</th>
                    <th>Updated</th>
                    <th>Next step</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id}>
                      <td>
                        <input
                          type="checkbox"
                          className="check"
                          checked={sel.isSelected(r.run_id)}
                          onChange={() => sel.toggle(r.run_id)}
                          aria-label={`Select run ${r.run_id.slice(0, 8)}`}
                        />
                      </td>
                      <td>
                        <code title={r.run_id}>{r.run_id.slice(0, 8)}</code>
                      </td>
                      <td>
                        <span className={`phase phase-${r.phase}`}>{PHASE_LABEL[r.phase] ?? r.phase}</span>
                      </td>
                      <td>{r.target ?? "—"}</td>
                      <td>{r.model ?? "—"}</td>
                      <td>{r.owner ?? "—"}</td>
                      <td title={r.last_touched}>{new Date(r.last_touched).toLocaleString()}</td>
                      <td className="muted small">{r.suggested_next}</td>
                      <td>
                        <button
                          type="button"
                          className="small"
                          onClick={() => {
                            newChat(`Continue run ${r.run_id}: `);
                            onOpenChat();
                          }}
                        >
                          Continue
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
