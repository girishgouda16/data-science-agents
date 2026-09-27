import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { deleteEach, useSelection } from "../lib/selection";
import type { ModelSummary, ModelVersions } from "../lib/types";
import { SelectionBar } from "./SelectionBar";

// The headline metrics worth showing per problem type; the rest stay in MLflow.
const KEY_METRICS = ["recall_positive", "precision_positive", "pr_auc", "roc_auc", "f1", "r2", "mae", "rmse", "mape",
  "silhouette", "detection_rate", "wape"];

// Promotion audit tags are per alias: "champion.approved_by", "champion.reason", "champion.forced".
function approvals(tags: Record<string, string>, aliases: string[]): string {
  return aliases
    .filter((a) => tags[`${a}.approved_by`])
    .map((a) => {
      const forced = tags[`${a}.forced`] && tags[`${a}.forced`] !== "false" ? " — FORCED" : "";
      const reason = tags[`${a}.reason`] ? ` (“${tags[`${a}.reason`]}”)` : "";
      return `${a}: ${tags[`${a}.approved_by`]}${forced}${reason}`;
    })
    .join("; ");
}

function fmt(n: number): string {
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(3);
}

function Versions({ name }: { name: string }) {
  const [data, setData] = useState<ModelVersions | null>(null);
  useEffect(() => {
    api.modelVersions(name).then(setData).catch((e) => setData({ error: String(e) }));
  }, [name]);
  if (!data) return <div className="muted small">Loading versions…</div>;
  if (data.error) return <div className="connect-error">{data.error}</div>;
  return (
    <table className="data-table nested">
      <thead>
        <tr>
          <th>Version</th>
          <th>Aliases</th>
          <th>Algorithm</th>
          <th>Readiness</th>
          <th>Metrics</th>
          <th>Owner</th>
          <th>Promotion</th>
        </tr>
      </thead>
      <tbody>
        {data.versions?.map((v) => (
          <tr key={v.version}>
            <td>v{v.version}</td>
            <td>{v.aliases.map((a) => <span key={a} className={`badge alias-${a}`}>{a}</span>)}</td>
            <td>{v.model ?? "—"}</td>
            <td>
              <span className={`phase readiness-${v.readiness}`}>{v.readiness ?? "—"}</span>
            </td>
            <td className="small">
              {KEY_METRICS.filter((k) => k in v.metrics)
                .slice(0, 3)
                .map((k) => `${k} ${fmt(v.metrics[k])}`)
                .join(" · ") || "—"}
            </td>
            <td>{v.version_tags.owner ?? "—"}</td>
            <td className="small">{approvals(v.version_tags, v.aliases) || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ModelsView({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const load = () => {
    api
      .models()
      .then((r) => {
        setError(r.error ?? null);
        setModels(r.models ?? []);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };
  useEffect(load, []);

  // A model with a champion is what serving loads — it has to be demoted (in a chat) before it can go.
  const sel = useSelection(models?.filter((m) => !("champion" in m.aliases)).map((m) => m.name) ?? []);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const deleteSelected = async () => {
    const n = sel.selected.length;
    if (!window.confirm(`Delete ${n} model(s) and every version from the registry? This cannot be undone.`)) return;
    setBusy(true);
    const failed = await deleteEach(sel.selected, api.deleteModel);
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
        <div className="session-title-bar">Model registry</div>
        <button type="button" className="ghost small" onClick={load}>
          Refresh
        </button>
      </div>
      <div className="log-wrap">
        <div className="page">
          <p className="muted">
            Every registered model and where its aliases point. <strong>champion</strong> is what serving loads.
            Promote or roll back by asking in a chat — promotion needs your confirmation and a reason, and only the
            owner or an ml-admin can move an alias or delete a model. A model with a champion can't be deleted until
            it is demoted.
          </p>
          {error && <div className="connect-error">{error}</div>}
          {notice && <div className="connect-error">{notice}</div>}
          {models === null && !error && <div className="muted">Loading…</div>}
          {models?.length === 0 && <div className="muted">Nothing registered yet.</div>}
          <SelectionBar
            what="models"
            total={models?.length ?? 0}
            count={sel.selected.length}
            allSelected={sel.allSelected}
            onToggleAll={sel.toggleAll}
            onDelete={() => void deleteSelected()}
            busy={busy}
          />
          {models?.map((m) => (
            <div key={m.name} className="model-card">
              <div className="model-row">
                <input
                  type="checkbox"
                  className="check"
                  checked={sel.isSelected(m.name)}
                  onChange={() => sel.toggle(m.name)}
                  disabled={"champion" in m.aliases}
                  title={"champion" in m.aliases ? "Serving loads its champion — demote it in a chat first" : undefined}
                  aria-label={`Select model ${m.name}`}
                />
                <button
                  type="button"
                  className="model-head"
                  aria-expanded={open === m.name}
                  onClick={() => setOpen(open === m.name ? null : m.name)}
                >
                  <span className="model-name">{m.name}</span>
                  <span className="model-aliases">
                    {Object.entries(m.aliases).map(([alias, v]) => (
                      <span key={alias} className={`badge alias-${alias}`}>
                        {alias} v{v}
                      </span>
                    ))}
                    <span className="muted small">latest v{m.latest_version ?? "?"}</span>
                  </span>
                </button>
              </div>
              {open === m.name && <Versions name={m.name} />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
