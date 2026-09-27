import { useEffect, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import { api } from "../lib/api";
import { useSessions } from "../state/SessionsContext";

interface ComposerProps {
  onSend: (message: string, filePath?: string) => void;
  disabled: boolean;
  isNewChat: boolean;
  placeholder: string;
}

const MB = 1024 * 1024;

export function Composer({ onSend, disabled, isNewChat, placeholder }: ComposerProps) {
  const { draft, setDraft, project, setProject, autopilot, setAutopilot } = useSessions();
  const [attachment, setAttachment] = useState<{ path: string; name: string } | null>(null);
  const [upload, setUpload] = useState<{ busy: boolean; error: string | null }>({ busy: false, error: null });
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + el.offsetHeight - el.clientHeight}px`; // + border (box-sizing: border-box)
    if (draft) el.focus();
  }, [draft]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!draft.trim() || disabled) return;
    onSend(draft.trim(), attachment?.path);
    setDraft("");
    setAttachment(null);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(e);
    }
  };

  const uploadFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUpload({ busy: false, error: "Only CSV files can be uploaded." });
      return;
    }
    setUpload({ busy: true, error: null });
    try {
      const { file_path } = await api.uploadDataset(file);
      setAttachment({ path: file_path, name: `${file.name} · ${(file.size / MB).toFixed(1)} MB` });
      setUpload({ busy: false, error: null });
    } catch (err) {
      setUpload({ busy: false, error: err instanceof Error ? err.message : String(err) });
    }
  };

  const onFileChosen = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-picking the same file
    if (file) void uploadFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && !disabled) void uploadFile(file);
  };

  return (
    <div
      className={`composer-wrap ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <div className="composer">
        {upload.error && <div className="connect-error">Upload failed: {upload.error}</div>}
        <div className="composer-box">
          {attachment && (
            <div className="attach-chip">
              <span>📄 {attachment.name}</span>
              <button type="button" onClick={() => setAttachment(null)} aria-label="Remove attachment">
                ✕
              </button>
            </div>
          )}
          <form id="composer-form" className="input-bar" onSubmit={submit}>
            <textarea
              ref={textareaRef}
              rows={1}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              aria-label="Message"
              autoFocus
            />
          </form>
          {/* Outside the <form> so Enter in the Project field can't send the message. */}
          <div className="composer-tools">
            <button
              type="button"
              className="icon-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={upload.busy || disabled}
              title="Attach a CSV dataset"
              aria-label="Attach a CSV dataset"
            >
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"
                />
              </svg>
            </button>
            <input ref={fileInputRef} type="file" accept=".csv,text/csv" hidden onChange={onFileChosen} />
            {isNewChat && (
              <label className="tool-chip project-field" title="Models from this chat register under this team project">
                Project
                <input
                  value={project}
                  onChange={(e) => setProject(e.target.value)}
                  placeholder="optional, e.g. retention"
                  maxLength={60}
                />
              </label>
            )}
            <label
              className="tool-chip autopilot-field"
              title="Agents answer their own questions with their recommended option instead of asking you"
            >
              <input
                type="checkbox"
                role="switch"
                checked={autopilot}
                onChange={(e) => setAutopilot(e.target.checked)}
              />
              Autopilot
            </label>
            {upload.busy && <span className="muted small">Uploading…</span>}
            <button
              type="submit"
              form="composer-form"
              className="primary send-btn"
              disabled={!draft.trim() || disabled}
              title="Send"
              aria-label="Send"
            >
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                <path fill="currentColor" d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z" />
              </svg>
            </button>
          </div>
        </div>
        {dragging && <div className="drop-hint">Drop the CSV to attach it</div>}
      </div>
    </div>
  );
}
