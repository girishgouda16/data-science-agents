import { useRef, useState, type ComponentPropsWithoutRef } from "react";
import { createPortal } from "react-dom";
import { rehypeChartGrid } from "../lib/chartGrid";
import { fileLabel, linkServerPaths, saveFile } from "../lib/files";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatTurn } from "../lib/types";
import { CodeBlock, PreBlock } from "./CodeBlock";

// Charts open full-size in an overlay (Esc or click closes; Cmd/Ctrl-click still opens a
// new tab); a broken one says so instead of an empty box.
function ChartImage({ src, alt }: ComponentPropsWithoutRef<"img">) {
  const [broken, setBroken] = useState(false);
  const zoom = useRef<HTMLDialogElement>(null);
  if (broken) return <span className="muted small">[chart unavailable: {alt || src}]</span>;
  return (
    <>
      <a
        href={src}
        target="_blank"
        rel="noreferrer"
        className="chart-link"
        onClick={(e) => {
          if (e.metaKey || e.ctrlKey || e.shiftKey) return;
          e.preventDefault();
          zoom.current?.showModal();
        }}
      >
        <img src={src} alt={alt ?? "chart"} loading="lazy" onError={() => setBroken(true)} />
      </a>
      {createPortal(
        <dialog ref={zoom} className="lightbox" aria-label={alt ?? "chart"} onClick={() => zoom.current?.close()}>
          <img src={src} alt={alt ?? "chart"} loading="lazy" />
        </dialog>,
        document.body,
      )}
    </>
  );
}

// Links the agent wrote open in a new tab; "#download=<path>" (see linkServerPaths) downloads the user's file.
function Link({ href = "", children }: ComponentPropsWithoutRef<"a">) {
  const [error, setError] = useState("");
  if (!href.startsWith("#download=")) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
  const path = decodeURIComponent(href.slice("#download=".length));
  return (
    <button type="button" className="file-chip" title={`Download ${fileLabel(path)}`}
      onClick={() => saveFile(path).catch((e) => setError(String(e.message ?? e)))}>
      {children} ⬇{error && <span className="connect-error"> {error}</span>}
    </button>
  );
}

const MD_COMPONENTS = { code: CodeBlock, pre: PreBlock, img: ChartImage, a: Link };

// Chats from before attachments were stored separately carry "Dataset: <path>" in the text.
function userParts(turn: ChatTurn): { text: string; attachment?: string } {
  const legacy = /^Dataset: (\S+)\n/.exec(turn.text);
  return legacy ? { text: turn.text.slice(legacy[0].length), attachment: fileLabel(legacy[1]) } : turn;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function MessageBubble({ turn }: { turn: ChatTurn }) {
  const isAgent = turn.role === "agent";
  const user = isAgent ? null : userParts(turn);
  return (
    <div className={`row ${turn.role}`}>
      <div className="turn">
        {isAgent && <div className="avatar" aria-hidden>◆</div>}
        <div className="bubble">
          {isAgent ? (
            <div className="md">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeChartGrid]} components={MD_COMPONENTS}>
                {linkServerPaths(turn.text)}
              </ReactMarkdown>
            </div>
          ) : (
            <>
              {user?.attachment && <div className="file-chip static">📄 {user.attachment}</div>}
              {user?.text}
            </>
          )}
        </div>
        {isAgent && (
          <div className="bubble-actions">
            <CopyButton text={turn.text} />
          </div>
        )}
      </div>
    </div>
  );
}

export function WorkingBubble({ sending }: { sending: boolean }) {
  return (
    <div className="row agent">
      <div className="turn">
        <div className="avatar" aria-hidden>◆</div>
        <div className="bubble pending">
          <span className="typing" aria-hidden>
            <i />
            <i />
            <i />
          </span>
          {sending ? "sending…" : "the agents are working — long steps like training can take a few minutes"}
        </div>
      </div>
    </div>
  );
}
