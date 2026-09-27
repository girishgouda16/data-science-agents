import { useState, type ReactNode } from "react";

/** react-markdown wraps every fenced code block in its own <pre> before
 * handing off to the `code` renderer below — CodeBlock builds its own
 * <pre> (inside the header+copy-button card), so the outer one here must
 * be a no-op or block code would render double-nested <pre><pre>. */
export function PreBlock({ children }: { children?: ReactNode }) {
  return <>{children}</>;
}

/** react-markdown's `code` renderer — invoked for both inline `code` and
 * fenced ```blocks```; only the latter (has a `className` like
 * "language-python" from remark-gfm) gets the header+copy-button treatment.
 * Inline code just renders as-is via the default <code> styling in App.css. */
export function CodeBlock({ className, children }: { className?: string; children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const language = /language-(\w+)/.exec(className ?? "")?.[1];

  if (!language) return <code className={className}>{children}</code>;

  const text = String(children).replace(/\n$/, "");
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language}</span>
        <button type="button" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}
