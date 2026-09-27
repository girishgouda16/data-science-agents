import { useState, type FormEvent } from "react";
import { useAuth } from "../state/AuthContext";

export function SignInScreen() {
  const { status, config, error, signIn, signInWithToken } = useAuth();
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const submitToken = async (e: FormEvent) => {
    e.preventDefault();
    if (!token.trim()) return;
    setBusy(true);
    await signInWithToken(token);
    setBusy(false);
  };

  return (
    <div className="connect-screen">
      <div className="connect-card">
        <div className="brand">
          <span className="brand-mark">◆</span> Holdout
        </div>
        <p>
          Train, evaluate, promote and monitor models by talking to a team of ML agents — from EDA to MLOps.
        </p>

        {status === "loading" && <div className="muted">Checking your session…</div>}

        {status === "error" && (
          <div className="connect-error">Can't reach the gateway: {error}</div>
        )}

        {status === "signed-out" && config?.mode === "oidc" && (
          <button type="button" className="primary wide" onClick={signIn}>
            Sign in with your company account
          </button>
        )}

        {status === "signed-out" && config?.mode === "dev" && (
          <form className="connect-form" onSubmit={submitToken}>
            <p className="muted small">
              Development mode (no identity provider configured). Mint a token from the repo root with{" "}
              <code>python -m scripts.print_token you</code> and paste it here.
            </p>
            <div>
              <label htmlFor="token">Dev token</label>
              <input
                id="token"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="eyJ..."
                autoFocus
              />
            </div>
            <div className="connect-error">{error}</div>
            <button type="submit" className="primary" disabled={!token.trim() || busy}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
