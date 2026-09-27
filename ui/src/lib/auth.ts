import { UserManager, WebStorageStateStore } from "oidc-client-ts";

// How to sign in is the gateway's call (GET /api/v1/auth/config):
//   oidc — Keycloak (or any OIDC provider): authorization code + PKCE, tokens
//          kept in sessionStorage and renewed silently before they expire.
//   dev  — no identity provider: paste a token from `python -m scripts.print_token <you>`.
export type AuthConfig = { mode: "oidc"; issuer: string; client_id: string } | { mode: "dev" };

const DEV_TOKEN_KEY = "apiToken";
let manager: UserManager | null = null;

export async function loadAuthConfig(): Promise<AuthConfig> {
  const res = await fetch(`${apiBase()}/api/v1/auth/config`);
  if (!res.ok) throw new Error(`gateway answered ${res.status} for /api/v1/auth/config`);
  return res.json();
}

export function apiBase(): string {
  // Same origin in production (the gateway serves the UI) and in dev (Vite proxies /api).
  return (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
}

export function initOidc(config: AuthConfig): UserManager | null {
  if (config.mode !== "oidc") return null;
  manager ??= new UserManager({
    authority: config.issuer,
    client_id: config.client_id,
    redirect_uri: `${window.location.origin}/`,
    post_logout_redirect_uri: `${window.location.origin}/`,
    response_type: "code",
    scope: "openid profile email",
    automaticSilentRenew: true,
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
  });
  return manager;
}

/** The bearer token for the next API call, or "" when signed out. */
export async function accessToken(): Promise<string> {
  if (manager) {
    const user = await manager.getUser();
    return user && !user.expired ? user.access_token : "";
  }
  return localStorage.getItem(DEV_TOKEN_KEY) ?? "";
}

export function setDevToken(token: string): void {
  if (token) localStorage.setItem(DEV_TOKEN_KEY, token);
  else localStorage.removeItem(DEV_TOKEN_KEY);
}

/** Finish a Keycloak redirect if this page load is one; true when it was. */
export async function completeSignInIfReturning(): Promise<boolean> {
  if (!manager) return false;
  const params = new URLSearchParams(window.location.search);
  if (!params.has("code") || !params.has("state")) return false;
  await manager.signinRedirectCallback();
  window.history.replaceState({}, document.title, window.location.pathname);
  return true;
}

export async function signIn(): Promise<void> {
  await manager?.signinRedirect();
}

export async function signOut(): Promise<void> {
  setDevToken("");
  if (manager) await manager.signoutRedirect();
}
