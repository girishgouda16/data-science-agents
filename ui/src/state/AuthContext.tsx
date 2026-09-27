import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setUnauthorizedHandler } from "../lib/api";
import {
  completeSignInIfReturning,
  initOidc,
  loadAuthConfig,
  setDevToken,
  signIn as oidcSignIn,
  signOut as oidcSignOut,
  type AuthConfig,
} from "../lib/auth";
import type { Me } from "../lib/types";

type Status = "loading" | "signed-out" | "signed-in" | "error";

interface AuthState {
  status: Status;
  config: AuthConfig | null;
  me: Me | null;
  error: string | null;
  signIn: () => void; // Keycloak redirect
  signInWithToken: (token: string) => Promise<void>; // dev mode only
  signOut: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>("loading");
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Signed in = the gateway accepts our token and says who we are.
  const check = useCallback(async () => {
    try {
      setMe(await api.me());
      setStatus("signed-in");
      setError(null);
    } catch {
      setMe(null);
      setStatus("signed-out");
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const cfg = await loadAuthConfig();
        setConfig(cfg);
        initOidc(cfg);
        await completeSignInIfReturning();
        await check();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      }
    })();
  }, [check]);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setMe(null);
      setStatus("signed-out");
    });
  }, []);

  const signInWithToken = useCallback(async (token: string) => {
    setDevToken(token.trim());
    try {
      setMe(await api.me());
      setStatus("signed-in");
      setError(null);
    } catch {
      setDevToken("");
      setError("That token was rejected — mint a fresh one (they expire after 24 h).");
      setStatus("signed-out");
    }
  }, []);

  const value: AuthState = {
    status,
    config,
    me,
    error,
    signIn: () => void oidcSignIn(),
    signInWithToken,
    signOut: () => {
      setMe(null);
      setStatus("signed-out");
      void oidcSignOut();
    },
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
