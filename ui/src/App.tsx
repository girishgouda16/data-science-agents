import { useState } from "react";
import "./App.css";
import { ChatView } from "./components/ChatView";
import { ModelsView } from "./components/ModelsView";
import { RunsView } from "./components/RunsView";
import { Sidebar, type View } from "./components/Sidebar";
import { SignInScreen } from "./components/SignInScreen";
import { AuthProvider, useAuth } from "./state/AuthContext";
import { SessionsProvider } from "./state/SessionsContext";

function Shell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState<View>("chat");
  const toggle = () => setSidebarOpen((v) => !v);
  return (
    <SessionsProvider>
      <div className="app-shell">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} view={view} onView={setView} />
        {sidebarOpen && <div className="scrim" aria-hidden onClick={() => setSidebarOpen(false)} />}
        {view === "chat" && <ChatView onToggleSidebar={toggle} />}
        {view === "runs" && <RunsView onToggleSidebar={toggle} onOpenChat={() => setView("chat")} />}
        {view === "models" && <ModelsView onToggleSidebar={toggle} />}
      </div>
    </SessionsProvider>
  );
}

function Gate() {
  const { status } = useAuth();
  return status === "signed-in" ? <Shell /> : <SignInScreen />;
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
