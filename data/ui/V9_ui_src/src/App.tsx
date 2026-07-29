import React, { useEffect, useRef } from "react";
import type { ErrorInfo } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

import Index from "@/pages/Index";
import NotFound from "@/pages/NotFound";
import { VisionScreen } from "@/components/screens/VisionScreen";

import { api } from "@/lib/api";
import { apiFetch } from "@/lib/config";
import { installSarahUiBus, useSarahStore } from "@/stores/useSarahStore";

const queryClient = new QueryClient();

class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: unknown }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: undefined };
  }

  static getDerivedStateFromError(error: unknown) {
    return { hasError: true, error };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("[AppErrorBoundary] Crash:", error, info);
  }

  render() {
    if (this.state.hasError) {
      let message = "Unknown error";

      if (this.state.error instanceof Error) message = this.state.error.message;
      else if (typeof this.state.error === "string") message = this.state.error;
      else if (this.state.error) {
        try {
          message = JSON.stringify(this.state.error, null, 2);
        } catch {
          message = String(this.state.error);
        }
      }

      return (
        <div style={{ padding: 16, fontFamily: "ui-sans-serif, system-ui" }}>
          <h1 style={{ fontSize: 18, fontWeight: 700 }}>SarahMemory UI crashed</h1>
          <p style={{ marginTop: 8, opacity: 0.9 }}>Check the console for details. Here’s the error:</p>
          <pre
            style={{
              marginTop: 12,
              padding: 12,
              background: "rgba(0,0,0,0.08)",
              borderRadius: 8,
              overflow: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {message}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

export default function App() {
  const setBootstrapData = useSarahStore((s) => s.setBootstrapData);
  const setBackendReady = useSarahStore((s) => s.setBackendReady);
  const enqueueUiActions = useSarahStore((s) => s.enqueueUiActions);

  // StrictMode guard
  const didBootRef = useRef(false);

  useEffect(() => {
    installSarahUiBus();
  }, []);

  useEffect(() => {
    let stopped = false;
    let busy = false;

    const pollUiActions = async () => {
      if (stopped || busy) return;
      busy = true;
      try {
        // SARAHMEMORY_PATCH_NOTE 2026-06-24:
        // Use the same governed API resolver as the rest of the UI. Raw
        // same-origin fetch breaks when pywebview/dev-server origin differs
        // from the Flask backend. apiFetch pins this to window.SARAH_API_BASE.
        const data = await apiFetch<any>("/api/ui/actions/poll?surface=webui&limit=25", { method: "GET" });
        const actions = Array.isArray(data?.actions) ? data.actions : [];
        if (actions.length > 0) enqueueUiActions(actions, "backend_queue");
      } catch {
        // backend queue is optional; do not destabilize the UI
      } finally {
        busy = false;
      }
    };

    void pollUiActions();
    const timer = window.setInterval(() => void pollUiActions(), 1500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [enqueueUiActions]);

  useEffect(() => {
    if (didBootRef.current) return;
    didBootRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        const res = await api.bootstrap.init();
        if (cancelled) return;

        // Your store will set backendReady and fire welcome (if ok)
        setBootstrapData(res);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[App] bootstrap failed, continuing:", err);
        if (cancelled) return;

        // Keep the shell visible, but do not falsely mark backend ready.
        // The UI should render while status indicators show the real backend state.
        setBackendReady(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [setBootstrapData, setBackendReady]);

  return (
    <AppErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/vision" element={<VisionScreen />} />
              <Route path="/vr-hud" element={<VisionScreen />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </AppErrorBoundary>
  );
}
