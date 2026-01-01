import React, { useEffect } from "react";
import { Toaster } from "./components/ui/toaster";
import { Toaster as Sonner } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

import { api } from "./lib/api";
import { useSarahStore } from "./stores/useSarahStore"; // ✅ THIS MUST MATCH YOUR FILE NAME

const queryClient = new QueryClient();

// Simple error boundary so we DON'T get a blank white screen on crash
class AppErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error?: any }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  componentDidCatch(error: any, info: any) {
    // eslint-disable-next-line no-console
    console.error("[AppErrorBoundary] Crash:", error, info);
  }

  render() {
    if (this.state.hasError) {
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
            }}
          >
            {String(this.state.error?.message || this.state.error || "Unknown error")}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

const App = () => {
  const setBootstrapData = useSarahStore((s) => s.setBootstrapData);
  const setBackendReady = useSarahStore((s) => s.setBackendReady);
  const playWelcomeIfNeeded = useSarahStore((s) => s.playWelcomeIfNeeded);

  useEffect(() => {
    let cancelled = false;

    const boot = async () => {
      try {
        const res = await api.bootstrap.init();
        if (cancelled) return;

        // Store bootstrap (even if ok=false)
        setBootstrapData(res as any);

        // Mark backend ready (this is what your greeting checks)
        setBackendReady(true);

        // Fire greeting after state is set
        setTimeout(() => {
          if (!cancelled) playWelcomeIfNeeded();
        }, 0);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[App] bootstrap failed, continuing:", err);

        if (cancelled) return;

        // Still allow greeting text even if backend is down
        setBackendReady(true);

        setTimeout(() => {
          if (!cancelled) playWelcomeIfNeeded();
        }, 0);
      }
    };

    boot();
    return () => {
      cancelled = true;
    };
  }, [setBootstrapData, setBackendReady, playWelcomeIfNeeded]);

  return (
    <AppErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </AppErrorBoundary>
  );
};

export default App;
