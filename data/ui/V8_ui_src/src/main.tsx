import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

// Install the global UI Control Bus router once at boot.
// This lets Chat-driven automation reliably control stores/panels.
import { installSarahUiBus } from "@/stores/useSarahStore";

installSarahUiBus();

createRoot(document.getElementById("root")!).render(<App />);
