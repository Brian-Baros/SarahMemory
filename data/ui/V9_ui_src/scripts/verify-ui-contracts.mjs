import { readFileSync } from "node:fs";

const checks = [
  ["panel chat bridge", "src/components/shell/PanelChatBridge.tsx", ["sarah:chat-draft", "openWindow(\"chat\")"]],
  ["chat draft receiver", "src/components/chat/ChatComposer.tsx", ["sarah:chat-pending-draft", "sarah:chat-draft"]],
  ["governed upload", "src/components/chat/ChatPanel.tsx", ["/api/files/upload", "SHA-256"]],
  ["explicit local ingestion", "src/components/chat/ChatPanel.tsx", ["/api/ingest/eat_this", "Ingest complete"]],
  ["microphone controls", "src/components/chat/ChatComposer.tsx", ["enumerateDevices", "getUserMedia", "SpeechRecognition"]],
  ["MCP JSON-RPC", "src/components/network/MCPConnectionsPanel.tsx", ["initialize", "tools/list", "resources/list", "prompts/list", "tools/call"]],
  ["MCP governance visibility", "src/components/network/MCPConnectionsPanel.tsx", ["/api/net/interop/status", "/api/net/interop/policy", "/api/net/interop/ingest"]],
  ["SarahNet MCP tab", "src/components/screens/SarahNetScreen.tsx", ["MCPConnectionsPanel", "value=\"mcp\""]],
  ["commercial workstation shell", "src/components/shell/DesktopShell.tsx", ["DESKTOP_SHORTCUTS", "WORKSPACE_PRESETS", "smaios-logo.jpg", "--wallpaper-brightness"]],
  ["appearance sliders", "src/components/screens/SettingsScreen.tsx", ["Background brightness", "Background dim overlay", "Background blur", "Panel opacity"]],
  ["mobile camera vision", "src/components/shell/MobileShell.tsx", ["viewport.isPortrait", "Open Camera Vision object-recognition HUD", "window.open(\"/vision\""]],
  ["local production bridge", "src/lib/config.ts", ["http://127.0.0.1:8000", "Local same-origin"]],
];

let failed = 0;
for (const [label, path, needles] of checks) {
  const source = readFileSync(path, "utf8");
  const missing = needles.filter((needle) => !source.includes(needle));
  if (missing.length) {
    failed += 1;
    console.error(`FAIL ${label}: missing ${missing.join(", ")}`);
  } else {
    console.log(`PASS ${label}`);
  }
}

if (failed) process.exit(1);
console.log(`PASS ${checks.length}/${checks.length} UI contract checks`);
