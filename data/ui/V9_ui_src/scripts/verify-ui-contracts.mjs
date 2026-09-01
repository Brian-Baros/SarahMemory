import { readFileSync } from "node:fs";

const checks = [
  ["panel chat bridge", "src/components/shell/PanelChatBridge.tsx", ["sarah:chat-draft", "openWindow(\"chat\")"]],
  ["chat draft receiver", "src/components/chat/ChatComposer.tsx", ["sarah:chat-pending-draft", "sarah:chat-draft"]],
  ["governed upload", "src/components/chat/ChatPanel.tsx", ["/api/files/upload", "SHA-256"]],
  ["explicit local ingestion", "src/components/chat/ChatPanel.tsx", ["/api/ingest/eat_this", "Ingest complete"]],
  ["microphone controls", "src/components/chat/ChatComposer.tsx", ["enumerateDevices", "getUserMedia", "SpeechRecognition"]],
  ["MCP JSON-RPC", "src/components/network/MCPConnectionsPanel.tsx", ["initialize", "tools/list", "resources/list", "prompts/list", "tools/call"]],
  ["MCP governance visibility", "src/components/network/MCPConnectionsPanel.tsx", ["/api/net/interop/status", "/api/net/interop/policy", "/api/net/interop/ingest"]],
  ["SarahNet MCP tab", "src/components/screens/sarah-net/SarahNetScreen.tsx", ["MCPConnectionsPanel", "value=\"mcp\""]],
  ["SarahNet GCAIOS fabric", "src/components/screens/sarah-net/SarahNetScreen.tsx", ["/api/net2/fabric/status", "/api/net/rt/status", "/api/self/gcaios/status?workload=sarahnet_xr", "value=\"fabric\"", "value=\"worlds\""]],
  ["commercial workstation shell", "src/components/shell/DesktopShell.tsx", ["DEFAULT_DESKTOP_SHORTCUTS", "WORKSPACE_PRESETS", "smaios-logo.jpg", "--wallpaper-brightness"]],
  ["draggable desktop shortcuts", "src/components/shell/DesktopShell.tsx", ["onShortcutPointerDown", "desktopShortcutPositions", "aria-grabbed", "pointer-events-none", "pointer-events-auto"]],
  ["custom desktop shortcuts", "src/components/shell/DesktopShell.tsx", ["Desktop Shortcuts", "Add Shortcut", "desktopShortcuts"]],
  ["custom shortcut deletion", "src/components/shell/DesktopShell.tsx", ["removeDesktopShortcut", "Delete ${shortcut.label}", "isPointOverTrashShortcut", "trashDropActive"]],
  ["desktop trash handoff", "src/components/shell/DesktopShell.tsx", ["desktop_trash", "files_open_trash", "openWindow(\"files\")", "sarahmemory:files:pending"]],
  ["audio mixer panel", "src/components/panels/audio-mixer/AudioMixerPanel.tsx", ["Master volume", "Bass", "Treble", "sarah:audio"]],
  ["taskbar audio tray", "src/components/StatusBar.tsx", ["AudioMixerPanel", "Open audio mixer", "masterVolume"]],
  ["appearance sliders", "src/components/screens/settings/SettingsScreen.tsx", ["Background brightness", "Background dim overlay", "Background blur", "Panel opacity"]],
  ["settings audio mixer", "src/components/screens/settings/SettingsScreen.tsx", ["AudioMixerPanel", "showSettingsButton={false}"]],
  ["mobile app launcher", "src/components/shell/MobileShell.tsx", ["SHELL_FEATURES", "Apps", "featureToMobileScreen"]],
  ["mobile audio mixer", "src/components/shell/MobileShell.tsx", ["AudioMixerPanel", "Audio", "audioMixerOpen"]],
  ["mobile camera vision", "src/components/shell/MobileShell.tsx", ["viewport.isPortrait", "Open Camera Vision object-recognition HUD", "window.open(\"/vision\""]],
  ["files trash surface", "src/components/screens/files/FilesScreen.tsx", ["files_open_trash", "openTrashView", "Trash is wired in the UI", "sarahmemory:files:pending"]],
  ["organized screen owners", "src/components/screens/README.md", ["Screen Component Layout", "files/FilesScreen.tsx", "vision/VisionScreen.tsx"]],
  ["organized panel owners", "src/components/panels/README.md", ["Panel Component Layout", "audio-mixer/AudioMixerPanel.tsx", "terminal/TerminalPanel.tsx"]],
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
