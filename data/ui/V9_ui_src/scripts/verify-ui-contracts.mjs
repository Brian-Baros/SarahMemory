import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = [resolve(scriptDir, ".."), scriptDir, process.cwd()].find(
  (candidate) =>
    existsSync(resolve(candidate, "src")) &&
    existsSync(resolve(candidate, "public")),
);

if (!projectRoot) {
  console.error(
    "FAIL UI contract runner: unable to locate the UI project root (expected src/ and public/ directories).",
  );
  process.exitCode = 1;
} else {
  const checks = [
    ["panel chat context wrapper", "src/components/shell/PanelChatBridge.tsx", ["data-panel-chat-context", "data-feature-purpose"]],
    ["panel chat bridge", "src/components/shell/PanelChatBridge.tsx", ["sarah:chat-draft", "openWindow(\"chat\")"]],
    ["titlebar ask sarah", "src/components/shell/Window.tsx", ["openWindow,", "Ask Sarah", "sarah:chat-draft", "openWindow(\"chat\")"]],
    ["chat draft receiver", "src/components/chat/ChatComposer.tsx", ["sarah:chat-pending-draft", "sarah:chat-draft"]],
    ["governed upload", "src/components/chat/ChatPanel.tsx", ["/api/files/upload", "SHA-256"]],
    ["explicit local ingestion", "src/components/chat/ChatPanel.tsx", ["/api/ingest/eat_this", "Ingest complete"]],
    ["microphone controls", "src/components/chat/ChatComposer.tsx", ["enumerateDevices", "getUserMedia", "SpeechRecognition"]],
    ["MCP JSON-RPC", "src/components/network/MCPConnectionsPanel.tsx", ["initialize", "tools/list", "resources/list", "prompts/list", "tools/call"]],
    ["MCP governance visibility", "src/components/network/MCPConnectionsPanel.tsx", ["/api/net/interop/status", "/api/net/interop/policy", "/api/net/interop/ingest"]],
    ["SarahNet MCP tab", "src/components/screens/sarah-net/SarahNetScreen.tsx", ["MCPConnectionsPanel", "value=\"mcp\""]],
    ["SarahNet GCAIOS fabric", "src/components/screens/sarah-net/SarahNetScreen.tsx", ["/api/net2/fabric/status", "/api/net/rt/status", "/api/self/gcaios/status?workload=sarahnet_xr", "value=\"fabric\"", "value=\"worlds\""]],
    ["commercial workstation shell", "src/components/shell/DesktopShell.tsx", ["DEFAULT_DESKTOP_SHORTCUTS", "WORKSPACE_PRESETS", "smaios-logo.jpg", "--wallpaper-brightness"]],
    ["nexus power sleep overlay", "src/components/shell/DesktopShell.tsx", ["sarah:system-sleep", "sleepModeActive", "/api/avatar/rem/stop"]],
    ["draggable desktop shortcuts", "src/components/shell/DesktopShell.tsx", ["onShortcutPointerDown", "desktopShortcutPositions", "aria-grabbed", "pointer-events-none", "pointer-events-auto"]],
    ["custom desktop shortcuts", "src/components/shell/DesktopShell.tsx", ["Desktop Shortcuts", "Add Shortcut", "desktopShortcuts"]],
    ["custom shortcut deletion", "src/components/shell/DesktopShell.tsx", ["removeDesktopShortcut", "Delete ${shortcut.label}", "isPointOverTrashShortcut", "trashDropActive"]],
    ["desktop trash handoff", "src/components/shell/DesktopShell.tsx", ["desktop_trash", "files_open_trash", "openWindow(\"files\")", "sarahmemory:files:pending"]],
    ["audio mixer panel", "src/components/panels/audio-mixer/AudioMixerPanel.tsx", ["Master volume", "Bass", "Treble", "sarah:audio"]],
    ["taskbar audio tray", "src/components/StatusBar.tsx", ["AudioMixerPanel", "Open audio mixer", "masterVolume"]],
    ["taskbar clock panel", "src/components/StatusBar.tsx", ["SystemClockPanel", "Open system clock", "PopoverContent"]],
    ["system clock controls", "src/components/panels/system-clock/SystemClockPanel.tsx", ["manualClockDate", "/api/system/time-authority", "/api/system/clock-court", "SelectTrigger", "supportedValuesOf", "timezonePreview"]],
    ["clock popover containment", "src/components/StatusBar.tsx", ["SystemClockPanel", "w-auto max-w-[calc(100vw-1rem)]"]],
    ["taskbar network device manager", "src/components/StatusBar.tsx", ["Open Network Device Manager", "sarahmemory:device-manager:tab", "device-manager"]],
    ["nexus power options", "src/components/shell/AiOSShellCenter.tsx", ["Power Options", "Power Down", "Reboot", "Sleep Mode", "/api/avatar/rem/start", "system.power.request"]],
    ["device manager screen", "src/components/screens/device-manager/DeviceManagerScreen.tsx", ["/api/drivers", "/api/drivers/capabilities", "/api/drivers/governance", "Networks", "SSID", "VPN Profile"]],
    ["dynamic device inventory", "src/components/screens/device-manager/DeviceManagerScreen.tsx", ["INVENTORY_ENDPOINTS", "/api/self/hardware-topology", "/api/self/body-capabilities", "/api/vision/devices", "collectBrowserDevices", "mergeDevices", "Discover", "Status"]],
    ["class-specific device settings", "src/components/screens/device-manager/DeviceManagerScreen.tsx", ["DEVICE_PROFILES", "DeviceSettingsEditor", "Network Adapter Options", "Camera Options", "Display Options", "Audio Device Options", "Input Device Options", "Printer Options", "Storage Device Options", "CAMERA_FIELDS", "DISPLAY_FIELDS", "AUDIO_FIELDS", "INPUT_FIELDS", "PRINTER_FIELDS", "STORAGE_FIELDS"]],
    ["hardcore device option coverage", "src/components/screens/device-manager/DeviceManagerScreen.tsx", ["Mirror Preview", "Brightness", "Zoom", "Resolution", "Color Temperature", "HDR Mode", "Mono", "Stereo", "5.1 Surround", "Bass", "Treble", "Reverse Mouse Clicking", "Pointer Trail", "Scroll Wheel Lines", "Duplex", "Write Cache", "VPN Profile", "Gateway", "Subnet Mask / Prefix"]],
    ["device manager functional guardrails", "src/components/screens/device-manager/DeviceManagerScreen.tsx", ["function driverKey", "configSourceForDevice", "selectedKey", "visibleDrivers.find", "body: { config: configPatch", "payload: { ...configPatch, action }", "registry update failed", "configuration save failed", "config read failed", "finally {\n      setBusy(\"\");", "bridgeDriverId ? `driver:${driver.bridgeDriverId}`"]],
    ["theme token coverage", "public/themes/Dark_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["light theme token coverage", "public/themes/Light_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["matrix theme token coverage", "public/themes/Matrix_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["tron theme token coverage", "public/themes/Tron.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["hal theme token coverage", "public/themes/HAL2000_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["skynet theme token coverage", "public/themes/Skynet_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["vibrant theme token coverage", "public/themes/Vibrant_Theme.css", ["--background", "--foreground", "--primary-rgb", "--sarah-material-alpha", "--status-online"]],
    ["appearance sliders", "src/components/screens/settings/SettingsScreen.tsx", ["Background brightness", "Background dim overlay", "Background blur", "Panel opacity"]],
    ["settings audio mixer", "src/components/screens/settings/SettingsScreen.tsx", ["AudioMixerPanel", "showSettingsButton={false}"]],
    ["settings MSDC device witness", "src/components/screens/settings/SettingsScreen.tsx", ["/api/drivers", "MSDC Device Witness", "present, healthy, ready, authorized, and active"]],
    ["mobile app launcher", "src/components/shell/MobileShell.tsx", ["SHELL_FEATURES", "Apps", "featureToMobileScreen"]],
    ["mobile audio mixer", "src/components/shell/MobileShell.tsx", ["AudioMixerPanel", "Audio", "audioMixerOpen"]],
    ["mobile camera vision", "src/components/shell/MobileShell.tsx", ["viewport.isPortrait", "Open Camera Vision object-recognition HUD", "window.open(\"/vision\""]],
    ["files trash surface", "src/components/screens/files/FilesScreen.tsx", ["files_open_trash", "openTrashView", "Trash is wired in the UI", "sarahmemory:files:pending"]],
    ["organized screen owners", "src/components/screens/README.md", ["Screen Component Layout", "files/FilesScreen.tsx", "vision/VisionScreen.tsx"]],
    ["organized panel owners", "src/components/panels/README.md", ["Panel Component Layout", "audio-mixer/AudioMixerPanel.tsx", "terminal/TerminalPanel.tsx"]],
    ["local production bridge", "src/lib/config.ts", ["http://127.0.0.1:8000", "Local same-origin"]],
  ];

  let passed = 0;
  let failed = 0;
  let missingFileChecks = 0;
  const missingFilePaths = new Set();
  const sourceCache = new Map();

  console.log(`UI contract root: ${projectRoot}`);

  for (const [label, relativePath, needles] of checks) {
    const fullPath = resolve(projectRoot, relativePath);

    if (!existsSync(fullPath)) {
      failed += 1;
      missingFileChecks += 1;
      missingFilePaths.add(relativePath);
      console.error(`FAIL ${label}: target file missing ${relativePath}`);
      continue;
    }

    let source = sourceCache.get(fullPath);
    if (source === undefined) {
      try {
        source = readFileSync(fullPath, "utf8");
        sourceCache.set(fullPath, source);
      } catch (error) {
        failed += 1;
        console.error(`FAIL ${label}: unable to read ${relativePath}: ${error.message}`);
        continue;
      }
    }

    const missing = needles.filter((needle) => !source.includes(needle));
    if (missing.length) {
      failed += 1;
      console.error(`FAIL ${label}: missing ${missing.join(", ")}`);
    } else {
      passed += 1;
      console.log(`PASS ${label}`);
    }
  }

  console.log(
    `UI contract summary: ${passed}/${checks.length} passed, ${failed} failed, ${missingFilePaths.size} missing target file(s) affecting ${missingFileChecks} check(s)`,
  );

  if (failed) {
    process.exitCode = 1;
  } else {
    console.log(`PASS ${checks.length}/${checks.length} UI contract checks`);
  }
}
