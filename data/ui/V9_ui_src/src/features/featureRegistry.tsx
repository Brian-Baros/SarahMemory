import type { WindowId } from "@/stores/useWindowStore";
import type { MobileScreen } from "@/stores/useNavigationStore";

import { ChatPanel } from "@/components/chat/ChatPanel";
import { HistoryScreen } from "@/components/screens/HistoryScreen";
import { FilesScreen } from "@/components/screens/FilesScreen";
import { ResearchScreen } from "@/components/screens/ResearchScreen";
import { StudiosScreen } from "@/components/screens/StudiosScreen";
import { AvatarScreen } from "@/components/screens/AvatarScreen";
import { SarahNetScreen } from "@/components/screens/SarahNetScreen";
import { MediaScreen } from "@/components/screens/MediaScreen";
import { DLEngineScreen } from "@/components/screens/DLEngineScreen";
import { SettingsScreen } from "@/components/screens/SettingsScreen";
import { AddonsScreen } from "@/components/screens/AddonsScreen";
import NAILDEScreen from "@/components/screens/NAILDEScreen";
import TerminalScreen from "@/components/screens/TerminalScreen";
import { PanelChatBridge } from "@/components/shell/PanelChatBridge";

export type ShellFeatureId = WindowId | MobileScreen | "studios";

export type ShellFeatureArea =
  | "conversation"
  | "assistant"
  | "operator"
  | "files"
  | "research"
  | "creation"
  | "network"
  | "media"
  | "settings";

export type ShellFeatureDefinition = {
  id: WindowId;
  aliases?: string[];
  title: string;
  area: ShellFeatureArea;
  purpose: string;
  sourceFile: string;
  component: JSX.Element;
};

export const SHELL_FEATURES: ShellFeatureDefinition[] = [
  {
    id: "chat",
    title: "Chat",
    area: "conversation",
    purpose: "Main Sarah conversation, backend chat calls, avatar speaking events, file ingestion.",
    sourceFile: "src/components/chat/ChatPanel.tsx",
    component: <ChatPanel />,
  },
  {
    id: "history",
    title: "History",
    area: "conversation",
    purpose: "Conversation history and session recall surface.",
    sourceFile: "src/components/screens/HistoryScreen.tsx",
    component: <HistoryScreen />,
  },
  {
    id: "files",
    title: "Files",
    area: "files",
    purpose: "Local-first file browser and file action UI.",
    sourceFile: "src/components/screens/FilesScreen.tsx",
    component: <FilesScreen />,
  },
  {
    id: "research",
    title: "Research",
    area: "research",
    purpose: "Research lane, web/local evidence gathering, and source review.",
    sourceFile: "src/components/screens/ResearchScreen.tsx",
    component: <ResearchScreen />,
  },
  {
    id: "studio",
    aliases: ["studios"],
    title: "Studios",
    area: "creation",
    purpose: "Creative media generation modules and studio tabs.",
    sourceFile: "src/components/screens/StudiosScreen.tsx",
    component: <StudiosScreen />,
  },
  {
    id: "avatar",
    title: "Avatar",
    area: "assistant",
    purpose: "2D/3D avatar, local camera overlay, voice state, mirror/media modes.",
    sourceFile: "src/components/screens/AvatarScreen.tsx",
    component: <AvatarScreen />,
  },
  {
    id: "sarahnet",
    title: "SarahNet",
    area: "network",
    purpose: "SarahNet broker, store-and-forward, node trust, and XR contract readiness.",
    sourceFile: "src/components/screens/SarahNetScreen.tsx",
    component: <SarahNetScreen />,
  },
  {
    id: "media",
    title: "Media",
    area: "media",
    purpose: "Media player/library panel.",
    sourceFile: "src/components/screens/MediaScreen.tsx",
    component: <MediaScreen />,
  },
  {
    id: "dlengine",
    title: "DL Engine",
    area: "operator",
    purpose: "DL runtime overview, REM, model governance weights, jobs, and traces.",
    sourceFile: "src/components/screens/DLEngineScreen.tsx",
    component: <DLEngineScreen />,
  },
  {
    id: "nailde",
    title: "NAILDE",
    area: "operator",
    purpose: "VS Code plus VB6-style governed development workbench.",
    sourceFile: "src/components/screens/NAILDEScreen.tsx",
    component: <NAILDEScreen />,
  },
  {
    id: "terminal",
    title: "Terminal",
    area: "operator",
    purpose: "Governed terminal request surface.",
    sourceFile: "src/components/screens/TerminalScreen.tsx",
    component: <TerminalScreen />,
  },
  {
    id: "addons",
    title: "Addons",
    area: "creation",
    purpose: "Addon registry and sandbox install visibility.",
    sourceFile: "src/components/screens/AddonsScreen.tsx",
    component: <AddonsScreen />,
  },
  {
    id: "settings",
    title: "Settings",
    area: "settings",
    purpose: "Runtime, theme, voice, model, device, and advanced configuration.",
    sourceFile: "src/components/screens/SettingsScreen.tsx",
    component: <SettingsScreen />,
  },
];

export const SHELL_FEATURE_BY_ID: Record<WindowId, ShellFeatureDefinition> = SHELL_FEATURES.reduce(
  (acc, feature) => {
    acc[feature.id] = feature;
    return acc;
  },
  {} as Record<WindowId, ShellFeatureDefinition>,
);

export function normalizeFeatureId(id: ShellFeatureId | string): WindowId {
  if (id === "studios") return "studio";
  if (id in SHELL_FEATURE_BY_ID) return id as WindowId;
  return "chat";
}

export function getFeatureComponent(id: ShellFeatureId | string): JSX.Element {
  const normalized = normalizeFeatureId(id);
  const feature = SHELL_FEATURE_BY_ID[normalized];
  if (!feature || normalized === "chat") return feature?.component ?? <ChatPanel />;
  return (
    <PanelChatBridge
      featureId={feature.id}
      title={feature.title}
      purpose={feature.purpose}
    >
      {feature.component}
    </PanelChatBridge>
  );
}
