import type { WindowId } from "@/stores/useWindowStore";
import type { MobileScreen } from "@/stores/useNavigationStore";

import { ChatPanel } from "@/components/chat/ChatPanel";
import { HistoryScreen } from "@/components/screens/history/HistoryScreen";
import { FilesScreen } from "@/components/screens/files/FilesScreen";
import { ResearchScreen } from "@/components/screens/research/ResearchScreen";
import { StudiosScreen } from "@/components/screens/studios/StudiosScreen";
import { AvatarScreen } from "@/components/screens/avatar/AvatarScreen";
import { SarahNetScreen } from "@/components/screens/sarah-net/SarahNetScreen";
import { MediaScreen } from "@/components/screens/media/MediaScreen";
import { DLEngineScreen } from "@/components/screens/dl-engine/DLEngineScreen";
import { SettingsScreen } from "@/components/screens/settings/SettingsScreen";
import { AddonsScreen } from "@/components/screens/addons/AddonsScreen";
import NAILDEScreen from "@/components/screens/nailde/NAILDEScreen";
import TerminalScreen from "@/components/screens/terminal/TerminalScreen";
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
    title: "Sarah Chat",
    area: "conversation",
    purpose: "Main Sarah conversation, backend chat calls, avatar speaking events, file ingestion.",
    sourceFile: "src/components/chat/ChatPanel.tsx",
    component: <ChatPanel />,
  },
  {
    id: "history",
    title: "Memory Trail",
    area: "conversation",
    purpose: "Conversation history and session recall surface.",
    sourceFile: "src/components/screens/history/HistoryScreen.tsx",
    component: <HistoryScreen />,
  },
  {
    id: "files",
    title: "File Cortex",
    area: "files",
    purpose: "Local-first file browser and file action UI.",
    sourceFile: "src/components/screens/files/FilesScreen.tsx",
    component: <FilesScreen />,
  },
  {
    id: "research",
    title: "Evidence Lens",
    area: "research",
    purpose: "Research lane, web/local evidence gathering, and source review.",
    sourceFile: "src/components/screens/research/ResearchScreen.tsx",
    component: <ResearchScreen />,
  },
  {
    id: "studio",
    aliases: ["studios"],
    title: "Creation Bay",
    area: "creation",
    purpose: "Creative media generation modules and studio tabs.",
    sourceFile: "src/components/screens/studios/StudiosScreen.tsx",
    component: <StudiosScreen />,
  },
  {
    id: "avatar",
    title: "Avatar Core",
    area: "assistant",
    purpose: "2D/3D avatar, local camera overlay, voice state, mirror/media modes.",
    sourceFile: "src/components/screens/avatar/AvatarScreen.tsx",
    component: <AvatarScreen />,
  },
  {
    id: "sarahnet",
    title: "SarahNet",
    area: "network",
    purpose: "SarahNet broker, store-and-forward, node trust, and XR contract readiness.",
    sourceFile: "src/components/screens/sarah-net/SarahNetScreen.tsx",
    component: <SarahNetScreen />,
  },
  {
    id: "media",
    title: "Media Deck",
    area: "media",
    purpose: "Media player/library panel.",
    sourceFile: "src/components/screens/media/MediaScreen.tsx",
    component: <MediaScreen />,
  },
  {
    id: "dlengine",
    title: "Model Forge",
    area: "operator",
    purpose: "DL runtime overview, REM, model governance weights, jobs, and traces.",
    sourceFile: "src/components/screens/dl-engine/DLEngineScreen.tsx",
    component: <DLEngineScreen />,
  },
  {
    id: "nailde",
    title: "NAILDE",
    area: "operator",
    purpose: "VS Code plus VB6-style governed development workbench.",
    sourceFile: "src/components/screens/nailde/NAILDEScreen.tsx",
    component: <NAILDEScreen />,
  },
  {
    id: "terminal",
    title: "Operator Terminal",
    area: "operator",
    purpose: "Governed terminal request surface.",
    sourceFile: "src/components/screens/terminal/TerminalScreen.tsx",
    component: <TerminalScreen />,
  },
  {
    id: "addons",
    title: "Addons",
    area: "creation",
    purpose: "Addon registry and sandbox install visibility.",
    sourceFile: "src/components/screens/addons/AddonsScreen.tsx",
    component: <AddonsScreen />,
  },
  {
    id: "settings",
    title: "System Tuning",
    area: "settings",
    purpose: "Runtime, theme, voice, model, device, and advanced configuration.",
    sourceFile: "src/components/screens/settings/SettingsScreen.tsx",
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
