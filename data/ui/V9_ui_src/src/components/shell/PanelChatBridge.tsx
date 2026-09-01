import type { ReactNode } from "react";

type Props = {
  featureId: string;
  title: string;
  purpose: string;
  children: ReactNode;
};

export function PanelChatBridge({ featureId, title, purpose, children }: Props) {
  return (
    <div
      className="h-full min-h-0"
      data-panel-chat-context="true"
      data-feature-id={featureId}
      data-feature-title={title}
      data-feature-purpose={purpose}
    >
      {children}
    </div>
  );
}
