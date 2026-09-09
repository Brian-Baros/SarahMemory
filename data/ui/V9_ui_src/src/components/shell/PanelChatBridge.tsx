import { useCallback, useEffect, type ReactNode } from "react";
import { useWindowStore } from "@/stores/useWindowStore";

type Props = {
  featureId: string;
  title: string;
  purpose: string;
  children: ReactNode;
};

export function PanelChatBridge({ featureId, title, purpose, children }: Props) {
  const { openWindow } = useWindowStore();

  const sendToChat = useCallback(() => {
    const draft = `[Panel context: ${title}]\n${purpose}\n\nHelp me with: `;
    try {
      window.sessionStorage.setItem("sarah:chat-pending-draft", draft);
    } catch {
      // non-fatal panel context handoff
    }
    window.dispatchEvent(
      new CustomEvent("sarah:chat-draft", {
        detail: { draft, featureId, title, purpose, ts: Date.now(), source: "PanelChatBridge" },
      }),
    );
    openWindow("chat");
  }, [featureId, openWindow, purpose, title]);

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ featureId?: string }>).detail || {};
      if (!detail.featureId || detail.featureId === featureId) sendToChat();
    };
    window.addEventListener("sarah:panel-ask", handler);
    return () => window.removeEventListener("sarah:panel-ask", handler);
  }, [featureId, sendToChat]);

  return (
    <div
      className="h-full min-h-0"
      data-panel-chat-context="true"
      data-feature-id={featureId}
      data-feature-title={title}
      data-feature-purpose={purpose}
      data-chat-draft-event="sarah:chat-draft"
    >
      {children}
    </div>
  );
}
