import type { ReactNode } from "react";
import { MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWindowStore } from "@/stores/useWindowStore";
import { useNavigationStore } from "@/stores/useNavigationStore";

type Props = {
  featureId: string;
  title: string;
  purpose: string;
  children: ReactNode;
};

export function PanelChatBridge({ featureId, title, purpose, children }: Props) {
  const openWindow = useWindowStore((state) => state.openWindow);
  const setCurrentScreen = useNavigationStore((state) => state.setCurrentScreen);

  const sendToChat = () => {
    const draft = `[Panel context: ${title}]\n${purpose}\n\nHelp me with: `;
    try { window.sessionStorage.setItem("sarah:chat-pending-draft", draft); } catch { /* non-fatal */ }
    window.dispatchEvent(
      new CustomEvent("sarah:chat-draft", {
        detail: { draft, featureId, title, purpose, ts: Date.now() },
      }),
    );

    if (window.matchMedia("(max-width: 767px)").matches) {
      setCurrentScreen("chat");
    } else {
      openWindow("chat");
    }
  };

  return (
    <div className="relative h-full min-h-0">
      <div className="absolute right-3 top-3 z-40">
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className="h-8 gap-1.5 border border-border/80 bg-background/90 shadow-sm backdrop-blur"
          onClick={sendToChat}
          title={`Send ${title} context to Chat`}
        >
          <MessageCircle className="h-3.5 w-3.5" />
          Ask Sarah
        </Button>
      </div>
      <div className="h-full min-h-0">{children}</div>
    </div>
  );
}
