import { useWindowStore } from "@/stores/useWindowStore";
import { Window } from "./Window";
import { getFeatureComponent } from "@/features/featureRegistry";

export function WindowManager() {
  const { windows } = useWindowStore();

  const workspaceStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    overflow: "hidden",
  };

  return (
    <div style={workspaceStyle}>
      {windows.map((w) => (
        <Window key={w.id} window={w}>
          {getFeatureComponent(w.id)}
        </Window>
      ))}
    </div>
  );
}
