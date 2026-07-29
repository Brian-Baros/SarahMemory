import * as React from "react";

const MOBILE_BREAKPOINT = 768;
const TABLET_PORTRAIT_BREAKPOINT = 1024;

type ShellMode = "mobile" | "desktop";

export type ViewportProfile = {
  width: number;
  height: number;
  isPortrait: boolean;
  isLandscape: boolean;
  isPhoneWidth: boolean;
  isTabletPortrait: boolean;
  isTouch: boolean;
  shellMode: ShellMode;
};

function canUseDom() {
  return typeof window !== "undefined" && typeof document !== "undefined";
}

function getViewportProfile(): ViewportProfile {
  if (!canUseDom()) {
    return {
      width: 1024,
      height: 768,
      isPortrait: false,
      isLandscape: true,
      isPhoneWidth: false,
      isTabletPortrait: false,
      isTouch: false,
      shellMode: "desktop",
    };
  }

  const visualViewport = window.visualViewport;
  const width = Math.max(1, Math.round(visualViewport?.width ?? window.innerWidth ?? 1024));
  const height = Math.max(1, Math.round(visualViewport?.height ?? window.innerHeight ?? 768));
  const isPortrait = height >= width;
  const isLandscape = !isPortrait;
  const isPhoneWidth = width < MOBILE_BREAKPOINT;
  const isTabletPortrait = isPortrait && width < TABLET_PORTRAIT_BREAKPOINT;
  const isTouch =
    typeof navigator !== "undefined" &&
    (navigator.maxTouchPoints > 0 || window.matchMedia?.("(hover: none) and (pointer: coarse)").matches === true);

  /**
   * Single SarahMemory viewport contract:
   * - Portrait phone/tablet uses the mobile shell.
   * - Landscape phone/tablet and desktop use the desktop shell.
   *
   * This keeps horizontal device posture aligned with the desktop-style shell
   * while keeping vertical phone posture aligned with the mobile shell.
   */
  const shellMode: ShellMode = isPortrait && (isPhoneWidth || isTabletPortrait) ? "mobile" : "desktop";

  return {
    width,
    height,
    isPortrait,
    isLandscape,
    isPhoneWidth,
    isTabletPortrait,
    isTouch,
    shellMode,
  };
}

function subscribeViewport(callback: () => void) {
  if (!canUseDom()) return () => {};

  let frame = 0;
  const notify = () => {
    window.cancelAnimationFrame(frame);
    frame = window.requestAnimationFrame(callback);
  };

  window.addEventListener("resize", notify, { passive: true });
  window.addEventListener("orientationchange", notify, { passive: true });
  window.visualViewport?.addEventListener("resize", notify, { passive: true });

  const orientationQuery = window.matchMedia?.("(orientation: portrait)");
  try {
    orientationQuery?.addEventListener?.("change", notify);
  } catch {
    orientationQuery?.addListener?.(notify);
  }

  return () => {
    window.cancelAnimationFrame(frame);
    window.removeEventListener("resize", notify);
    window.removeEventListener("orientationchange", notify);
    window.visualViewport?.removeEventListener("resize", notify);
    try {
      orientationQuery?.removeEventListener?.("change", notify);
    } catch {
      orientationQuery?.removeListener?.(notify);
    }
  };
}

export function useViewportProfile(): ViewportProfile {
  const [profile, setProfile] = React.useState<ViewportProfile>(() => getViewportProfile());

  React.useEffect(() => {
    if (!canUseDom()) return undefined;

    const update = () => setProfile(getViewportProfile());
    update();
    return subscribeViewport(update);
  }, []);

  return profile;
}

export function useIsMobile() {
  return useViewportProfile().shellMode === "mobile";
}

export function useIsDesktopShell() {
  return useViewportProfile().shellMode === "desktop";
}
