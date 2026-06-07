import React, { useState } from "react";
import {
  Sheet,
  SheetTrigger,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
  SheetClose,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { LayoutGrid, Laptop2, Monitor, User2 } from "lucide-react";
import { useSarahStore } from "@/stores/useSarahStore";
import { cn } from "@/lib/utils";

/**
 * ControlCenter
 *
 * A slide‑out panel providing quick access to global toggles and mode selectors
 * for the SarahMemory WebUI.  This component surfaces UI mode selection
 * (simple/operator/engineer), an offline/local only toggle and quick media
 * enable/disable controls (camera, microphone, voice).  It is implemented as
 * a Radix Sheet anchored to the right side of the viewport and can be
 * triggered via a button placed in the StatusBar.
 */
export function ControlCenter() {
  const {
    settings,
    updateSettings,
    mediaState,
    toggleWebcam,
    toggleMicrophone,
    toggleVoice,
  } = useSarahStore();

  const uiMode: any = (settings as any)?.uiMode || 'simple';
  const localOnly: boolean = Boolean((settings as any)?.localOnlyMode);

  const setUiMode = (mode: 'simple' | 'operator' | 'engineer') => {
    updateSettings({ uiMode: mode });
  };
  const toggleLocalOnly = () => {
    updateSettings({ localOnlyMode: !localOnly });
  };

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
          title="Control Center"
          aria-label="Control Center"
        >
          <LayoutGrid className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-80 sm:w-96">
        <SheetHeader>
          <SheetTitle>Control Center</SheetTitle>
          <SheetDescription>
            Manage your SarahMemory operating environment.
          </SheetDescription>
        </SheetHeader>
        <div className="py-4 space-y-6 overflow-y-auto">
          {/* UI Mode selector */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">UI Mode</h3>
            <div className="flex gap-2">
              {(['simple', 'operator', 'engineer'] as const).map((mode) => {
                const isActive = uiMode === mode;
                return (
                  <Button
                    key={mode}
                    variant={isActive ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1 capitalize"
                    onClick={() => setUiMode(mode)}
                  >
                    {mode}
                  </Button>
                );
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              Simple is streamlined; Operator enables SarahNet/Media; Engineer enables all panels including the developer terminal and DL Engine.
            </p>
          </div>

          {/* Local Only Mode toggle */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">Offline / Local Only</h3>
            <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
              <span className="text-sm">Local Only Mode</span>
              <Switch checked={localOnly} onCheckedChange={toggleLocalOnly} />
            </div>
            <p className="text-xs text-muted-foreground">
              When enabled, SarahMemory will avoid remote API calls and operate in a self‑contained offline mode.
            </p>
          </div>

          {/* Media controls */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">Media</h3>
            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
                <span className="flex items-center gap-2 text-sm">
                  <Laptop2 className="h-4 w-4" /> Webcam
                </span>
                <Switch checked={mediaState.webcamEnabled} onCheckedChange={toggleWebcam} />
              </div>
              <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
                <span className="flex items-center gap-2 text-sm">
                  <Monitor className="h-4 w-4" /> Microphone
                </span>
                <Switch checked={mediaState.microphoneEnabled} onCheckedChange={toggleMicrophone} />
              </div>
              <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
                <span className="flex items-center gap-2 text-sm">
                  <User2 className="h-4 w-4" /> Voice Output
                </span>
                <Switch checked={mediaState.voiceEnabled} onCheckedChange={toggleVoice} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Quickly enable or disable your camera, microphone, and voice output.
            </p>
          </div>
        </div>
        <SheetFooter className="pt-4">
          <SheetClose asChild>
            <Button variant="secondary">Close</Button>
          </SheetClose>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}