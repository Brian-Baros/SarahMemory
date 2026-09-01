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
 * Command gate panel for global shell posture, locality, and media switches.
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
          title="Command Gates"
          aria-label="Command Gates"
        >
          <LayoutGrid className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-80 sm:w-96">
        <SheetHeader>
          <SheetTitle>Command Gates</SheetTitle>
          <SheetDescription>
            Manage shell posture, locality, and media intent.
          </SheetDescription>
        </SheetHeader>
        <div className="py-4 space-y-6 overflow-y-auto">
          {/* UI Mode selector */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">Shell Posture</h3>
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
                    {mode === "simple" ? "core" : mode === "engineer" ? "builder" : mode}
                  </Button>
                );
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              Core is streamlined; Operator enables network/media surfaces; Builder exposes governed terminal and model tooling.
            </p>
          </div>

          {/* Local Only Mode toggle */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">Local-First Lock</h3>
            <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
              <span className="text-sm">Keep execution local where possible</span>
              <Switch checked={localOnly} onCheckedChange={toggleLocalOnly} />
            </div>
            <p className="text-xs text-muted-foreground">
              When enabled, SarahMemory should prefer self-contained local lanes before remote services.
            </p>
          </div>

          {/* Media controls */}
          <div className="space-y-2">
            <h3 className="text-base font-medium">Embodiment Gates</h3>
            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
                <span className="flex items-center gap-2 text-sm">
                  <Laptop2 className="h-4 w-4" /> Camera Gate
                </span>
                <Switch checked={mediaState.webcamEnabled} onCheckedChange={toggleWebcam} />
              </div>
              <div className="flex items-center justify-between px-1 py-2 rounded-md bg-secondary/40">
                <span className="flex items-center gap-2 text-sm">
                  <Monitor className="h-4 w-4" /> Microphone Gate
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
              Quickly set camera, microphone, and voice-output intent without bypassing backend governance.
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
