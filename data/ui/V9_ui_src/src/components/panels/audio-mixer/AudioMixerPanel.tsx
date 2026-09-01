import { Headphones, Mic, SlidersHorizontal, Volume2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useSarahStore } from "@/stores/useSarahStore";
import { useWindowStore } from "@/stores/useWindowStore";

function clampNumber(value: any, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.round(parsed)));
}

function AudioSlider(props: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  const min = props.min ?? 0;
  const max = props.max ?? 100;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-medium text-foreground/85">{props.label}</span>
        <span className="tabular-nums text-muted-foreground">
          {props.value}
          {props.suffix ?? "%"}
        </span>
      </div>
      <Slider
        value={[props.value]}
        min={min}
        max={max}
        step={1}
        onValueChange={(value) => props.onChange(value[0] ?? props.value)}
        aria-label={props.label}
      />
    </div>
  );
}

function announceAudioChange(key: string, value: unknown) {
  try {
    window.dispatchEvent(
      new CustomEvent("sarah:audio", {
        detail: {
          source: "taskbar-audio-mixer",
          settings: { [key]: value },
        },
      }),
    );
  } catch {
    // local UI event only
  }
}

export function AudioMixerPanel({ showSettingsButton = true }: { showSettingsButton?: boolean }) {
  const { settings, updateSettings, mediaState, toggleMicrophone, toggleVoice } = useSarahStore();
  const { openWindow } = useWindowStore();

  const masterVolume = clampNumber((settings as any).masterVolume, 78, 0, 100);
  const outputVolume = clampNumber((settings as any).outputVolume, 82, 0, 100);
  const inputVolume = clampNumber((settings as any).inputVolume, 70, 0, 100);
  const bassLevel = clampNumber((settings as any).bassLevel, 0, -12, 12);
  const trebleLevel = clampNumber((settings as any).trebleLevel, 0, -12, 12);
  const balance = clampNumber((settings as any).balance, 0, -50, 50);
  const spatialAudio = Boolean((settings as any).spatialAudio);
  const noiseSuppression = Boolean((settings as any).noiseSuppression);

  const setNumber = (key: string, value: number) => {
    updateSettings({ [key]: value } as any);
    announceAudioChange(key, value);
  };

  const setBoolean = (key: string, value: boolean) => {
    updateSettings({ [key]: value } as any);
    announceAudioChange(key, value);
  };

  return (
    <div className="w-[min(92vw,380px)] space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Volume2 className="h-4 w-4 text-primary" />
            Audio Mixer
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            Frontend mixer ready for driver/AppSys binding.
          </div>
        </div>
        {showSettingsButton && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => openWindow("settings")}
          >
            Settings
          </Button>
        )}
      </div>

      <div className="grid gap-3 rounded-lg border border-border/70 bg-background/50 p-3">
        <AudioSlider label="Master volume" value={masterVolume} onChange={(value) => setNumber("masterVolume", value)} />
        <AudioSlider label="Output volume" value={outputVolume} onChange={(value) => setNumber("outputVolume", value)} />
        <AudioSlider label="Microphone gain" value={inputVolume} onChange={(value) => setNumber("inputVolume", value)} />
      </div>

      <div className="grid gap-3 rounded-lg border border-border/70 bg-background/50 p-3">
        <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <SlidersHorizontal className="h-4 w-4" />
          Equalizer
        </div>
        <AudioSlider label="Bass" value={bassLevel} min={-12} max={12} suffix=" dB" onChange={(value) => setNumber("bassLevel", value)} />
        <AudioSlider label="Treble" value={trebleLevel} min={-12} max={12} suffix=" dB" onChange={(value) => setNumber("trebleLevel", value)} />
        <AudioSlider label="Balance" value={balance} min={-50} max={50} suffix="" onChange={(value) => setNumber("balance", value)} />
      </div>

      <div className="grid gap-2 rounded-lg border border-border/70 bg-background/50 p-3">
        <label className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-sm">
            <Volume2 className="h-4 w-4 text-muted-foreground" />
            Voice output
          </span>
          <Switch checked={mediaState.voiceEnabled} onCheckedChange={toggleVoice} />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-sm">
            <Mic className="h-4 w-4 text-muted-foreground" />
            Microphone input
          </span>
          <Switch checked={mediaState.microphoneEnabled} onCheckedChange={toggleMicrophone} />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-sm">
            <Headphones className="h-4 w-4 text-muted-foreground" />
            Spatial audio
          </span>
          <Switch checked={spatialAudio} onCheckedChange={(checked) => setBoolean("spatialAudio", checked)} />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-sm">
            <Mic className="h-4 w-4 text-muted-foreground" />
            Noise suppression
          </span>
          <Switch checked={noiseSuppression} onCheckedChange={(checked) => setBoolean("noiseSuppression", checked)} />
        </label>
      </div>

      <div className="rounded-lg border border-border/60 bg-muted/20 p-2 text-[11px] text-muted-foreground">
        Driver binding event: <span className="font-mono">sarah:audio</span>. Backend driver/AppSys
        routing can subscribe without changing this panel contract.
      </div>
    </div>
  );
}
