import { useEffect, useMemo, useState } from "react";
import { CalendarClock, Clock, Save, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useSarahStore } from "@/stores/useSarahStore";

const COMMON_TIMEZONES = [
  "America/Chicago",
  "America/New_York",
  "America/Denver",
  "America/Los_Angeles",
  "America/Phoenix",
  "America/Anchorage",
  "Pacific/Honolulu",
  "UTC",
  "Europe/London",
  "Europe/Paris",
  "Asia/Tokyo",
  "Australia/Sydney",
] as const;

function pad(value: number) {
  return String(value).padStart(2, "0");
}

function dateValue(now: Date) {
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

function timeValue(now: Date) {
  return `${pad(now.getHours())}:${pad(now.getMinutes())}`;
}

function browserTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
}

function normalizeTimezone(value: unknown) {
  const text = String(value || "").trim();
  return text && text !== "local" ? text : browserTimezone();
}

function getTimezones(currentTimezone: string) {
  const supported = typeof (Intl as any).supportedValuesOf === "function"
    ? ((Intl as any).supportedValuesOf("timeZone") as string[])
    : [];
  return Array.from(new Set([currentTimezone, ...COMMON_TIMEZONES, ...supported])).filter(Boolean).sort((a, b) => a.localeCompare(b));
}

function formatInTimezone(now: Date, timezone: string) {
  try {
    return now.toLocaleString([], {
      timeZone: timezone === "UTC" ? "UTC" : timezone,
      weekday: "short",
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return "Timezone preview unavailable.";
  }
}

export function SystemClockPanel() {
  const { settings, updateSettings } = useSarahStore();
  const [now, setNow] = useState(() => new Date());
  const [date, setDate] = useState(() => String((settings as any).manualClockDate || dateValue(new Date())));
  const [time, setTime] = useState(() => String((settings as any).manualClockTime || timeValue(new Date())));
  const [timezone, setTimezone] = useState(() => normalizeTimezone((settings as any).manualClockTimezone));
  const [authority, setAuthority] = useState<any>(null);
  const [status, setStatus] = useState("Clock Court not queried yet.");
  const timezoneOptions = useMemo(() => getTimezones(timezone), [timezone]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const result = await api.proxy.call("/api/system/time-authority?claim=what%20time%20is%20it", { method: "GET" });
      if (cancelled) return;
      setAuthority(result);
      setStatus((result as any)?.ok ? "Clock Court read authority online." : "Clock Court bridge unavailable; local browser clock shown.");
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const localPreview = useMemo(() => {
    return now.toLocaleString([], {
      weekday: "short",
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }, [now]);

  const timezonePreview = useMemo(() => formatInTimezone(now, timezone), [now, timezone]);

  const syncFromLocalClock = () => {
    const current = new Date();
    setDate(dateValue(current));
    setTime(timeValue(current));
    setTimezone(browserTimezone());
  };

  const saveClockContext = async () => {
    updateSettings({
      manualClockDate: date,
      manualClockTime: time,
      manualClockTimezone: timezone,
    } as any);

    await Promise.all([
      api.settings.setSetting("manualClockDate", date),
      api.settings.setSetting("manualClockTime", time),
      api.settings.setSetting("manualClockTimezone", timezone),
    ]);

    const result = await api.proxy.call("/api/system/clock-court", {
      method: "POST",
      body: {
        claim: "operator requested clock/date context adjustment",
        meta: {
          source: "frontend:system_clock_panel",
          requested_date: date,
          requested_time: time,
          requested_timezone: timezone,
        },
      },
    });
    setAuthority(result);
    setStatus((result as any)?.ok ? "Clock context saved and Clock Court queried." : "Clock context saved locally; backend clock mutation route is pending.");
  };

  return (
    <div className="w-[min(calc(100vw-1rem),380px)] max-w-full space-y-3 overflow-hidden text-sm">
      <div className="flex items-center gap-2">
        <CalendarClock className="h-4 w-4 text-primary" />
        <div className="min-w-0">
          <div className="font-semibold">System Clock</div>
          <div className="truncate text-xs text-muted-foreground">{localPreview}</div>
        </div>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
        <label className="min-w-0 space-y-1 text-xs">
          <span className="text-muted-foreground">Date</span>
          <Input className="min-w-0" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <label className="min-w-0 space-y-1 text-xs">
          <span className="text-muted-foreground">Time</span>
          <Input className="min-w-0" type="time" value={time} onChange={(e) => setTime(e.target.value)} />
        </label>
      </div>

      <label className="block min-w-0 space-y-1 text-xs">
        <span className="text-muted-foreground">Timezone / Locality Context</span>
        <Select value={timezone} onValueChange={setTimezone}>
          <SelectTrigger className="min-w-0 bg-background/90">
            <SelectValue placeholder="Select timezone" />
          </SelectTrigger>
          <SelectContent className="z-[100000] max-h-72">
            {timezoneOptions.map((tz) => (
              <SelectItem key={tz} value={tz}>
                {tz}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>

      <div className="rounded-md border border-border/70 bg-secondary/25 p-2 text-xs">
        <div className="text-muted-foreground">Selected timezone preview</div>
        <div className="mt-1 truncate font-medium">{timezonePreview}</div>
      </div>

      <div className="flex gap-2">
        <Button type="button" variant="outline" size="sm" className="flex-1 gap-2" onClick={syncFromLocalClock}>
          <Clock className="h-3.5 w-3.5" />
          Use Local
        </Button>
        <Button type="button" size="sm" className="flex-1 gap-2" onClick={() => void saveClockContext()}>
          <Save className="h-3.5 w-3.5" />
          Save Context
        </Button>
      </div>

      <div className="max-w-full overflow-hidden rounded-lg border border-border/70 bg-card/70 p-2 text-xs text-muted-foreground">
        <div className="mb-1 flex items-center gap-1.5 font-medium text-foreground">
          <ShieldCheck className="h-3.5 w-3.5 text-primary" />
          Clock Authority
        </div>
        <div>{status}</div>
        {authority ? <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words text-[10px]">{JSON.stringify(authority, null, 2).slice(0, 900)}</pre> : null}
      </div>
    </div>
  );
}
