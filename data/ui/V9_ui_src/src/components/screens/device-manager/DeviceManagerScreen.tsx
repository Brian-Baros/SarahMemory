import { useEffect, useMemo, useState } from "react";
import {
  Camera,
  Cpu,
  HardDrive,
  Keyboard,
  Loader2,
  MonitorCog,
  PlugZap,
  Printer,
  RefreshCw,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Speaker,
  Wifi,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

type DeviceTab = "all" | "network" | "audio" | "printers" | "storage" | "input" | "camera" | "other";

type DriverItem = {
  id: string;
  manifest?: Record<string, any>;
  enabled?: boolean;
  autoload?: boolean;
  trusted?: boolean;
  connected?: boolean;
  instance_id?: string;
  level?: string | number;
  dependencies?: string[];
  bridgeDriverId?: string;
  source?: string;
  sourceLabel?: string;
  deviceClass?: string;
  status?: string;
  raw?: Record<string, any>;
};

type InventorySource = {
  label: string;
  ok: boolean;
  detail: string;
};

const TAB_META: Record<DeviceTab, { label: string; icon: any; needles: string[] }> = {
  all: { label: "All", icon: MonitorCog, needles: [] },
  network: { label: "Networks", icon: Wifi, needles: ["network", "wifi", "wi-fi", "wireless", "ethernet", "tcp", "vpn"] },
  audio: { label: "Audio", icon: Speaker, needles: ["audio", "sound", "speaker", "microphone", "voice"] },
  printers: { label: "Printers", icon: Printer, needles: ["printer", "print", "scanner"] },
  storage: { label: "Storage", icon: HardDrive, needles: ["storage", "disk", "drive", "nvme", "sata", "usb"] },
  input: { label: "Input", icon: Keyboard, needles: ["keyboard", "mouse", "touch", "gamepad", "controller", "input"] },
  camera: { label: "Cameras", icon: Camera, needles: ["camera", "webcam", "vision", "uvc"] },
  other: { label: "Other", icon: Cpu, needles: [] },
};

const CONFIG_FIELDS = [
  ["network_name", "Network Name"],
  ["ssid", "SSID"],
  ["security_mode", "Security Mode"],
  ["tcp_ip", "TCP/IP"],
  ["subnet_mask", "Subnet Mask"],
  ["gateway", "Gateway"],
  ["dns_servers", "DNS Servers"],
  ["vpn_profile", "VPN Profile"],
  ["password", "Password"],
  ["notes", "Notes"],
] as const;

const INVENTORY_ENDPOINTS = [
  { label: "Driver Bridge", route: "/api/drivers", source: "driver-bridge" },
  { label: "Manifest Audit", route: "/api/drivers/manifest/audit", source: "driver-audit" },
  { label: "Hardware Topology", route: "/api/self/hardware-topology", source: "hardware-topology" },
  { label: "Body Map", route: "/api/self/body", source: "body-map" },
  { label: "Body Capabilities", route: "/api/self/body-capabilities", source: "body-capabilities" },
  { label: "Vision Devices", route: "/api/vision/devices", source: "vision-devices" },
] as const;

function flattenDeviceText(driver: DriverItem) {
  const manifest = driver.manifest || {};
  return [
    driver.id,
    manifest.id,
    manifest.name,
    manifest.label,
    manifest.family,
    manifest.device_class,
    manifest.description,
    Array.isArray(manifest.families) ? manifest.families.join(" ") : "",
    Array.isArray(manifest.tags) ? manifest.tags.join(" ") : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function classifyDevice(driver: DriverItem): DeviceTab {
  const text = flattenDeviceText(driver);
  for (const key of ["network", "audio", "printers", "storage", "input", "camera"] as DeviceTab[]) {
    if (TAB_META[key].needles.some((needle) => text.includes(needle))) return key;
  }
  return "other";
}

function displayName(driver: DriverItem) {
  return String(driver.manifest?.name || driver.manifest?.label || driver.id);
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function safeString(value: unknown, fallback = "") {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  return text || fallback;
}

function slug(value: unknown) {
  return safeString(value, "device")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64) || "device";
}

function boolValue(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
      const v = value.trim().toLowerCase();
      if (["true", "yes", "on", "enabled", "online", "connected", "ready", "ok"].includes(v)) return true;
      if (["false", "no", "off", "disabled", "offline", "blocked", "error"].includes(v)) return false;
    }
  }
  return false;
}

function categoryFromText(value: unknown): DeviceTab {
  const text = safeString(value).toLowerCase();
  for (const key of ["network", "audio", "printers", "storage", "input", "camera"] as DeviceTab[]) {
    if (TAB_META[key].needles.some((needle) => text.includes(needle))) return key;
  }
  return "other";
}

function normalizeDevice(rawValue: unknown, source: string, label: string, index: number, fallbackCategory?: DeviceTab): DriverItem | null {
  const raw = isRecord(rawValue) ? rawValue : { name: safeString(rawValue, `${label} device`) };
  const manifest = isRecord(raw.manifest)
    ? { ...raw.manifest }
    : isRecord(raw.driver?.manifest)
      ? { ...raw.driver.manifest }
      : isRecord(raw.metadata)
        ? { ...raw.metadata }
        : {};

  const id = safeString(
    raw.id || raw.driver_id || raw.device_id || raw.instance_id || raw.DeviceID || raw.guid || manifest.id || manifest.name || raw.name || raw.label,
    `${slug(source)}-${index}`,
  );
  const name = safeString(raw.name || raw.label || raw.Name || raw.adapter || raw.interface || raw.device || raw.description || manifest.name || manifest.label, id);
  const family = safeString(raw.family || raw.kind || raw.type || raw.device_class || raw.class || manifest.family || manifest.device_class || fallbackCategory, "");
  const category = fallbackCategory || categoryFromText(`${id} ${name} ${family} ${safeString(raw.description)} ${safeString(manifest.description)}`);

  return {
    id: source === "driver-audit" && raw.driver_id ? String(raw.driver_id) : id,
    bridgeDriverId: source === "driver-bridge" || source === "driver-audit" ? safeString(raw.driver_id || id) : undefined,
    manifest: {
      ...manifest,
      id: manifest.id || id,
      name: manifest.name || manifest.label || name,
      label: manifest.label || name,
      family: manifest.family || family || category,
      device_class: manifest.device_class || raw.device_class || category,
      description: manifest.description || raw.description || raw.caption || `${label} inventory entry`,
    },
    enabled: isRecord(raw.registry) && raw.registry.enabled !== undefined ? boolValue(raw.registry.enabled) : boolValue(raw.enabled, raw.active, raw.present, raw.ok, true),
    autoload: boolValue(raw.autoload, raw.registry?.autoload),
    trusted: boolValue(raw.trusted, raw.registry?.trusted),
    connected: boolValue(raw.connected, raw.online, raw.present, raw.active, raw.session),
    instance_id: safeString(raw.instance_id || raw.device_id || raw.DeviceID || raw.guid, ""),
    level: raw.level || raw.registry?.level || manifest.level,
    dependencies: Array.isArray(raw.dependencies) ? raw.dependencies : Array.isArray(raw.registry?.dependencies) ? raw.registry.dependencies : [],
    source,
    sourceLabel: label,
    deviceClass: category,
    status: safeString(raw.status || raw.state || raw.health || raw.decision, ""),
    raw,
  };
}

function extractDeviceCandidates(payload: unknown, source: string, label: string): DriverItem[] {
  const out: DriverItem[] = [];
  const add = (value: unknown, fallbackCategory?: DeviceTab) => {
    const item = normalizeDevice(value, source, label, out.length, fallbackCategory);
    if (item) out.push(item);
  };

  if (Array.isArray(payload)) {
    payload.forEach((item) => add(item));
    return out;
  }
  if (!isRecord(payload)) return out;

  const directKeys = ["drivers", "devices", "items", "adapters", "network_adapters", "storage_devices", "camera_devices", "printers", "audio_devices", "input_devices"];
  for (const key of directKeys) {
    const value = payload[key];
    if (Array.isArray(value)) value.forEach((item) => add(item, categoryFromText(key)));
  }

  const bodyMap = isRecord(payload.body_map) ? payload.body_map : {};
  for (const key of directKeys) {
    const value = bodyMap[key];
    if (Array.isArray(value)) value.forEach((item) => add(item, categoryFromText(key)));
  }

  if (out.length === 0 && source === "hardware-topology") {
    Object.entries(payload).forEach(([key, value]) => {
      if (isRecord(value) && /(network|storage|usb|camera|audio|printer|display|gpu|cpu|memory|bluetooth|ethernet|wifi)/i.test(key)) {
        add({ ...value, id: key, name: key.replace(/_/g, " ") }, categoryFromText(key));
      }
    });
  }

  return out;
}

async function collectBrowserDevices(): Promise<DriverItem[]> {
  const items: DriverItem[] = [];
  const nav = typeof navigator !== "undefined" ? (navigator as any) : {};

  try {
    if (navigator.mediaDevices?.enumerateDevices) {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      mediaDevices.forEach((device, index) => {
        const category: DeviceTab = device.kind === "videoinput" ? "camera" : device.kind === "audioinput" || device.kind === "audiooutput" ? "audio" : "input";
        items.push({
          id: `browser-media-${slug(device.kind)}-${slug(device.deviceId || index)}`,
          manifest: {
            id: device.deviceId || `media-${index}`,
            name: device.label || `${device.kind} device`,
            family: category,
            device_class: category,
            description: "Browser-reported media device. Labels may stay private until permission is granted.",
          },
          enabled: true,
          connected: true,
          source: "browser-runtime",
          sourceLabel: "Browser Runtime",
          deviceClass: category,
          raw: { kind: device.kind, groupId: device.groupId, deviceId: device.deviceId },
        });
      });
    }
  } catch {}

  try {
    if (nav.storage?.estimate) {
      const estimate = await nav.storage.estimate();
      items.push(normalizeDevice({
        id: "browser-storage-estimate",
        name: "Browser Storage Estimate",
        device_class: "storage",
        present: true,
        quota: estimate.quota,
        usage: estimate.usage,
      }, "browser-runtime", "Browser Runtime", items.length, "storage")!);
    }
  } catch {}

  if (typeof nav.hardwareConcurrency === "number") {
    items.push(normalizeDevice({
      id: "browser-cpu-threads",
      name: `${nav.hardwareConcurrency} logical CPU threads visible to browser`,
      device_class: "cpu",
      present: true,
    }, "browser-runtime", "Browser Runtime", items.length, "other")!);
  }

  if (typeof screen !== "undefined") {
    items.push(normalizeDevice({
      id: "browser-display",
      name: `${screen.width} x ${screen.height} display surface`,
      device_class: "display",
      present: true,
      width: screen.width,
      height: screen.height,
      colorDepth: screen.colorDepth,
    }, "browser-runtime", "Browser Runtime", items.length, "other")!);
  }

  if (nav.connection) {
    items.push(normalizeDevice({
      id: "browser-network-link",
      name: `Browser network link ${nav.connection.effectiveType || ""}`.trim(),
      device_class: "network",
      present: true,
      effectiveType: nav.connection.effectiveType,
      downlink: nav.connection.downlink,
      rtt: nav.connection.rtt,
    }, "browser-runtime", "Browser Runtime", items.length, "network")!);
  }

  return items;
}

function mergeDevices(items: DriverItem[]) {
  const merged = new Map<string, DriverItem>();
  for (const item of items) {
    const key = item.bridgeDriverId ? `driver:${item.bridgeDriverId}` : `${item.source || "device"}:${item.id}`;
    const existing = merged.get(key);
    if (!existing) {
      merged.set(key, item);
      continue;
    }
    merged.set(key, {
      ...existing,
      ...item,
      manifest: { ...(existing.manifest || {}), ...(item.manifest || {}) },
      sourceLabel: Array.from(new Set([existing.sourceLabel, item.sourceLabel].filter(Boolean))).join(" + "),
      raw: { ...(existing.raw || {}), [`source_${slug(item.sourceLabel)}`]: item.raw },
    });
  }
  return Array.from(merged.values()).sort((a, b) => displayName(a).localeCompare(displayName(b)));
}

export function DeviceManagerScreen() {
  const [tab, setTab] = useState<DeviceTab>("all");
  const [drivers, setDrivers] = useState<DriverItem[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [configDraft, setConfigDraft] = useState<Record<string, string>>({});
  const [capabilities, setCapabilities] = useState<any>(null);
  const [governance, setGovernance] = useState<any>(null);
  const [inventorySources, setInventorySources] = useState<InventorySource[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<any>(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("Device Manager is waiting for driver inventory.");

  const selected = drivers.find((driver) => driver.id === selectedId) || drivers[0] || null;

  const loadDevices = async (preferredTab?: DeviceTab) => {
    setBusy("refresh");
    setMessage("Loading driver and device inventory...");
    try {
      const [caps, gov, ...inventory] = await Promise.allSettled([
        api.proxy.call("/api/drivers/capabilities", { method: "GET" }),
        api.proxy.call("/api/drivers/governance", { method: "GET" }),
        ...INVENTORY_ENDPOINTS.map((endpoint) => api.proxy.call(endpoint.route, { method: "GET" })),
      ]);
      setCapabilities(caps.status === "fulfilled" ? caps.value : { ok: false, error: String(caps.reason || "capabilities unavailable") });
      setGovernance(gov.status === "fulfilled" ? gov.value : { ok: false, error: String(gov.reason || "governance unavailable") });

      const sourceStates: InventorySource[] = [];
      const normalized: DriverItem[] = [];
      inventory.forEach((result, index) => {
        const endpoint = INVENTORY_ENDPOINTS[index];
        if (result.status === "fulfilled") {
          const devices = extractDeviceCandidates(result.value, endpoint.source, endpoint.label);
          normalized.push(...devices);
          sourceStates.push({ label: endpoint.label, ok: true, detail: `${devices.length} entries` });
        } else {
          sourceStates.push({ label: endpoint.label, ok: false, detail: String(result.reason || "unavailable") });
        }
      });

      const browserDevices = await collectBrowserDevices();
      normalized.push(...browserDevices);
      sourceStates.push({ label: "Browser Runtime", ok: true, detail: `${browserDevices.length} entries` });

      const items = mergeDevices(normalized);
      setDrivers(items);
      setInventorySources(sourceStates);
      const nextTab = preferredTab || tab;
      const firstVisible = items.find((driver) => nextTab === "all" || classifyDevice(driver) === nextTab) || items[0];
      setSelectedId((current) => (items.some((driver) => driver.id === current) ? current : firstVisible?.id || ""));
      setMessage(items.length ? `Detected ${items.length} dynamic device entries from ${sourceStates.filter((source) => source.ok).length} inventory sources.` : "No device inventory was returned by local sources.");
    } catch (error: any) {
      setMessage(String(error?.message || error || "Driver inventory failed."));
    } finally {
      setBusy("");
    }
  };

  useEffect(() => {
    let preferred: DeviceTab | undefined;
    try {
      const stored = window.sessionStorage.getItem("sarahmemory:device-manager:tab");
      if (stored && stored in TAB_META) preferred = stored as DeviceTab;
      if (preferred) setTab(preferred);
    } catch {}
    void loadDevices(preferred);

    const onDeviceManager = (event: Event) => {
      const next = ((event as CustomEvent).detail?.tab || "") as DeviceTab;
      if (next && next in TAB_META) {
        setTab(next);
        void loadDevices(next);
      }
    };
    window.addEventListener("sarah:device-manager", onDeviceManager);
    return () => window.removeEventListener("sarah:device-manager", onDeviceManager);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) return;
    if (!selected.bridgeDriverId) {
      setConfigDraft({});
      setSelectedEvidence(null);
      return;
    }
    let cancelled = false;
    const loadConfig = async () => {
      const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(selected.bridgeDriverId || selected.id)}/config`, { method: "GET" });
      if (cancelled) return;
      const cfg = ((result as any)?.config || {}) as Record<string, any>;
      const next: Record<string, string> = {};
      for (const [key] of CONFIG_FIELDS) next[key] = String(cfg[key] || "");
      setConfigDraft(next);
    };
    void loadConfig();
    return () => {
      cancelled = true;
    };
  }, [selected?.id]);

  const visibleDrivers = useMemo(() => {
    return drivers.filter((driver) => tab === "all" || classifyDevice(driver) === tab);
  }, [drivers, tab]);

  useEffect(() => {
    if (!visibleDrivers.length) return;
    if (!visibleDrivers.some((driver) => driver.id === selectedId)) {
      setSelectedId(visibleDrivers[0].id);
    }
  }, [selectedId, visibleDrivers]);

  const updateRegistry = async (driver: DriverItem, patch: Record<string, any>) => {
    if (!driver.bridgeDriverId) {
      setMessage(`${displayName(driver)} is a read-only detected device. Registry mutation requires an appdrivers bridge entry.`);
      return;
    }
    if (!window.confirm(`Update ${displayName(driver)} registry settings?`)) return;
    setBusy(`registry:${driver.id}`);
    const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(driver.bridgeDriverId)}/registry`, {
      method: "POST",
      body: { registry: patch, user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager" },
    });
    setMessage((result as any)?.ok ? "Registry updated." : `Registry update pending/blocked: ${(result as any)?.error || "bridge authorization required"}`);
    await loadDevices(tab);
  };

  const saveConfig = async () => {
    if (!selected) return;
    if (!selected.bridgeDriverId) {
      setMessage(`${displayName(selected)} is detected read-only. Save requires a governed driver bridge entry.`);
      return;
    }
    if (!window.confirm(`Save configuration for ${displayName(selected)}?`)) return;
    setBusy(`config:${selected.id}`);
    const configPatch = Object.fromEntries(
      Object.entries(configDraft).filter(([, value]) => value.trim().length > 0),
    );
    const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(selected.bridgeDriverId)}/config`, {
      method: "POST",
      body: { config: configPatch, user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager" },
    });
    setBusy("");
    setMessage((result as any)?.ok ? "Device configuration saved." : `Configuration save pending/blocked: ${(result as any)?.error || "bridge authorization required"}`);
  };

  const runDriverSession = async (driver: DriverItem, action: "connect" | "disconnect") => {
    if (!driver.bridgeDriverId) {
      setMessage(`${displayName(driver)} is detected read-only. ${action} requires a governed driver bridge entry.`);
      return;
    }
    if (!window.confirm(`${action === "connect" ? "Connect" : "Disconnect"} ${displayName(driver)} through the governed driver bridge?`)) return;
    setBusy(`${action}:${driver.id}`);
    const endpoint = `/api/drivers/${encodeURIComponent(driver.bridgeDriverId)}/${action}`;
    const result = await api.proxy.call(endpoint, {
      method: "POST",
      body: { user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager", payload: configDraft },
    });
    setBusy("");
    setMessage((result as any)?.ok ? `${displayName(driver)} ${action} request accepted.` : `${displayName(driver)} ${action} pending/blocked: ${(result as any)?.error || (result as any)?.reason || "governance response required"}`);
    await loadDevices(tab);
  };

  const readDriverSignal = async (driver: DriverItem, signal: "discover" | "status") => {
    if (!driver.bridgeDriverId) {
      setSelectedEvidence({ ok: false, reason: "No appdrivers bridge entry for this read-only detected device.", source: driver.sourceLabel });
      return;
    }
    setBusy(`${signal}:${driver.id}`);
    try {
      const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(driver.bridgeDriverId)}/${signal}`, { method: "GET" });
      setSelectedEvidence(result);
      setMessage(`${displayName(driver)} ${signal} response loaded.`);
    } catch (error: any) {
      setSelectedEvidence({ ok: false, error: String(error?.message || error || `${signal} failed`) });
      setMessage(`${displayName(driver)} ${signal} failed.`);
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      <div className="shrink-0 border-b border-border bg-card/70 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <MonitorCog className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Device Manager</h2>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Boot-detected driver and hardware control surface. Enable, disable, configure, discover, connect, and disconnect through the governed driver bridge.
            </p>
          </div>
          <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => void loadDevices(tab)} disabled={busy === "refresh"}>
            {busy === "refresh" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </div>

      <Tabs value={tab} onValueChange={(value) => setTab(value as DeviceTab)} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="grid h-auto grid-cols-4 rounded-none border-b border-border bg-background/80 p-1 lg:grid-cols-8">
          {(Object.keys(TAB_META) as DeviceTab[]).map((key) => {
            const Icon = TAB_META[key].icon;
            return (
              <TabsTrigger key={key} value={key} className="gap-1 text-xs">
                <Icon className="h-3.5 w-3.5" />
                {TAB_META[key].label}
              </TabsTrigger>
            );
          })}
        </TabsList>

        {(Object.keys(TAB_META) as DeviceTab[]).map((key) => (
          <TabsContent key={key} value={key} className="m-0 min-h-0 flex-1 overflow-hidden">
            <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[320px_1fr]">
              <div className="min-h-0 overflow-auto border-b border-border bg-card/35 p-3 lg:border-b-0 lg:border-r">
                <div className="mb-2 flex items-center justify-between gap-2 text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  <span>Devices</span>
                  <span>{visibleDrivers.length}</span>
                </div>
                <div className="space-y-2">
                  {visibleDrivers.length ? visibleDrivers.map((driver) => {
                    const active = selected?.id === driver.id;
                    const family = classifyDevice(driver);
                    const Icon = TAB_META[family].icon;
                    return (
                      <button
                        key={driver.id}
                        type="button"
                        onClick={() => setSelectedId(driver.id)}
                        className={cn(
                          "sarah-focus-ring w-full rounded-lg border p-3 text-left transition",
                          active ? "border-primary/60 bg-primary/10" : "border-border/70 bg-background/70 hover:border-primary/35",
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4 text-primary" />
                          <span className="truncate text-sm font-semibold">{displayName(driver)}</span>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-1 text-[10px] uppercase">
                          <span className={cn("rounded px-1.5 py-0.5", driver.enabled ? "bg-status-online/15 text-status-online" : "bg-muted text-muted-foreground")}>{driver.enabled ? "enabled" : "disabled"}</span>
                          <span className={cn("rounded px-1.5 py-0.5", driver.connected ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground")}>{driver.connected ? "connected" : "offline"}</span>
                          <span className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">{family}</span>
                          <span className="rounded bg-secondary/70 px-1.5 py-0.5 text-muted-foreground">{driver.sourceLabel || "source"}</span>
                        </div>
                      </button>
                    );
                  }) : (
                    <div className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
                      No devices in this category.
                    </div>
                  )}
                </div>
              </div>

              <div className="min-h-0 overflow-auto p-4">
                {selected ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-border bg-card/60 p-4">
                      <div>
                        <div className="flex items-center gap-2 text-base font-semibold">
                          <PlugZap className="h-5 w-5 text-primary" />
                          {displayName(selected)}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">{selected.id}</div>
                        <div className="mt-2 flex flex-wrap gap-1 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                          <span className="rounded bg-secondary/70 px-2 py-0.5">{selected.sourceLabel || "Detected"}</span>
                          <span className="rounded bg-secondary/70 px-2 py-0.5">{classifyDevice(selected)}</span>
                          {selected.bridgeDriverId ? <span className="rounded bg-primary/15 px-2 py-0.5 text-primary">appdrivers</span> : <span className="rounded bg-muted px-2 py-0.5">read-only</span>}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Button type="button" variant="outline" size="sm" onClick={() => void readDriverSignal(selected, "discover")} disabled={!selected.bridgeDriverId || busy.endsWith(selected.id)}>Discover</Button>
                        <Button type="button" variant="outline" size="sm" onClick={() => void readDriverSignal(selected, "status")} disabled={!selected.bridgeDriverId || busy.endsWith(selected.id)}>Status</Button>
                        <Button type="button" variant="outline" size="sm" onClick={() => void runDriverSession(selected, "connect")} disabled={!selected.bridgeDriverId || busy.endsWith(selected.id)}>Connect</Button>
                        <Button type="button" variant="outline" size="sm" onClick={() => void runDriverSession(selected, "disconnect")} disabled={!selected.bridgeDriverId || busy.endsWith(selected.id)}>Disconnect</Button>
                      </div>
                    </div>

                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-xl border border-border bg-card/60 p-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <span className="text-sm font-medium">Enabled</span>
                          <Switch disabled={!selected.bridgeDriverId} checked={Boolean(selected.enabled)} onCheckedChange={(checked) => void updateRegistry(selected, { enabled: checked })} />
                        </div>
                        <p className="text-xs text-muted-foreground">Controls whether this driver is allowed by registry policy.</p>
                      </div>
                      <div className="rounded-xl border border-border bg-card/60 p-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <span className="text-sm font-medium">Autoload</span>
                          <Switch disabled={!selected.bridgeDriverId} checked={Boolean(selected.autoload)} onCheckedChange={(checked) => void updateRegistry(selected, { autoload: checked })} />
                        </div>
                        <p className="text-xs text-muted-foreground">Allows boot/runtime auto-attach when governance permits it.</p>
                      </div>
                      <div className="rounded-xl border border-border bg-card/60 p-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <span className="text-sm font-medium">Trusted</span>
                          <Switch disabled={!selected.bridgeDriverId} checked={Boolean(selected.trusted)} onCheckedChange={(checked) => void updateRegistry(selected, { trusted: checked })} />
                        </div>
                        <p className="text-xs text-muted-foreground">Marks operator trust intent; backend validation remains authority.</p>
                      </div>
                    </div>

                    <div className="rounded-xl border border-border bg-card/60 p-4">
                      <div className="mb-3 flex items-center gap-2 font-medium">
                        <SlidersHorizontal className="h-4 w-4 text-primary" />
                        Network / Device Configuration
                      </div>
                      <div className="grid gap-3 md:grid-cols-2">
                        {CONFIG_FIELDS.map(([key, label]) => (
                          <label key={key} className={cn("space-y-1 text-xs", key === "notes" && "md:col-span-2")}>
                            <span className="text-muted-foreground">{label}</span>
                            <Input
                              type={key === "password" ? "password" : "text"}
                              value={configDraft[key] || ""}
                              onChange={(e) => setConfigDraft((draft) => ({ ...draft, [key]: e.target.value }))}
                              placeholder={key === "password" ? "Stored only if saved through an authorized driver bridge" : label}
                            />
                          </label>
                        ))}
                      </div>
                      <Button type="button" className="mt-3 gap-2" onClick={() => void saveConfig()} disabled={!selected || busy.startsWith("config:")}>
                        <Save className="h-4 w-4" />
                        Save Device Config
                      </Button>
                    </div>

                    <div className="grid gap-3 xl:grid-cols-2">
                      <div className="rounded-xl border border-border bg-card/60 p-4">
                        <div className="mb-2 flex items-center gap-2 font-medium">
                          <ShieldCheck className="h-4 w-4 text-primary" />
                          Driver Manifest
                        </div>
                        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-[11px] text-muted-foreground">{JSON.stringify(selected.manifest || {}, null, 2)}</pre>
                      </div>
                      <div className="rounded-xl border border-border bg-card/60 p-4">
                        <div className="mb-2 flex items-center gap-2 font-medium">
                          <ShieldCheck className="h-4 w-4 text-primary" />
                          Inventory / Governance
                        </div>
                        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-[11px] text-muted-foreground">{JSON.stringify({ capabilities, governance, inventorySources, selectedEvidence, message }, null, 2).slice(0, 2600)}</pre>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">
                    Select a device to inspect and configure it.
                  </div>
                )}
              </div>
            </div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
