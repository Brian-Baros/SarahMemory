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
  Monitor,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

type DeviceTab = "all" | "network" | "audio" | "printers" | "storage" | "input" | "camera" | "display" | "other";

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

type SettingKind = "text" | "password" | "textarea" | "select" | "switch" | "slider" | "number";

type SettingField = {
  key: string;
  label: string;
  kind: SettingKind;
  section: string;
  placeholder?: string;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  defaultValue?: string | number | boolean;
  wide?: boolean;
};

type DeviceProfile = {
  title: string;
  description: string;
  fields: SettingField[];
};

const NETWORK_FIELDS: SettingField[] = [
  { key: "adapter_alias", label: "Adapter Alias", kind: "text", section: "Identity", placeholder: "Office LAN / Workshop WiFi" },
  { key: "network_name", label: "Network Name", kind: "text", section: "Identity" },
  { key: "ssid", label: "SSID", kind: "text", section: "Wireless" },
  { key: "security_mode", label: "Security Mode", kind: "select", section: "Wireless", options: ["Auto", "Open", "WPA2 Personal", "WPA3 Personal", "Enterprise", "Hidden Network"], defaultValue: "Auto" },
  { key: "password", label: "Password / Passphrase", kind: "password", section: "Wireless", placeholder: "Stored only through authorized driver bridge" },
  { key: "band_preference", label: "Band Preference", kind: "select", section: "Wireless", options: ["Auto", "2.4 GHz", "5 GHz", "6 GHz"], defaultValue: "Auto" },
  { key: "auto_connect", label: "Auto Connect", kind: "switch", section: "Wireless", defaultValue: true },
  { key: "metered_connection", label: "Metered Connection", kind: "switch", section: "Wireless" },
  { key: "mac_randomization", label: "Random Hardware Address", kind: "switch", section: "Wireless" },
  { key: "roaming_aggression", label: "Roaming Aggression", kind: "slider", section: "Wireless", min: 0, max: 100, step: 5, unit: "%", defaultValue: 50 },
  { key: "ip_assignment", label: "IP Assignment", kind: "select", section: "TCP/IP", options: ["DHCP", "Manual IPv4", "Manual IPv6", "Static Dual Stack"], defaultValue: "DHCP" },
  { key: "tcp_ip", label: "IPv4 Address", kind: "text", section: "TCP/IP", placeholder: "192.168.1.50" },
  { key: "subnet_mask", label: "Subnet Mask / Prefix", kind: "text", section: "TCP/IP", placeholder: "255.255.255.0 or /24" },
  { key: "gateway", label: "Gateway", kind: "text", section: "TCP/IP", placeholder: "192.168.1.1" },
  { key: "dns_servers", label: "DNS Servers", kind: "textarea", section: "TCP/IP", placeholder: "1.1.1.1\n8.8.8.8", wide: true },
  { key: "ipv6_mode", label: "IPv6 Mode", kind: "select", section: "TCP/IP", options: ["Automatic", "Disabled", "Manual"], defaultValue: "Automatic" },
  { key: "mtu", label: "MTU", kind: "number", section: "TCP/IP", placeholder: "1500" },
  { key: "vlan_id", label: "VLAN ID", kind: "number", section: "TCP/IP" },
  { key: "vpn_profile", label: "VPN Profile", kind: "text", section: "VPN / Proxy" },
  { key: "vpn_protocol", label: "VPN Protocol", kind: "select", section: "VPN / Proxy", options: ["None", "WireGuard", "OpenVPN", "IKEv2", "L2TP/IPsec", "SSTP"], defaultValue: "None" },
  { key: "vpn_server", label: "VPN Server", kind: "text", section: "VPN / Proxy" },
  { key: "vpn_split_tunnel", label: "Split Tunnel", kind: "switch", section: "VPN / Proxy" },
  { key: "proxy_mode", label: "Proxy Mode", kind: "select", section: "VPN / Proxy", options: ["Off", "Auto Detect", "PAC Script", "Manual"], defaultValue: "Off" },
  { key: "proxy_address", label: "Proxy Address", kind: "text", section: "VPN / Proxy" },
  { key: "firewall_profile", label: "Firewall Profile", kind: "select", section: "Governance", options: ["Private", "Public", "Domain", "Isolated"], defaultValue: "Private" },
  { key: "wake_on_lan", label: "Wake on LAN", kind: "switch", section: "Governance" },
  { key: "priority", label: "Network Priority", kind: "slider", section: "Governance", min: 0, max: 100, step: 5, unit: "%", defaultValue: 50 },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const CAMERA_FIELDS: SettingField[] = [
  { key: "camera_profile", label: "Camera Profile", kind: "text", section: "Identity", placeholder: "Streaming / security / vision" },
  { key: "resolution", label: "Resolution", kind: "select", section: "Capture", options: ["Auto", "640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"], defaultValue: "Auto" },
  { key: "frame_rate", label: "Frame Rate", kind: "select", section: "Capture", options: ["Auto", "24 fps", "30 fps", "60 fps", "120 fps"], defaultValue: "Auto" },
  { key: "mirror", label: "Mirror Preview", kind: "switch", section: "Capture", defaultValue: true },
  { key: "rotation", label: "Rotation", kind: "select", section: "Capture", options: ["0 deg", "90 deg", "180 deg", "270 deg"], defaultValue: "0 deg" },
  { key: "video_hdr", label: "Video HDR", kind: "switch", section: "Capture" },
  { key: "brightness", label: "Brightness", kind: "slider", section: "Image", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "contrast", label: "Contrast", kind: "slider", section: "Image", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "saturation", label: "Saturation", kind: "slider", section: "Image", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "sharpness", label: "Sharpness", kind: "slider", section: "Image", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "hue", label: "Hue", kind: "slider", section: "Image", min: -180, max: 180, step: 1, unit: "deg", defaultValue: 0 },
  { key: "white_balance", label: "White Balance", kind: "slider", section: "Image", min: 2500, max: 9000, step: 100, unit: "K", defaultValue: 5000 },
  { key: "exposure", label: "Exposure", kind: "slider", section: "Lens", min: -10, max: 10, step: 1, defaultValue: 0 },
  { key: "gain", label: "Gain", kind: "slider", section: "Lens", min: 0, max: 100, step: 1, unit: "%", defaultValue: 25 },
  { key: "zoom", label: "Zoom", kind: "slider", section: "Lens", min: 100, max: 400, step: 5, unit: "%", defaultValue: 100 },
  { key: "pan", label: "Pan", kind: "slider", section: "Lens", min: -100, max: 100, step: 1, defaultValue: 0 },
  { key: "tilt", label: "Tilt", kind: "slider", section: "Lens", min: -100, max: 100, step: 1, defaultValue: 0 },
  { key: "autofocus", label: "Autofocus", kind: "switch", section: "Lens", defaultValue: true },
  { key: "focus_distance", label: "Focus Distance", kind: "slider", section: "Lens", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "low_light_boost", label: "Low Light Boost", kind: "switch", section: "Enhancement" },
  { key: "noise_reduction", label: "Noise Reduction", kind: "slider", section: "Enhancement", min: 0, max: 100, step: 5, unit: "%", defaultValue: 30 },
  { key: "auto_framing", label: "Auto Framing", kind: "switch", section: "Enhancement" },
  { key: "background_blur", label: "Background Blur", kind: "slider", section: "Enhancement", min: 0, max: 100, step: 5, unit: "%", defaultValue: 0 },
  { key: "privacy_shutter", label: "Privacy Shutter", kind: "switch", section: "Governance" },
  { key: "vision_lane", label: "Vision Lane", kind: "select", section: "Governance", options: ["Preview only", "Object recognition", "Face recognition review", "Avatar witness", "Disabled"], defaultValue: "Preview only" },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const DISPLAY_FIELDS: SettingField[] = [
  { key: "display_alias", label: "Display Alias", kind: "text", section: "Identity", placeholder: "Main monitor / projector" },
  { key: "primary_display", label: "Primary Display", kind: "switch", section: "Layout" },
  { key: "resolution", label: "Resolution", kind: "select", section: "Layout", options: ["Auto", "1280x720", "1600x900", "1920x1080", "2560x1440", "3440x1440", "3840x2160"], defaultValue: "Auto" },
  { key: "refresh_rate", label: "Refresh Rate", kind: "select", section: "Layout", options: ["Auto", "60 Hz", "75 Hz", "120 Hz", "144 Hz", "165 Hz", "240 Hz"], defaultValue: "Auto" },
  { key: "scale", label: "Scale", kind: "slider", section: "Layout", min: 75, max: 300, step: 5, unit: "%", defaultValue: 100 },
  { key: "orientation", label: "Orientation", kind: "select", section: "Layout", options: ["Landscape", "Portrait", "Landscape flipped", "Portrait flipped"], defaultValue: "Landscape" },
  { key: "screen_position", label: "Screen Position", kind: "select", section: "Layout", options: ["Auto", "Left", "Center", "Right", "Above", "Below"], defaultValue: "Auto" },
  { key: "brightness", label: "Brightness", kind: "slider", section: "Color", min: 0, max: 100, step: 1, unit: "%", defaultValue: 70 },
  { key: "contrast", label: "Contrast", kind: "slider", section: "Color", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "gamma", label: "Gamma", kind: "slider", section: "Color", min: 80, max: 140, step: 1, unit: "%", defaultValue: 100 },
  { key: "hue", label: "Hue", kind: "slider", section: "Color", min: -180, max: 180, step: 1, unit: "deg", defaultValue: 0 },
  { key: "saturation", label: "Saturation", kind: "slider", section: "Color", min: 0, max: 150, step: 1, unit: "%", defaultValue: 100 },
  { key: "color_temperature", label: "Color Temperature", kind: "slider", section: "Color", min: 2500, max: 10000, step: 100, unit: "K", defaultValue: 6500 },
  { key: "color_profile", label: "Color Profile", kind: "text", section: "Color", placeholder: "ICC / calibrated profile" },
  { key: "hdr_mode", label: "HDR Mode", kind: "select", section: "HDR / Advanced", options: ["Off", "Auto", "On", "Video only"], defaultValue: "Auto" },
  { key: "sdr_brightness", label: "SDR Brightness", kind: "slider", section: "HDR / Advanced", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "dynamic_range", label: "Dynamic Range", kind: "select", section: "HDR / Advanced", options: ["Auto", "Full", "Limited"], defaultValue: "Auto" },
  { key: "bit_depth", label: "Bit Depth", kind: "select", section: "HDR / Advanced", options: ["Auto", "8-bit", "10-bit", "12-bit"], defaultValue: "Auto" },
  { key: "adaptive_sync", label: "Adaptive Sync", kind: "switch", section: "HDR / Advanced" },
  { key: "overscan", label: "Overscan / Underscan", kind: "slider", section: "HDR / Advanced", min: -20, max: 20, step: 1, unit: "%", defaultValue: 0 },
  { key: "night_filter", label: "Blue-Light Filter", kind: "switch", section: "Comfort" },
  { key: "power_saver", label: "Display Power Saver", kind: "switch", section: "Comfort" },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const AUDIO_FIELDS: SettingField[] = [
  { key: "device_role", label: "Device Role", kind: "select", section: "Routing", options: ["Default output", "Default input", "Communications", "Studio monitor", "Disabled"], defaultValue: "Default output" },
  { key: "channel_mode", label: "Channel Mode", kind: "select", section: "Routing", options: ["Mono", "Stereo", "2.1", "5.1 Surround", "7.1 Surround", "Spatial / HRTF"], defaultValue: "Stereo" },
  { key: "sample_rate", label: "Sample Rate", kind: "select", section: "Routing", options: ["44.1 kHz", "48 kHz", "96 kHz", "192 kHz"], defaultValue: "48 kHz" },
  { key: "bit_depth", label: "Bit Depth", kind: "select", section: "Routing", options: ["16-bit", "24-bit", "32-bit float"], defaultValue: "24-bit" },
  { key: "exclusive_mode", label: "Exclusive Mode", kind: "switch", section: "Routing" },
  { key: "master_volume", label: "Master Volume", kind: "slider", section: "Levels", min: 0, max: 100, step: 1, unit: "%", defaultValue: 78 },
  { key: "input_gain", label: "Input Gain", kind: "slider", section: "Levels", min: 0, max: 100, step: 1, unit: "%", defaultValue: 70 },
  { key: "monitor_mix", label: "Monitor Mix", kind: "slider", section: "Levels", min: 0, max: 100, step: 1, unit: "%", defaultValue: 0 },
  { key: "balance", label: "Left / Right Balance", kind: "slider", section: "Levels", min: -100, max: 100, step: 1, defaultValue: 0 },
  { key: "bass", label: "Bass", kind: "slider", section: "Equalizer", min: -12, max: 12, step: 1, unit: "dB", defaultValue: 0 },
  { key: "mid", label: "Mid", kind: "slider", section: "Equalizer", min: -12, max: 12, step: 1, unit: "dB", defaultValue: 0 },
  { key: "treble", label: "Treble", kind: "slider", section: "Equalizer", min: -12, max: 12, step: 1, unit: "dB", defaultValue: 0 },
  { key: "loudness_eq", label: "Loudness Equalization", kind: "switch", section: "Enhancement" },
  { key: "spatial_audio", label: "Spatial Audio", kind: "select", section: "Enhancement", options: ["Off", "Virtual surround", "Headphone spatial", "Room model"], defaultValue: "Off" },
  { key: "noise_suppression", label: "Noise Suppression", kind: "slider", section: "Enhancement", min: 0, max: 100, step: 5, unit: "%", defaultValue: 35 },
  { key: "echo_cancellation", label: "Echo Cancellation", kind: "switch", section: "Enhancement", defaultValue: true },
  { key: "voice_focus", label: "Voice Focus", kind: "switch", section: "Enhancement" },
  { key: "latency_buffer", label: "Latency Buffer", kind: "slider", section: "Advanced", min: 32, max: 512, step: 16, unit: "ms", defaultValue: 128 },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const INPUT_FIELDS: SettingField[] = [
  { key: "input_profile", label: "Input Profile", kind: "text", section: "Identity", placeholder: "Gaming / accessibility / workstation" },
  { key: "primary_button", label: "Primary Button", kind: "select", section: "Mouse", options: ["Left", "Right"], defaultValue: "Left" },
  { key: "reverse_clicks", label: "Reverse Mouse Clicking", kind: "switch", section: "Mouse" },
  { key: "pointer_speed", label: "Pointer Speed", kind: "slider", section: "Mouse", min: 1, max: 20, step: 1, defaultValue: 10 },
  { key: "enhance_precision", label: "Pointer Acceleration", kind: "switch", section: "Mouse", defaultValue: true },
  { key: "double_click_speed", label: "Double-Click Speed", kind: "slider", section: "Mouse", min: 0, max: 100, step: 1, unit: "%", defaultValue: 50 },
  { key: "click_lock", label: "Click Lock", kind: "switch", section: "Mouse" },
  { key: "pointer_trails", label: "Pointer Trail", kind: "switch", section: "Mouse" },
  { key: "pointer_size", label: "Pointer Size", kind: "slider", section: "Mouse", min: 1, max: 15, step: 1, defaultValue: 3 },
  { key: "scroll_lines", label: "Scroll Wheel Lines", kind: "slider", section: "Scroll Wheel", min: 1, max: 20, step: 1, defaultValue: 3 },
  { key: "horizontal_scroll_chars", label: "Horizontal Scroll Characters", kind: "slider", section: "Scroll Wheel", min: 1, max: 40, step: 1, defaultValue: 3 },
  { key: "reverse_scroll", label: "Reverse Scroll Direction", kind: "switch", section: "Scroll Wheel" },
  { key: "keyboard_layout", label: "Keyboard Layout", kind: "select", section: "Keyboard", options: ["US", "US International", "UK", "Custom"], defaultValue: "US" },
  { key: "repeat_delay", label: "Repeat Delay", kind: "slider", section: "Keyboard", min: 0, max: 100, step: 5, unit: "%", defaultValue: 35 },
  { key: "repeat_rate", label: "Repeat Rate", kind: "slider", section: "Keyboard", min: 0, max: 100, step: 5, unit: "%", defaultValue: 70 },
  { key: "sticky_keys", label: "Sticky Keys", kind: "switch", section: "Keyboard" },
  { key: "filter_keys", label: "Filter Keys", kind: "switch", section: "Keyboard" },
  { key: "tap_to_click", label: "Tap to Click", kind: "switch", section: "Touch / Gesture", defaultValue: true },
  { key: "gesture_sensitivity", label: "Gesture Sensitivity", kind: "slider", section: "Touch / Gesture", min: 0, max: 100, step: 5, unit: "%", defaultValue: 50 },
  { key: "palm_rejection", label: "Palm Rejection", kind: "slider", section: "Touch / Gesture", min: 0, max: 100, step: 5, unit: "%", defaultValue: 60 },
  { key: "gamepad_deadzone", label: "Gamepad Deadzone", kind: "slider", section: "Gamepad", min: 0, max: 50, step: 1, unit: "%", defaultValue: 8 },
  { key: "gamepad_vibration", label: "Gamepad Vibration", kind: "switch", section: "Gamepad", defaultValue: true },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const PRINTER_FIELDS: SettingField[] = [
  { key: "printer_alias", label: "Printer Alias", kind: "text", section: "Identity" },
  { key: "default_printer", label: "Default Printer", kind: "switch", section: "Identity" },
  { key: "paper_size", label: "Paper Size", kind: "select", section: "Job Defaults", options: ["Letter", "Legal", "A4", "A3", "4x6 Photo", "Envelope", "Custom"], defaultValue: "Letter" },
  { key: "orientation", label: "Orientation", kind: "select", section: "Job Defaults", options: ["Portrait", "Landscape"], defaultValue: "Portrait" },
  { key: "duplex", label: "Duplex", kind: "select", section: "Job Defaults", options: ["Off", "Long edge", "Short edge"], defaultValue: "Off" },
  { key: "color_mode", label: "Color Mode", kind: "select", section: "Job Defaults", options: ["Color", "Black and white", "Grayscale", "Auto"], defaultValue: "Auto" },
  { key: "quality", label: "Quality", kind: "select", section: "Job Defaults", options: ["Draft", "Normal", "High", "Photo"], defaultValue: "Normal" },
  { key: "copies", label: "Copies", kind: "number", section: "Job Defaults", defaultValue: 1 },
  { key: "scale", label: "Scale", kind: "slider", section: "Job Defaults", min: 25, max: 200, step: 1, unit: "%", defaultValue: 100 },
  { key: "input_tray", label: "Input Tray", kind: "select", section: "Media", options: ["Auto", "Tray 1", "Tray 2", "Manual feed", "Photo tray"], defaultValue: "Auto" },
  { key: "media_type", label: "Media Type", kind: "select", section: "Media", options: ["Plain", "Matte", "Glossy", "Cardstock", "Envelope", "Transparency"], defaultValue: "Plain" },
  { key: "collate", label: "Collate", kind: "switch", section: "Finishing", defaultValue: true },
  { key: "staple", label: "Staple", kind: "switch", section: "Finishing" },
  { key: "hole_punch", label: "Hole Punch", kind: "switch", section: "Finishing" },
  { key: "toner_saver", label: "Toner / Ink Saver", kind: "switch", section: "Economy" },
  { key: "secure_print", label: "Secure Print", kind: "switch", section: "Governance" },
  { key: "share_name", label: "Share Name", kind: "text", section: "Governance" },
  { key: "port_name", label: "Port / Queue", kind: "text", section: "Governance" },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const STORAGE_FIELDS: SettingField[] = [
  { key: "volume_label", label: "Volume Label", kind: "text", section: "Identity" },
  { key: "drive_letter", label: "Drive Letter / Mount", kind: "text", section: "Identity", placeholder: "D: or /mnt/data" },
  { key: "file_system", label: "File System", kind: "select", section: "Format", options: ["NTFS", "exFAT", "FAT32", "ReFS", "ext4", "APFS", "Unknown"], defaultValue: "Unknown" },
  { key: "allocation_unit", label: "Allocation Unit", kind: "select", section: "Format", options: ["Default", "4 KB", "16 KB", "64 KB", "1 MB"], defaultValue: "Default" },
  { key: "removal_policy", label: "Removal Policy", kind: "select", section: "Performance", options: ["Quick removal", "Better performance", "Internal fixed disk"], defaultValue: "Quick removal" },
  { key: "write_cache", label: "Write Cache", kind: "switch", section: "Performance" },
  { key: "trim_enabled", label: "TRIM / Optimize", kind: "switch", section: "Performance", defaultValue: true },
  { key: "indexing", label: "Indexing", kind: "switch", section: "Performance" },
  { key: "compression", label: "Compression", kind: "switch", section: "Performance" },
  { key: "encryption", label: "Encryption", kind: "select", section: "Security", options: ["Off", "Software", "Hardware", "External vault"], defaultValue: "Off" },
  { key: "quota_gb", label: "Quota GB", kind: "number", section: "Security" },
  { key: "smart_monitoring", label: "SMART Monitoring", kind: "switch", section: "Health", defaultValue: true },
  { key: "health_alert_temp", label: "Temperature Alert", kind: "slider", section: "Health", min: 35, max: 90, step: 1, unit: "C", defaultValue: 65 },
  { key: "backup_profile", label: "Backup Profile", kind: "text", section: "Health" },
  { key: "power_policy", label: "Power Policy", kind: "select", section: "Health", options: ["Balanced", "Performance", "Power saver", "Never sleep"], defaultValue: "Balanced" },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const OTHER_FIELDS: SettingField[] = [
  { key: "device_alias", label: "Device Alias", kind: "text", section: "Identity" },
  { key: "device_role", label: "Device Role", kind: "text", section: "Identity", placeholder: "Compute, sensor, controller, bridge" },
  { key: "power_mode", label: "Power Mode", kind: "select", section: "Runtime", options: ["Auto", "Low power", "Balanced", "Performance", "Disabled"], defaultValue: "Auto" },
  { key: "performance_limit", label: "Performance Limit", kind: "slider", section: "Runtime", min: 0, max: 100, step: 5, unit: "%", defaultValue: 80 },
  { key: "telemetry_level", label: "Telemetry Level", kind: "select", section: "Runtime", options: ["Off", "Basic", "Detailed", "Diagnostic"], defaultValue: "Basic" },
  { key: "firmware_channel", label: "Firmware Channel", kind: "select", section: "Maintenance", options: ["Stable", "Preview", "Manual only"], defaultValue: "Stable" },
  { key: "diagnostic_mode", label: "Diagnostic Mode", kind: "switch", section: "Maintenance" },
  { key: "notes", label: "Notes", kind: "textarea", section: "Governance", wide: true },
];

const DEVICE_PROFILES: Record<DeviceTab, DeviceProfile> = {
  all: { title: "Device Options", description: "Select a concrete device class to edit its settings.", fields: OTHER_FIELDS },
  network: { title: "Network Adapter Options", description: "LAN, WiFi, TCP/IP, DNS, VPN, proxy, roaming, and governed network profile settings.", fields: NETWORK_FIELDS },
  audio: { title: "Audio Device Options", description: "Output/input role, mono/stereo/surround routing, EQ, spatial audio, noise control, and latency tuning.", fields: AUDIO_FIELDS },
  printers: { title: "Printer Options", description: "Paper, color, duplex, quality, finishing, queue, sharing, and secure-print profile settings.", fields: PRINTER_FIELDS },
  storage: { title: "Storage Device Options", description: "Mount identity, file system, write cache, removal policy, indexing, health, encryption, and backup settings.", fields: STORAGE_FIELDS },
  input: { title: "Input Device Options", description: "Mouse, keyboard, scroll wheel, touchpad, gesture, pointer, accessibility, and gamepad settings.", fields: INPUT_FIELDS },
  camera: { title: "Camera Options", description: "Image, lens, capture, mirror, zoom, autofocus, HDR, low-light, and vision-lane controls.", fields: CAMERA_FIELDS },
  display: { title: "Display Options", description: "Resolution, refresh, scale, orientation, color, HDR, calibration, comfort, and advanced display controls.", fields: DISPLAY_FIELDS },
  other: { title: "General Device Options", description: "Fallback governed settings for compute, sensors, controllers, and unknown hardware.", fields: OTHER_FIELDS },
};

const TAB_META: Record<DeviceTab, { label: string; icon: any; needles: string[] }> = {
  all: { label: "All", icon: MonitorCog, needles: [] },
  network: { label: "Networks", icon: Wifi, needles: ["network", "wifi", "wi-fi", "wireless", "ethernet", "tcp", "vpn"] },
  audio: { label: "Audio", icon: Speaker, needles: ["audio", "sound", "speaker", "microphone", "voice"] },
  printers: { label: "Printers", icon: Printer, needles: ["printer", "print", "scanner"] },
  storage: { label: "Storage", icon: HardDrive, needles: ["storage", "disk", "drive", "nvme", "sata", "usb"] },
  input: { label: "Input", icon: Keyboard, needles: ["keyboard", "mouse", "touch", "gamepad", "controller", "input"] },
  camera: { label: "Cameras", icon: Camera, needles: ["camera", "webcam", "vision", "uvc"] },
  display: { label: "Displays", icon: Monitor, needles: ["display", "monitor", "screen", "hdr", "resolution", "gpu", "video controller"] },
  other: { label: "Other", icon: Cpu, needles: [] },
};

const CLASSIFICATION_ORDER: DeviceTab[] = ["network", "audio", "printers", "storage", "camera", "display", "input"];

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
  for (const key of CLASSIFICATION_ORDER) {
    if (TAB_META[key].needles.some((needle) => text.includes(needle))) return key;
  }
  return "other";
}

function displayName(driver: DriverItem) {
  return String(driver.manifest?.name || driver.manifest?.label || driver.id);
}

function driverKey(driver: DriverItem) {
  return driver.bridgeDriverId ? `driver:${driver.bridgeDriverId}` : `${driver.source || "device"}:${driver.id}`;
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
  for (const key of CLASSIFICATION_ORDER) {
    if (TAB_META[key].needles.some((needle) => text.includes(needle))) return key;
  }
  return "other";
}

function getDeviceProfile(driver: DriverItem | null): DeviceProfile {
  return DEVICE_PROFILES[driver ? classifyDevice(driver) : "other"] || DEVICE_PROFILES.other;
}

function buildConfigDraft(profile: DeviceProfile, config: Record<string, any> = {}) {
  const draft: Record<string, string | number | boolean> = {};
  for (const field of profile.fields) {
    const raw = config[field.key];
    if (raw !== undefined && raw !== null) draft[field.key] = raw;
    else if (field.defaultValue !== undefined) draft[field.key] = field.defaultValue;
    else draft[field.key] = field.kind === "switch" ? false : "";
  }
  return draft;
}

function configSourceForDevice(driver: DriverItem) {
  const raw = driver.raw || {};
  return {
    ...(isRecord(raw.settings) ? raw.settings : {}),
    ...(isRecord(raw.config) ? raw.config : {}),
    ...raw,
  };
}

function serializeConfigDraft(draft: Record<string, string | number | boolean>) {
  return Object.fromEntries(
    Object.entries(draft).filter(([, value]) => {
      if (typeof value === "boolean") return true;
      if (typeof value === "number") return Number.isFinite(value);
      return String(value || "").trim().length > 0;
    }),
  );
}

function sectionsFor(profile: DeviceProfile) {
  const sections: Array<[string, SettingField[]]> = [];
  for (const field of profile.fields) {
    let section = sections.find(([name]) => name === field.section);
    if (!section) {
      section = [field.section, []];
      sections.push(section);
    }
    section[1].push(field);
  }
  return sections;
}

function sliderNumber(value: unknown, fallback: unknown) {
  const parsed = Number(value ?? fallback ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
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

  const directKeys = ["drivers", "devices", "items", "adapters", "network_adapters", "storage_devices", "camera_devices", "display_devices", "monitors", "printers", "audio_devices", "input_devices"];
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
    }, "browser-runtime", "Browser Runtime", items.length, "display")!);
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
    const key = driverKey(item);
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

function DeviceSettingField({
  field,
  value,
  disabled,
  onChange,
}: {
  field: SettingField;
  value: string | number | boolean | undefined;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
}) {
  const label = (
    <div className="mb-1 flex items-center justify-between gap-2">
      <span className="text-xs text-muted-foreground">{field.label}</span>
      {field.kind === "slider" ? (
        <span className="rounded bg-secondary/70 px-1.5 py-0.5 text-[10px] tabular-nums text-muted-foreground">
          {sliderNumber(value, field.defaultValue)}
          {field.unit || ""}
        </span>
      ) : null}
    </div>
  );

  if (field.kind === "switch") {
    const checked = value !== undefined ? boolValue(value) : boolValue(field.defaultValue);
    return (
      <div className={cn("rounded-lg border border-border/70 bg-background/45 p-3", field.wide && "md:col-span-2")}>
        <div className="flex items-center justify-between gap-3">
          <Label className="text-sm font-medium">{field.label}</Label>
          <Switch disabled={disabled} checked={checked} onCheckedChange={(checkedValue) => onChange(field.key, checkedValue)} />
        </div>
      </div>
    );
  }

  if (field.kind === "select") {
    const selected = safeString(value, safeString(field.defaultValue, field.options?.[0] || ""));
    return (
      <label className={cn("min-w-0", field.wide && "md:col-span-2")}>
        {label}
        <Select disabled={disabled} value={selected} onValueChange={(next) => onChange(field.key, next)}>
          <SelectTrigger className="min-w-0 bg-background/80">
            <SelectValue placeholder={field.placeholder || field.label} />
          </SelectTrigger>
          <SelectContent className="z-[100000]">
            {(field.options || []).map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
    );
  }

  if (field.kind === "slider") {
    const n = sliderNumber(value, field.defaultValue);
    return (
      <div className={cn("min-w-0 rounded-lg border border-border/70 bg-background/45 p-3", field.wide && "md:col-span-2")}>
        {label}
        <Slider
          disabled={disabled}
          value={[n]}
          min={field.min ?? 0}
          max={field.max ?? 100}
          step={field.step ?? 1}
          onValueChange={(next) => onChange(field.key, next[0] ?? n)}
        />
      </div>
    );
  }

  if (field.kind === "textarea") {
    return (
      <label className={cn("min-w-0", field.wide && "md:col-span-2")}>
        {label}
        <Textarea
          disabled={disabled}
          className="min-h-20 resize-y bg-background/80"
          value={safeString(value)}
          onChange={(e) => onChange(field.key, e.target.value)}
          placeholder={field.placeholder || field.label}
        />
      </label>
    );
  }

  return (
    <label className={cn("min-w-0", field.wide && "md:col-span-2")}>
      {label}
      <Input
        disabled={disabled}
        className="min-w-0 bg-background/80"
        type={field.kind === "password" ? "password" : field.kind === "number" ? "number" : "text"}
        value={safeString(value, safeString(field.defaultValue))}
        onChange={(e) => onChange(field.key, field.kind === "number" ? (e.target.value === "" ? "" : Number(e.target.value)) : e.target.value)}
        placeholder={field.placeholder || field.label}
      />
    </label>
  );
}

function DeviceSettingsEditor({
  profile,
  draft,
  disabled,
  onChange,
}: {
  profile: DeviceProfile;
  draft: Record<string, string | number | boolean>;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
}) {
  return (
    <div className="space-y-4">
      <div>
        <div className="text-sm font-semibold">{profile.title}</div>
        <p className="mt-1 text-xs text-muted-foreground">{profile.description}</p>
      </div>
      {sectionsFor(profile).map(([section, fields]) => (
        <section key={section} className="rounded-lg border border-border/70 bg-secondary/10 p-3">
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            {section}
          </div>
          <div className="grid min-w-0 gap-3 md:grid-cols-2">
            {fields.map((field) => (
              <DeviceSettingField
                key={field.key}
                field={field}
                value={draft[field.key]}
                disabled={disabled}
                onChange={onChange}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export function DeviceManagerScreen() {
  const [tab, setTab] = useState<DeviceTab>("all");
  const [drivers, setDrivers] = useState<DriverItem[]>([]);
  const [selectedKey, setSelectedKey] = useState("");
  const [configDraft, setConfigDraft] = useState<Record<string, string | number | boolean>>({});
  const [capabilities, setCapabilities] = useState<any>(null);
  const [governance, setGovernance] = useState<any>(null);
  const [inventorySources, setInventorySources] = useState<InventorySource[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<any>(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("Device Manager is waiting for driver inventory.");

  const visibleDrivers = useMemo(() => {
    return drivers.filter((driver) => tab === "all" || classifyDevice(driver) === tab);
  }, [drivers, tab]);

  const selected = visibleDrivers.find((driver) => driverKey(driver) === selectedKey) || visibleDrivers[0] || null;
  const selectedProfile = getDeviceProfile(selected);

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
      const nextVisible = items.filter((driver) => nextTab === "all" || classifyDevice(driver) === nextTab);
      const firstVisible = nextVisible[0] || items[0];
      setSelectedKey((current) => (nextVisible.some((driver) => driverKey(driver) === current) ? current : firstVisible ? driverKey(firstVisible) : ""));
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
    const profile = getDeviceProfile(selected);
    if (!selected.bridgeDriverId) {
      setConfigDraft(buildConfigDraft(profile, configSourceForDevice(selected)));
      setSelectedEvidence(null);
      return;
    }
    let cancelled = false;
    const loadConfig = async () => {
      try {
        const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(selected.bridgeDriverId || selected.id)}/config`, { method: "GET" });
        if (cancelled) return;
        const cfg = { ...(((result as any)?.defaults || {}) as Record<string, any>), ...(((result as any)?.config || {}) as Record<string, any>) };
        setConfigDraft(buildConfigDraft(profile, cfg));
      } catch (error: any) {
        if (cancelled) return;
        setConfigDraft(buildConfigDraft(profile, configSourceForDevice(selected)));
        setMessage(`${displayName(selected)} config read failed: ${String(error?.message || error || "driver bridge unavailable")}`);
      }
    };
    void loadConfig();
    return () => {
      cancelled = true;
    };
  }, [selected ? driverKey(selected) : ""]);

  useEffect(() => {
    if (!visibleDrivers.length) return;
    if (!visibleDrivers.some((driver) => driverKey(driver) === selectedKey)) {
      setSelectedKey(driverKey(visibleDrivers[0]));
    }
  }, [selectedKey, visibleDrivers]);

  const updateRegistry = async (driver: DriverItem, patch: Record<string, any>) => {
    if (!driver.bridgeDriverId) {
      setMessage(`${displayName(driver)} is a read-only detected device. Registry mutation requires an appdrivers bridge entry.`);
      return;
    }
    if (!window.confirm(`Update ${displayName(driver)} registry settings?`)) return;
    setBusy(`registry:${driver.id}`);
    try {
      const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(driver.bridgeDriverId)}/registry`, {
        method: "POST",
        body: { registry: patch, user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager" },
      });
      setMessage((result as any)?.ok ? "Registry updated." : `Registry update pending/blocked: ${(result as any)?.error || "bridge authorization required"}`);
      await loadDevices(tab);
    } catch (error: any) {
      setMessage(`${displayName(driver)} registry update failed: ${String(error?.message || error || "driver bridge unavailable")}`);
    } finally {
      setBusy("");
    }
  };

  const saveConfig = async () => {
    if (!selected) return;
    if (!selected.bridgeDriverId) {
      setMessage(`${displayName(selected)} is detected read-only. Save requires a governed driver bridge entry.`);
      return;
    }
    if (!window.confirm(`Save configuration for ${displayName(selected)}?`)) return;
    setBusy(`config:${selected.id}`);
    const configPatch = serializeConfigDraft(configDraft);
    try {
      const result = await api.proxy.call(`/api/drivers/${encodeURIComponent(selected.bridgeDriverId)}/config`, {
        method: "POST",
        body: { config: configPatch, user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager" },
      });
      setMessage((result as any)?.ok ? "Device configuration saved." : `Configuration save pending/blocked: ${(result as any)?.error || "bridge authorization required"}`);
    } catch (error: any) {
      setMessage(`${displayName(selected)} configuration save failed: ${String(error?.message || error || "driver bridge unavailable")}`);
    } finally {
      setBusy("");
    }
  };

  const runDriverSession = async (driver: DriverItem, action: "connect" | "disconnect") => {
    if (!driver.bridgeDriverId) {
      setMessage(`${displayName(driver)} is detected read-only. ${action} requires a governed driver bridge entry.`);
      return;
    }
    if (!window.confirm(`${action === "connect" ? "Connect" : "Disconnect"} ${displayName(driver)} through the governed driver bridge?`)) return;
    setBusy(`${action}:${driver.id}`);
    const endpoint = `/api/drivers/${encodeURIComponent(driver.bridgeDriverId)}/${action}`;
    const configPatch = serializeConfigDraft(configDraft);
    try {
      const result = await api.proxy.call(endpoint, {
        method: "POST",
        body: { config: configPatch, user_confirmed: true, operator_confirmed: true, source: "frontend:device_manager", payload: { ...configPatch, action } },
      });
      setMessage((result as any)?.ok ? `${displayName(driver)} ${action} request accepted.` : `${displayName(driver)} ${action} pending/blocked: ${(result as any)?.error || (result as any)?.reason || "governance response required"}`);
      await loadDevices(tab);
    } catch (error: any) {
      setMessage(`${displayName(driver)} ${action} failed: ${String(error?.message || error || "driver bridge unavailable")}`);
    } finally {
      setBusy("");
    }
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

  const updateConfigDraft = (key: string, value: string | number | boolean) => {
    setConfigDraft((draft) => ({ ...draft, [key]: value }));
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
        <TabsList className="grid h-auto grid-cols-3 rounded-none border-b border-border bg-background/80 p-1 sm:grid-cols-5 xl:grid-cols-9">
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
                        key={driverKey(driver)}
                        type="button"
                        onClick={() => setSelectedKey(driverKey(driver))}
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

                    <div className="overflow-hidden rounded-xl border border-border bg-card/60 p-4">
                      <div className="mb-3 flex items-center gap-2 font-medium">
                        <SlidersHorizontal className="h-4 w-4 text-primary" />
                        {selectedProfile.title}
                      </div>
                      <DeviceSettingsEditor
                        profile={selectedProfile}
                        draft={configDraft}
                        disabled={busy.startsWith("config:")}
                        onChange={updateConfigDraft}
                      />
                      {!selected.bridgeDriverId ? (
                        <div className="mt-3 rounded-md border border-dashed border-border/80 bg-background/55 p-2 text-xs text-muted-foreground">
                          This device was detected from read-only inventory. A matching appdrivers bridge entry is required before SarahMemory can apply settings to hardware.
                        </div>
                      ) : null}
                      <Button type="button" className="mt-3 gap-2" onClick={() => void saveConfig()} disabled={!selected || !selected.bridgeDriverId || busy.startsWith("config:")}>
                        <Save className="h-4 w-4" />
                        Save {classifyDevice(selected)} Config
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
