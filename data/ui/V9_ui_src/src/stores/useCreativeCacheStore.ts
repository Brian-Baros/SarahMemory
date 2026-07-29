import { create } from 'zustand';

// Session-based cache - cleared on tab/browser close (no persist)

export type CachedItemType = 'image' | 'music' | 'voice' | 'video';

export type UiAction = { type: string; payload?: any };

export interface CachedItem {
  id: string;
  type: CachedItemType;
  prompt: string;
  url?: string;
  blob?: Blob;
  preview?: string;
  createdAt: Date;
  metadata?: Record<string, any>;
}

interface CreativeCacheState {
  // Cached items by module
  items: {
    image: CachedItem[];
    music: CachedItem[];
    voice: CachedItem[];
    video: CachedItem[];
  };
  
  // Add item to cache
  addItem: (type: CachedItemType, item: Omit<CachedItem, 'id' | 'createdAt'>) => string;
  
  // Remove single item
  removeItem: (type: CachedItemType, id: string) => void;
  
  // Clear items for a specific module
  clearModule: (type: CachedItemType) => void;
  
  // Reset entire stack (all modules)
  resetStack: () => void;
  
  // Get items for a module
  getItems: (type: CachedItemType) => CachedItem[];
  
  // Download item
  downloadItem: (type: CachedItemType, id: string) => void;

  // UI Control Bus hook
  applyUiAction: (action: UiAction) => boolean;
}


const generateId = () => Math.random().toString(36).slice(2, 11);

export const useCreativeCacheStore = create<CreativeCacheState>((set, get) => ({
  items: {
    image: [],
    music: [],
    voice: [],
    video: [],
  },
  
  addItem: (type, item) => {
    const id = generateId();
    const newItem: CachedItem = {
      ...item,
      id,
      type,
      createdAt: new Date(),
    };
    
    set((state) => ({
      items: {
        ...state.items,
        [type]: [newItem, ...state.items[type]],
      },
    }));
    
    return id;
  },
  
  removeItem: (type, id) => {
    set((state) => ({
      items: {
        ...state.items,
        [type]: state.items[type].filter((item) => item.id !== id),
      },
    }));
  },
  
  clearModule: (type) => {
    set((state) => ({
      items: {
        ...state.items,
        [type]: [],
      },
    }));
  },
  
  resetStack: () => {
    set({
      items: {
        image: [],
        music: [],
        voice: [],
        video: [],
      },
    });
  },
  
  getItems: (type) => get().items[type],
  
  applyUiAction: (action) => {
    const type = String(action?.type || "");
    const p = action?.payload || {};

    if (type === "creative.cache.add") {
      if (!p?.itemType) return false;
      const t = String(p.itemType) as CachedItemType;
      const item = p.item || {};
      get().addItem(t, item);
      return true;
    }

    if (type === "creative.cache.remove") {
      const t = String(p?.itemType || "") as CachedItemType;
      const id = String(p?.id || "");
      if (!t || !id) return false;
      get().removeItem(t, id);
      return true;
    }

    if (type === "creative.cache.clear") {
      const t = String(p?.itemType || "") as CachedItemType;
      if (!t) return false;
      get().clearModule(t);
      return true;
    }

    if (type === "creative.cache.reset") {
      get().resetStack();
      return true;
    }

    if (type === "creative.cache.download") {
      const t = String(p?.itemType || "") as CachedItemType;
      const id = String(p?.id || "");
      if (!t || !id) return false;
      get().downloadItem(t, id);
      return true;
    }

    return false;
  },

  downloadItem: (type, id) => {
    const item = get().items[type].find((i) => i.id === id);
    if (!item) return;
    
    const downloadUrl = item.url || (item.blob ? URL.createObjectURL(item.blob) : null);
    if (!downloadUrl) return;
    
    const extension = type === 'image' ? 'png' : type === 'music' ? 'mp3' : type === 'voice' ? 'mp3' : 'mp4';
    const filename = `${type}-${item.id}.${extension}`;
    
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Cleanup blob URL if we created one
    if (item.blob && !item.url) {
      URL.revokeObjectURL(downloadUrl);
    }
  },
}));
