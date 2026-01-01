import { useState, useEffect } from 'react';
import { config, apiFetch } from '@/lib/config';
import { useSarahStore } from '@/stores/useSarahStore';

export function useBackendVersion() {
  const [version, setVersion] = useState<string>(config.version);
  const [loading, setLoading] = useState(true);
  const { bootstrapData } = useSarahStore();

  useEffect(() => {
    // Use bootstrap data if available
    if (bootstrapData?.version) {
      setVersion(bootstrapData.version);
      setLoading(false);
      return;
    }

    const fetchVersion = async () => {
      try {
        const response = await apiFetch<{ ok?: boolean; version?: string }>('/api/health');
        if (response && response.version) {
          setVersion(response.version);
        }
      } catch (error) {
        console.error('Failed to fetch backend version:', error);
        // Keep fallback version from config
      } finally {
        setLoading(false);
      }
    };

    fetchVersion();
    // Refresh version every 60 seconds
    const interval = setInterval(fetchVersion, 60000);
    return () => clearInterval(interval);
  }, [bootstrapData]);

  return { version, loading };
}
