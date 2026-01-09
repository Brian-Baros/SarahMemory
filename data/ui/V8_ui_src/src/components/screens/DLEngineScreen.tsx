import { useState, useEffect } from 'react';
import { 
  Cpu, 
  Activity, 
  BarChart3, 
  Loader2,
  AlertCircle,
  TrendingUp,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';

interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  progress: number;
  startedAt?: Date;
}

interface EngineStats {
  modelsLoaded: number;
  activeJobs: number;
  memoryUsage: number;
  gpuUsage: number;
}

/**
 * DL Engine Screen - Deep learning model training/inference
 * Shows training progress, model stats, and job queue
 */
export function DLEngineScreen() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    setIsLoading(true);
    try {
      const response = await api.proxy.call('/api/dlengine/status');
      
      if (response) {
        setIsAvailable(true);
        if ((response as any).jobs) {
          setJobs((response as any).jobs);
        }
        if ((response as any).stats) {
          setStats((response as any).stats);
        }
      }
    } catch (error) {
      console.warn('[DLEngine] Not available:', error);
      setIsAvailable(false);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (status: TrainingJob['status']) => {
    switch (status) {
      case 'running': return 'text-blue-500';
      case 'complete': return 'text-green-500';
      case 'error': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="shrink-0 p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">DL Engine</h1>
          <Button 
            variant="ghost" 
            size="icon" 
            className="ml-auto h-8 w-8"
            onClick={checkStatus}
            disabled={isLoading}
          >
            <Activity className={cn("h-4 w-4", isLoading && "animate-pulse")} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Model training and inference
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Availability Notice */}
          {isAvailable === false && (
            <div className="p-4 rounded-xl bg-muted/50 border border-border">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Not available (server configuration)</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    DL Engine requires server-side GPU support.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Stats Cards */}
          {stats && (
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground">Models Loaded</p>
                <p className="text-2xl font-bold mt-1">{stats.modelsLoaded}</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground">Active Jobs</p>
                <p className="text-2xl font-bold mt-1">{stats.activeJobs}</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground mb-2">Memory</p>
                <Progress value={stats.memoryUsage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{stats.memoryUsage}%</p>
              </div>
              <div className="p-3 rounded-xl bg-card border border-border">
                <p className="text-xs text-muted-foreground mb-2">GPU</p>
                <Progress value={stats.gpuUsage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{stats.gpuUsage}%</p>
              </div>
            </div>
          )}

          {/* Training Jobs */}
          <div>
            <p className="text-sm font-medium mb-3 flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Training Jobs
            </p>
            
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : jobs.length === 0 ? (
              <div className="text-center py-8 rounded-xl bg-card border border-border">
                <BarChart3 className="h-10 w-10 mx-auto text-muted-foreground/50 mb-2" />
                <p className="text-sm text-muted-foreground">
                  {isAvailable ? "No active jobs" : "Engine not available"}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {jobs.map((job) => (
                  <div 
                    key={job.id}
                    className="p-3 rounded-xl bg-card border border-border"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-medium text-sm">{job.name}</p>
                      <span className={cn(
                        "text-xs font-medium capitalize",
                        getStatusColor(job.status)
                      )}>
                        {job.status}
                      </span>
                    </div>
                    {job.status === 'running' && (
                      <>
                        <Progress value={job.progress} className="h-2 mb-1" />
                        <p className="text-xs text-muted-foreground">{job.progress}% complete</p>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
