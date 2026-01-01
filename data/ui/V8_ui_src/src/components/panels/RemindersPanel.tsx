import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Bell, Plus, Check, Clock, Trash2, Loader2, RefreshCw, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSarahStore } from '@/stores/useSarahStore';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface BackendReminder {
  id: string;
  title: string;
  description?: string;
  dueDate: Date;
  completed: boolean;
  priority?: 'low' | 'medium' | 'high';
  category?: string;
}

export function RemindersPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [backendReminders, setBackendReminders] = useState<BackendReminder[]>([]);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  
  // Form state
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    dueDate: '',
    dueTime: '',
    priority: 'medium' as 'low' | 'medium' | 'high',
    category: '',
  });

  const { 
    reminders: localReminders, 
    addReminder: addLocalReminder, 
    toggleReminderComplete: toggleLocalComplete, 
    deleteReminder: deleteLocalReminder 
  } = useSarahStore();

  // Fetch reminders from backend
  const fetchReminders = async () => {
    setIsLoading(true);
    try {
      const response = await api.proxy.getReminders();
      if (response.reminders && Array.isArray(response.reminders)) {
        const mapped = response.reminders.map((r: any) => ({
          id: r.id || r.reminder_id || String(Math.random()),
          title: r.title || r.text || '',
          description: r.description || '',
          dueDate: new Date(r.due_date || r.dueDate || r.time || Date.now()),
          completed: r.completed || r.is_completed || false,
          priority: r.priority || 'medium',
          category: r.category || '',
        }));
        setBackendReminders(mapped);
      }
      setHasLoaded(true);
    } catch (error) {
      console.error('Failed to fetch reminders:', error);
      setHasLoaded(true);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isExpanded && !hasLoaded) {
      fetchReminders();
    }
  }, [isExpanded, hasLoaded]);

  // Use backend reminders if available
  const reminders = backendReminders.length > 0 ? backendReminders : localReminders.map(r => ({
    ...r,
    dueDate: r.dueDate instanceof Date ? r.dueDate : new Date(r.dueDate),
  }));

  const activeReminders = reminders.filter(r => !r.completed);
  const completedReminders = reminders.filter(r => r.completed);

  const formatDueDate = (date: Date) => {
    const now = new Date();
    const diff = date.getTime() - now.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    
    if (diff < 0) return 'Overdue';
    if (hours < 1) return 'Soon';
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d`;
  };

  const getPriorityColor = (priority?: string) => {
    switch (priority) {
      case 'high': return 'text-status-error';
      case 'medium': return 'text-status-warning';
      default: return 'text-muted-foreground';
    }
  };

  const handleAddReminder = async () => {
    if (!formData.title.trim()) {
      toast.error('Title is required');
      return;
    }

    const dueDate = formData.dueDate && formData.dueTime
      ? new Date(`${formData.dueDate}T${formData.dueTime}`)
      : new Date(Date.now() + 86400000); // Tomorrow by default

    try {
      await api.proxy.call('/api/reminders', {
        method: 'POST',
        body: {
          title: formData.title,
          description: formData.description,
          due_date: dueDate.toISOString(),
          priority: formData.priority,
          category: formData.category,
        },
      });
      toast.success('Reminder created');
      fetchReminders();
    } catch (error) {
      // Fall back to local storage
      addLocalReminder({
        title: formData.title,
        description: formData.description || undefined,
        dueDate,
        completed: false,
        priority: formData.priority,
      });
      toast.success('Reminder saved locally');
    }

    setFormData({
      title: '',
      description: '',
      dueDate: '',
      dueTime: '',
      priority: 'medium',
      category: '',
    });
    setIsAddDialogOpen(false);
  };

  const handleToggleComplete = async (id: string, currentCompleted: boolean) => {
    try {
      await api.proxy.call(`/api/reminders/${id}`, {
        method: 'PUT',
        body: { completed: !currentCompleted },
      });
      fetchReminders();
    } catch (error) {
      toggleLocalComplete(id);
    }
  };

  const handleDeleteReminder = async (id: string) => {
    try {
      await api.proxy.call(`/api/reminders/${id}`, {
        method: 'DELETE',
      });
      toast.success('Reminder deleted');
      fetchReminders();
    } catch (error) {
      deleteLocalReminder(id);
      toast.success('Reminder removed');
    }
  };

  const handleSnooze = async (id: string) => {
    const snoozeTime = new Date(Date.now() + 3600000); // 1 hour later
    try {
      await api.proxy.call(`/api/reminders/${id}`, {
        method: 'PUT',
        body: { due_date: snoozeTime.toISOString() },
      });
      toast.success('Snoozed for 1 hour');
      fetchReminders();
    } catch (error) {
      toast.error('Failed to snooze');
    }
  };

  return (
    <div className="border-b border-sidebar-border">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-sidebar-accent transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-sidebar-foreground">
          <Bell className="h-4 w-4" />
          Reminders
          {activeReminders.length > 0 && (
            <span className="px-1.5 py-0.5 text-xs bg-primary text-primary-foreground rounded-full">
              {activeReminders.length}
            </span>
          )}
        </span>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-3 pt-0 space-y-3 animate-fade-in">
          {/* Add Reminder Button & Refresh */}
          <div className="flex gap-2">
            <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm" className="flex-1 h-8 text-xs">
                  <Plus className="h-3 w-3 mr-1.5" />
                  Add Reminder
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create Reminder</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 pt-4">
                  <div className="space-y-2">
                    <Label htmlFor="title">Title *</Label>
                    <Input
                      id="title"
                      value={formData.title}
                      onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
                      placeholder="Meeting with team"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Input
                      id="description"
                      value={formData.description}
                      onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Optional details..."
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="date">Date</Label>
                      <Input
                        id="date"
                        type="date"
                        value={formData.dueDate}
                        onChange={(e) => setFormData(prev => ({ ...prev, dueDate: e.target.value }))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="time">Time</Label>
                      <Input
                        id="time"
                        type="time"
                        value={formData.dueTime}
                        onChange={(e) => setFormData(prev => ({ ...prev, dueTime: e.target.value }))}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Priority</Label>
                      <Select 
                        value={formData.priority} 
                        onValueChange={(v) => setFormData(prev => ({ ...prev, priority: v as any }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="category">Category</Label>
                      <Input
                        id="category"
                        value={formData.category}
                        onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
                        placeholder="Work, Personal..."
                      />
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAddReminder}>
                      Create Reminder
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8"
              onClick={fetchReminders}
              disabled={isLoading}
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
          </div>

          {/* Active Reminders */}
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {isLoading && !hasLoaded ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : activeReminders.length === 0 ? (
              <div className="text-center py-4 text-muted-foreground text-sm">
                No active reminders
              </div>
            ) : (
              activeReminders.map((reminder) => (
                <div
                  key={reminder.id}
                  className="flex items-center gap-2 p-2 rounded-lg bg-sidebar-accent/50 hover:bg-sidebar-accent transition-colors group"
                >
                  <button
                    onClick={() => handleToggleComplete(reminder.id, reminder.completed)}
                    className="w-5 h-5 rounded-full border border-border flex items-center justify-center hover:border-primary hover:bg-primary/10 transition-colors"
                  >
                    <Check className="h-3 w-3 opacity-0 group-hover:opacity-50" />
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-foreground truncate">{reminder.title}</div>
                    {reminder.description && (
                      <div className="text-xs text-muted-foreground truncate">{reminder.description}</div>
                    )}
                  </div>
                  <div className={cn("flex items-center gap-1 text-xs", getPriorityColor(reminder.priority))}>
                    <Clock className="h-3 w-3" />
                    {formatDueDate(reminder.dueDate)}
                  </div>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => handleSnooze(reminder.id)}
                    title="Snooze 1 hour"
                  >
                    <Clock className="h-3 w-3 text-muted-foreground" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => handleDeleteReminder(reminder.id)}
                  >
                    <Trash2 className="h-3 w-3 text-destructive" />
                  </Button>
                </div>
              ))
            )}
          </div>

          {/* Completed Reminders */}
          {completedReminders.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground">Completed</div>
              {completedReminders.slice(0, 3).map((reminder) => (
                <div
                  key={reminder.id}
                  className="flex items-center gap-2 p-2 rounded-lg bg-sidebar-accent/30 opacity-60"
                >
                  <div className="w-5 h-5 rounded-full bg-status-online/20 flex items-center justify-center">
                    <Check className="h-3 w-3 text-status-online" />
                  </div>
                  <span className="text-sm text-muted-foreground line-through flex-1 truncate">
                    {reminder.title}
                  </span>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-6 w-6"
                    onClick={() => handleDeleteReminder(reminder.id)}
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
