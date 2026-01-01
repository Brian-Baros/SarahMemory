import { Users, Phone, Wrench, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useSarahStore, type RightPanelPage } from '@/stores/useSarahStore';

const tabs: { id: RightPanelPage; label: string; icon: React.ElementType }[] = [
  { id: 'contacts', label: 'Contacts', icon: Users },
  { id: 'keypad', label: 'Keypad', icon: Phone },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'settings', label: 'Settings', icon: Settings },
];

/**
 * Tab bar for right panel page navigation
 * Works for both desktop sidebar and mobile drawer
 */
export function RightPanelTabs() {
  const { rightPanelPage, setRightPanelPage } = useSarahStore();

  return (
    <div className="flex border-b border-sidebar-border bg-sidebar-accent/50">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = rightPanelPage === tab.id;
        
        return (
          <button
            key={tab.id}
            onClick={() => setRightPanelPage(tab.id)}
            className={cn(
              "flex-1 flex flex-col items-center gap-1 py-2.5 px-1 text-xs transition-all",
              "border-b-2 -mb-[1px]",
              isActive 
                ? "text-primary border-primary bg-primary/5" 
                : "text-muted-foreground border-transparent hover:text-foreground hover:bg-sidebar-accent"
            )}
          >
            <Icon className="h-4 w-4" />
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        );
      })}
    </div>
  );
}
