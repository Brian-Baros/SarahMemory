import { ChevronLeft, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useSarahStore } from '@/stores/useSarahStore';
import { cn } from '@/lib/utils';
import { LeftSidebarContent } from './LeftSidebarContent';

/**
 * Left Sidebar - Desktop only
 * Contains chat history/navigation
 * Hidden on mobile (replaced by drawer)
 */
export function LeftSidebar() {
  const { leftSidebarCollapsed, toggleLeftSidebar } = useSarahStore();

  return (
    <aside 
      className={cn(
        "hidden lg:flex bg-sidebar border-r border-sidebar-border flex-col transition-all duration-300",
        leftSidebarCollapsed ? "w-14" : "w-72"
      )}
    >
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-3 border-b border-sidebar-border shrink-0">
        {!leftSidebarCollapsed && (
          <h2 className="font-semibold text-sidebar-foreground flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Chats
          </h2>
        )}
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={toggleLeftSidebar}
          className="text-sidebar-foreground hover:text-foreground hover:bg-sidebar-accent shrink-0 ml-auto"
        >
          <ChevronLeft className={cn("h-4 w-4 transition-transform", leftSidebarCollapsed && "rotate-180")} />
        </Button>
      </div>

      {!leftSidebarCollapsed && (
        <ScrollArea className="flex-1">
          <LeftSidebarContent />
        </ScrollArea>
      )}
    </aside>
  );
}
