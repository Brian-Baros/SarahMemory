import { ChevronRight, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useSarahStore } from '@/stores/useSarahStore';
import { cn } from '@/lib/utils';
import { RightSidebarContent } from './RightSidebarContent';

/**
 * Right Sidebar - Desktop shell surface
 * Contains preview, tools, and utilities organized into tabbed pages
 *
 * NOTE:
 * Desktop vs Mobile is controlled by orientation in Index.tsx,
 * so this component should not hard-hide itself using lg: breakpoints.
 */
export function RightSidebar() {
  const { rightSidebarCollapsed, toggleRightSidebar } = useSarahStore();

  return (
    <aside
      className={cn(
        "flex bg-sidebar border-l border-sidebar-border flex-col transition-all duration-300",
        rightSidebarCollapsed ? "w-14" : "w-80"
      )}
    >
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-3 border-b border-sidebar-border shrink-0">
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleRightSidebar}
          className="text-sidebar-foreground hover:text-foreground hover:bg-sidebar-accent shrink-0"
        >
          <ChevronRight className={cn("h-4 w-4 transition-transform", rightSidebarCollapsed && "rotate-180")} />
        </Button>
        {!rightSidebarCollapsed && (
          <h2 className="font-semibold text-sidebar-foreground flex items-center gap-2">
            <Monitor className="h-4 w-4" />
            Preview
          </h2>
        )}
      </div>

      {!rightSidebarCollapsed && (
        <ScrollArea className="flex-1">
          <RightSidebarContent />
        </ScrollArea>
      )}
    </aside>
  );
}
