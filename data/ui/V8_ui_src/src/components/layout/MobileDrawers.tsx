import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { useSarahStore } from '@/stores/useSarahStore';
import { ScrollArea } from '@/components/ui/scroll-area';
import { LeftSidebarContent } from './LeftSidebarContent';
import { RightSidebarContent } from './RightSidebarContent';
import { useSwipeGesture } from '@/hooks/useSwipeGesture';

/**
 * Mobile off-canvas drawers for left (chat history) and right (tools) panels.
 * These replace the sidebars on small screens.
 * Supports swipe gestures: swipe left to close left drawer, swipe right to close right drawer.
 */
export function MobileDrawers() {
  const { 
    leftDrawerOpen, 
    setLeftDrawerOpen,
    rightDrawerOpen,
    setRightDrawerOpen,
  } = useSarahStore();

  // Swipe left to close the left drawer
  const leftSwipeHandlers = useSwipeGesture({
    onSwipeLeft: () => setLeftDrawerOpen(false),
    threshold: 50,
  });

  // Swipe right to close the right drawer
  const rightSwipeHandlers = useSwipeGesture({
    onSwipeRight: () => setRightDrawerOpen(false),
    threshold: 50,
  });

  return (
    <>
      {/* Left Drawer - Chat History */}
      <Sheet open={leftDrawerOpen} onOpenChange={setLeftDrawerOpen}>
        <SheetContent 
          side="left" 
          className="w-80 p-0 bg-sidebar border-sidebar-border"
          {...leftSwipeHandlers}
        >
          <SheetHeader className="p-4 border-b border-sidebar-border">
            <SheetTitle className="text-sidebar-foreground">Chat History</SheetTitle>
          </SheetHeader>
          <ScrollArea className="h-[calc(100dvh-60px)]">
            <LeftSidebarContent />
          </ScrollArea>
        </SheetContent>
      </Sheet>

      {/* Right Drawer - Tools & Utilities */}
      <Sheet open={rightDrawerOpen} onOpenChange={setRightDrawerOpen}>
        <SheetContent 
          side="right" 
          className="w-80 p-0 bg-sidebar border-sidebar-border"
          {...rightSwipeHandlers}
        >
          <SheetHeader className="p-4 border-b border-sidebar-border">
            <SheetTitle className="text-sidebar-foreground">Tools & Settings</SheetTitle>
          </SheetHeader>
          <ScrollArea className="h-[calc(100dvh-60px)]">
            <RightSidebarContent />
          </ScrollArea>
        </SheetContent>
      </Sheet>
    </>
  );
}

