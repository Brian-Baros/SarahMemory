import { useState } from 'react';
import { Phone, PhoneOff, User, Hash, Globe } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown } from 'lucide-react';
import { api } from '@/lib/api';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

export function DialerPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [dialInput, setDialInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isVoipAvailable, setIsVoipAvailable] = useState<boolean | null>(null);
  const [isInCall, setIsInCall] = useState(false);

  const dialPad = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['*', '0', '#'],
  ];

  const handleDialPress = (digit: string) => {
    setDialInput(prev => prev + digit);
  };

  const handleBackspace = () => {
    setDialInput(prev => prev.slice(0, -1));
  };

  const handleCall = async () => {
    if (!dialInput.trim()) {
      toast.error('Please enter a number or address');
      return;
    }

    setIsLoading(true);
    
    try {
      // Check availability first
      if (isVoipAvailable === null) {
        const availability = await api.dialer.checkAvailability();
        setIsVoipAvailable(availability.available || false);
        
        if (!availability.available) {
          toast.info(availability.message || 'VoIP feature coming soon!');
          setIsLoading(false);
          return;
        }
      }

      // Determine call type
      const isIP = /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$/.test(dialInput);
      const isRoomId = dialInput.startsWith('#');
      
      const response = await api.dialer.initiateCall({
        number: !isIP && !isRoomId ? dialInput : undefined,
        ip_address: isIP ? dialInput : undefined,
        room_id: isRoomId ? dialInput.slice(1) : undefined,
      });

      if (response.success) {
        setIsInCall(true);
        toast.success('Call initiated');
      } else {
        toast.info(response.message || 'Feature coming soon');
      }
    } catch (error) {
      console.error('Dialer error:', error);
      toast.error('Failed to initiate call');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEndCall = async () => {
    setIsLoading(true);
    
    try {
      await api.dialer.endCall();
      setIsInCall(false);
      toast.info('Call ended');
    } catch (error) {
      console.error('End call error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAsContact = () => {
    if (!dialInput.trim()) {
      toast.error('Please enter a number first');
      return;
    }
    
    // This will be handled by the contacts panel
    toast.info('Contact saving feature coming soon');
  };

  return (
    <div className="border-b border-sidebar-border">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-sidebar-accent transition-colors"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-sidebar-foreground">
          <Phone className="h-4 w-4" />
          Phone Dialer
        </span>
        <ChevronDown className={cn(
          "h-4 w-4 text-muted-foreground transition-transform",
          isExpanded && "rotate-180"
        )} />
      </button>
      
      {isExpanded && (
        <div className="p-3 pt-0 space-y-3 animate-fade-in">
          {/* Input Field */}
          <div className="relative">
            <Input
              value={dialInput}
              onChange={(e) => setDialInput(e.target.value)}
              placeholder="Phone, IP, or #room-id"
              className="text-center text-lg font-mono bg-sidebar-accent border-sidebar-border pr-10"
            />
            {dialInput && (
              <button
                onClick={handleBackspace}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                ←
              </button>
            )}
          </div>

          {/* Input Type Hints */}
          <div className="flex justify-center gap-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Hash className="h-3 w-3" />
              <span>Phone</span>
            </div>
            <div className="flex items-center gap-1">
              <Globe className="h-3 w-3" />
              <span>IP/Room</span>
            </div>
          </div>

          {/* Dial Pad */}
          <div className="grid grid-cols-3 gap-2">
            {dialPad.flat().map((digit) => (
              <Button
                key={digit}
                variant="outline"
                size="sm"
                onClick={() => handleDialPress(digit)}
                className="h-10 text-lg font-medium hover:bg-primary/20"
              >
                {digit}
              </Button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-3 gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSaveAsContact}
              disabled={!dialInput.trim()}
              className="h-10"
            >
              <User className="h-4 w-4" />
            </Button>
            
            {isInCall ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleEndCall}
                disabled={isLoading}
                className="h-10 col-span-2"
              >
                <PhoneOff className="h-4 w-4 mr-2" />
                End Call
              </Button>
            ) : (
              <Button
                variant="default"
                size="sm"
                onClick={handleCall}
                disabled={isLoading || !dialInput.trim()}
                className="h-10 col-span-2"
              >
                <Phone className="h-4 w-4 mr-2" />
                {isLoading ? 'Calling...' : 'Call'}
              </Button>
            )}
          </div>

          {/* VoIP Status */}
          <p className="text-xs text-center text-muted-foreground">
            {isVoipAvailable === false 
              ? 'VoIP/Video calling coming soon!'
              : 'Enter a phone number or IP address'}
          </p>
        </div>
      )}
    </div>
  );
}
