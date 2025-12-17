import { useRef, useEffect } from 'react';
import { useSarahStore } from '@/stores/useSarahStore';
import { ChatMessage } from './ChatMessage';
import { ChatComposer } from './ChatComposer';
import { TypingIndicator } from './TypingIndicator';
import { api } from '@/lib/api';

export function ChatPanel() {
  const { messages, isTyping, addMessage, setTyping } = useSarahStore();
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  // Handle follow-up suggestions
  const handleSendFollowUp = async (text: string) => {
    addMessage({
      role: 'user',
      content: text,
    });

    setTyping(true);

    try {
      const messageHistory = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));
      
      messageHistory.push({
        role: 'user' as const,
        content: text,
      });

      const response = await api.chat.sendMessage(messageHistory);
      
      setTyping(false);
      
      if (!response.error) {
        addMessage({
          role: 'assistant',
          content: response.content,
        });
      }
    } catch (error) {
      console.error('Follow-up error:', error);
      setTyping(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Messages Area - ONLY scrollable element */}
      <div 
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-3 sm:px-4 py-2 sm:py-4"
      >
        <div className="max-w-3xl mx-auto space-y-3 sm:space-y-4">
          {messages.map((message) => (
            <ChatMessage 
              key={message.id} 
              message={message} 
              onSendFollowUp={handleSendFollowUp}
            />
          ))}
          {isTyping && <TypingIndicator />}
        </div>
      </div>

      {/* Composer - Fixed at bottom, never scrolls */}
      <ChatComposer />
    </div>
  );
}
