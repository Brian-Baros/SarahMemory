import sarahIcon from '@/assets/sarah-icon.ico';

export function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full overflow-hidden shrink-0 ring-2 ring-primary/40 animate-pulse-glow">
        <img src={sarahIcon} alt="Sarah AI" className="w-full h-full object-cover" />
      </div>
      
      <div className="bubble-assistant px-4 py-3">
        <div className="flex gap-1.5">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
