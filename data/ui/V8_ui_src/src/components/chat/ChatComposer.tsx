import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Mic, Paperclip, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSarahStore } from '@/stores/useSarahStore';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';
import { toast } from 'sonner';

// Web Speech API types
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  onresult: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionErrorEvent) => void) | null;
  onend: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

export function ChatComposer() {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  
  const { messages, addMessage, setTyping, mediaState, settings } = useSarahStore();

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  const handleSubmit = async () => {
    if (!message.trim() && selectedFiles.length === 0) return;
    if (isLoading) return;

    const userMessage = message.trim();
    
    // Add user message
    addMessage({
      role: 'user',
      content: userMessage,
    });

    setMessage('');
    setSelectedFiles([]);
    setTyping(true);
    setIsLoading(true);

    // Notify avatar that user is sending
    try {
      await api.avatar.setListening(true);
    } catch (e) {
      // Silent - avatar state is not critical
    }

    try {
      // Build message history for context
      const messageHistory = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));
      
      // Add the new user message
      messageHistory.push({
        role: 'user' as const,
        content: userMessage,
      });

      // Send to backend
      const response = await api.chat.sendMessage(messageHistory);
      
      setTyping(false);
      
      if (response.error) {
        toast.error(response.error);
        addMessage({
          role: 'assistant',
          content: "I'm sorry, I encountered an error. Please try again.",
        });
      } else {
        addMessage({
          role: 'assistant',
          content: response.content,
        });
        
        // If voice is enabled and we got audio, play it
        if (mediaState.voiceEnabled && settings.autoSpeak) {
          await speakResponse(response.content);
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setTyping(false);
      
      toast.error('Failed to send message. Please try again.');
      addMessage({
        role: 'assistant',
        content: "I'm having trouble connecting right now. Please try again in a moment.",
      });
    } finally {
      setIsLoading(false);
      // Reset avatar state
      try {
        await api.avatar.setListening(false);
      } catch (e) {
        // Silent
      }
    }
  };

  const speakResponse = async (text: string) => {
    try {
      // Notify avatar that speaking is starting
      await api.avatar.setSpeaking(true);
      
      // Use the selected voice from settings
      const response = await api.voice.speak(text, settings.selectedVoice);
      
      if (response.success && (response.audio_url || response.audio_base64)) {
        const audioSrc = response.audio_url || 
          (response.audio_base64 ? `data:audio/mp3;base64,${response.audio_base64}` : null);
        
        if (audioSrc) {
          const audio = new Audio(audioSrc);
          
          audio.onended = async () => {
            try {
              await api.avatar.setSpeaking(false);
            } catch (e) {
              // Silent
            }
          };
          
          audio.onerror = async () => {
            try {
              await api.avatar.setSpeaking(false);
            } catch (e) {
              // Silent
            }
          };
          
          await audio.play();
          return;
        }
      }
      
      // Fallback to browser TTS
      if (response.fallback || !response.success) {
        useBrowserTTS(text);
      }
    } catch (error) {
      console.error('TTS error:', error);
      useBrowserTTS(text);
    }
  };

  const useBrowserTTS = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Get voices (may need to wait for them to load)
      let voices = speechSynthesis.getVoices();
      
      // If voices aren't loaded yet, wait for them
      if (voices.length === 0) {
        speechSynthesis.onvoiceschanged = () => {
          voices = speechSynthesis.getVoices();
          setVoiceAndSpeak(utterance, voices);
        };
      } else {
        setVoiceAndSpeak(utterance, voices);
      }
    }
  };

  const setVoiceAndSpeak = (utterance: SpeechSynthesisUtterance, voices: SpeechSynthesisVoice[]) => {
    // Default to female voice for Sarah AI
    const femaleKeywords = ['female', 'samantha', 'victoria', 'karen', 'zira', 'susan', 'hazel', 'fiona', 'moira', 'tessa', 'kate'];
    const englishVoices = voices.filter(v => v.lang.startsWith('en'));
    
    // Find a female voice
    let selectedVoice = englishVoices.find(v => 
      femaleKeywords.some(keyword => v.name.toLowerCase().includes(keyword))
    );
    
    // If no obvious female voice, try to avoid obvious male voices
    if (!selectedVoice) {
      const maleKeywords = ['male', 'david', 'daniel', 'james', 'alex', 'tom', 'mark', 'fred', 'ralph'];
      selectedVoice = englishVoices.find(v => 
        !maleKeywords.some(keyword => v.name.toLowerCase().includes(keyword))
      );
    }
    
    // Fall back to first English voice
    if (!selectedVoice && englishVoices.length > 0) {
      selectedVoice = englishVoices[0];
    }
    
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    
    // Set slightly higher pitch for more feminine sound
    utterance.pitch = 1.1;
    utterance.rate = 0.95;
    
    utterance.onend = async () => {
      try {
        await api.avatar.setSpeaking(false);
      } catch (e) {
        // Silent
      }
    };
    
    speechSynthesis.speak(utterance);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(files);
    
    // Optionally analyze files immediately
    if (files.length > 0) {
      toast.info(`${files.length} file(s) selected. They will be analyzed when you send.`);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsRecording(false);
    } else {
      // Check for Web Speech API support
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      
      if (!SpeechRecognition) {
        toast.error('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
        return;
      }

      try {
        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
          setIsRecording(true);
          toast.info('Listening... Speak now');
        };

        recognition.onresult = async (event: any) => {
          const transcript = event.results[0][0].transcript;
          
          if (transcript.trim()) {
            // Set the message and auto-send
            setMessage(transcript);
            setIsRecording(false);
            
            // Auto-submit after a brief delay to show the text
            setTimeout(async () => {
              // Directly add and send the message
              addMessage({
                role: 'user',
                content: transcript,
              });

              setMessage('');
              setTyping(true);
              setIsLoading(true);

              try {
                await api.avatar.setListening(true);
              } catch (e) {
                // Silent
              }

              try {
                const messageHistory = messages.map(m => ({
                  role: m.role,
                  content: m.content,
                }));
                
                messageHistory.push({
                  role: 'user' as const,
                  content: transcript,
                });

                const response = await api.chat.sendMessage(messageHistory);
                
                setTyping(false);
                
                if (response.error) {
                  toast.error(response.error);
                  addMessage({
                    role: 'assistant',
                    content: "I'm sorry, I encountered an error. Please try again.",
                  });
                } else {
                  addMessage({
                    role: 'assistant',
                    content: response.content,
                  });
                  
                  if (mediaState.voiceEnabled && settings.autoSpeak) {
                    await speakResponse(response.content);
                  }
                }
              } catch (error) {
                console.error('Chat error:', error);
                setTyping(false);
                toast.error('Failed to send message');
              } finally {
                setIsLoading(false);
                try {
                  await api.avatar.setListening(false);
                } catch (e) {
                  // Silent
                }
              }
            }, 200);
          }
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          setIsRecording(false);
          
          if (event.error === 'not-allowed') {
            toast.error('Microphone access denied. Please allow microphone access.');
          } else if (event.error === 'no-speech') {
            toast.info('No speech detected. Try again.');
          } else {
            toast.error(`Speech recognition error: ${event.error}`);
          }
        };

        recognition.onend = () => {
          setIsRecording(false);
        };

        recognition.start();
      } catch (error) {
        console.error('Speech recognition error:', error);
        toast.error('Could not start speech recognition');
        setIsRecording(false);
      }
    }
  };

  return (
    <div className="shrink-0 border-t border-border bg-card/50 backdrop-blur-sm p-2 sm:p-3 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:pb-3">
      <div className="max-w-3xl mx-auto">
        {/* Selected Files Preview */}
        {selectedFiles.length > 0 && (
          <div className="flex gap-2 mb-3 flex-wrap">
            {selectedFiles.map((file, idx) => (
              <div 
                key={idx}
                className="px-3 py-1.5 bg-secondary rounded-full text-xs text-secondary-foreground flex items-center gap-2"
              >
                <Paperclip className="h-3 w-3" />
                {file.name}
                <button 
                  onClick={() => setSelectedFiles(files => files.filter((_, i) => i !== idx))}
                  className="hover:text-destructive"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Composer Row */}
        <div className="flex items-center gap-2 sm:gap-3">
          <Button 
            variant="ghost" 
            size="icon"
            onClick={toggleRecording}
            disabled={isLoading}
            className={cn(
              "shrink-0 h-9 w-9 sm:h-10 sm:w-10",
              isRecording 
                ? "text-destructive animate-pulse ring-2 ring-destructive bg-destructive/10" 
                : "text-muted-foreground hover:text-foreground"
            )}
            title={isRecording ? "Stop recording" : "Start voice input"}
          >
            <Mic className="h-5 w-5" />
          </Button>

          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            disabled={isLoading || isRecording}
            className="flex-1 bg-secondary border-border text-foreground placeholder:text-muted-foreground h-9 sm:h-10"
          />

          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
          />
          
          <Button 
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="shrink-0 text-muted-foreground hover:text-foreground h-9 w-9 sm:h-10 sm:w-10"
            title="Attach files"
          >
            <Paperclip className="h-5 w-5" />
          </Button>

          <Button 
            onClick={handleSubmit}
            disabled={(!message.trim() && selectedFiles.length === 0) || isLoading}
            size="icon"
            className="shrink-0 bg-primary text-primary-foreground hover:bg-primary/90 h-9 w-9 sm:h-10 sm:w-10"
            title="Send message"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}