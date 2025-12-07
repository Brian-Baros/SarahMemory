# SarahMemory Mobile v1.0.0 (Kivy)
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
import json, os
try:
    import requests
except Exception:
    requests = None

API_BASE = os.environ.get("SARAHMOBILE_API_BASE", "https://api.sarahmemory.com")
API_KEY  = os.environ.get("SARAHMOBILE_API_KEY", "")

def _headers():
    h = {"Content-Type":"application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return h

class Chat(BoxLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.orientation = "vertical"
        self.inp = TextInput(size_hint=(1,0.35), multiline=True, hint_text="Say hi to Sarah…")
        self.btn = Button(text="Send", size_hint=(1,0.2))
        self.out = Label(text="", size_hint=(1,0.45))
        self.btn.bind(on_press=self.on_send)
        self.add_widget(self.inp); self.add_widget(self.btn); self.add_widget(self.out)

    def on_send(self, *_):
        text = (self.inp.text or "").strip()
        if not text: return
        # local calculator fast-path
        try:
            cleaned = text.lower().replace("what is","").replace("=","").strip()
            if cleaned and all(c in "0123456789+-*/(). %" for c in cleaned):
                res = eval(cleaned, {"__builtins__":{}}, {})
                self.out.text = f"{text} = {res}\n[Source: Calculator]"
                return
        except Exception:
            pass
        # hub logging so mobile stays in sync
        if not requests:
            self.out.text = "Network lib not available in this build.\n[Source: Local]"
            return
        try:
            payload = {"node_id":"mobile-client", "text": text, "tags":["mobile"]}
            r = requests.post(API_BASE + "/api/context-update", headers=_headers(), data=json.dumps(payload), timeout=8)
            self.out.text = "Got it. I logged your message to the hub.\n[Source: Web]" if r.status_code == 200 else f"Hub error: {r.status_code}\n[Source: Web]"
        except Exception as e:
            self.out.text = f"Network error: {e}\n[Source: Web]"

class SarahMemoryMobile(App):
    def build(self):
        self.title = "SarahMemory"
        return Chat()

if __name__ == "__main__":
    SarahMemoryMobile().run()
