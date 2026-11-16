import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    print("\n🌐 Opening Kick AI Clip Engine Dashboard: http://localhost:8500\n")
    webbrowser.open("http://localhost:8500")

if __name__ == "__main__":
    print("=" * 65)
    print("  🚀 KICK AI CLIP ENGINE & AUTONOMOUS SOCIAL PUBLISHER")
    print("  🟢 Lightweight Ingestion: Direct AWS IVS Stream-Copy (0% Screen Strain)")
    print("  🧠 AI Narrative Engine: Gemini 2.5 Flash + Local Whisper")
    print("  📱 TikTok Studio: Auto 9:16 Vertical Crop, Subtitles & Hooks")
    print("=" * 65)
    
    # Open dashboard in default browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run FastAPI server
    uvicorn.run("web_dashboard:app", host="127.0.0.1", port=8500, log_level="info")
