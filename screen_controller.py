import os
import time
import json
from pathlib import Path
from config import BASE_DIR
from database import log_event

class KickScreenController:
    """
    Browser & Screen Controller agent.
    Automates Kick.com browsing, stream scrubbing, and TikTok/Social uploading.
    Uses persistent real Chrome profile so login session is preserved forever.
    """
    def __init__(self):
        self.user_data_dir = str(BASE_DIR / "chrome_user_data")
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.pw = None
        self.context = None
        self.page = None

    def _launch_persistent(self, headless: bool = False):
        from playwright.sync_api import sync_playwright
        self.pw = sync_playwright().start()

        # Prioritize system Chrome -> Edge -> Playwright Chromium
        for ch in ["chrome", "msedge", None]:
            try:
                kwargs = {
                    "user_data_dir": self.user_data_dir,
                    "headless": headless,
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox"
                    ]
                }
                if ch:
                    kwargs["channel"] = ch
                context = self.pw.chromium.launch_persistent_context(**kwargs)
                log_event("INFO", "ScreenController", f"Launched browser using channel: {ch or 'chromium'}")
                return context
            except Exception as e:
                continue

        raise RuntimeError("No compatible browser found. Please make sure Google Chrome or Edge is installed.")

    def open_stream_in_browser(self, username: str, headless: bool = False):
        try:
            self.context = self._launch_persistent(headless=headless)
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            
            url = f"https://kick.com/{username.lower()}"
            log_event("INFO", "ScreenController", f"Navigating to {url}")
            self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
            
            # Dismiss cookies banner if present
            try:
                self.page.click("button:has-text('Accept')", timeout=4000)
            except Exception:
                pass
                
            return True
        except Exception as e:
            log_event("ERROR", "ScreenController", f"Browser launch error: {e}")
            return False

    def scrub_player(self, seconds_offset: int = -30):
        if not self.page:
            return False
        try:
            self.page.click("video", timeout=3000)
            press_count = max(1, abs(seconds_offset) // 5)
            key = "ArrowLeft" if seconds_offset < 0 else "ArrowRight"
            for _ in range(press_count):
                self.page.keyboard.press(key)
                time.sleep(0.1)
            log_event("INFO", "ScreenController", f"Scrubbed stream by {seconds_offset} seconds")
            return True
        except Exception as e:
            log_event("WARNING", "ScreenController", f"Player scrub failed: {e}")
            return False

    def upload_to_tiktok(self, video_path: str, caption: str, tags: str = "", headless: bool = False) -> bool:
        """
        Automates publishing to TikTok Studio:
        1. Opens real Chrome window with persistent profile.
        2. Navigates to TikTok Studio upload page.
        3. Prompts user to log in if not already logged in (login persists in chrome_user_data).
        4. Staged video file into upload input.
        5. Fills caption and hashtags.
        6. User can review and click Post directly in the open Chrome window!
        """
        resolved_video_path = str(Path(video_path).resolve())
        if not os.path.exists(resolved_video_path):
            log_event("ERROR", "ScreenController", f"Video file not found: {resolved_video_path}")
            return False

        log_event("INFO", "ScreenController", f"Opening TikTok Studio for upload: {resolved_video_path}")
        try:
            context = self._launch_persistent(headless=headless)
            page = context.pages[0] if context.pages else context.new_page()

            target_url = "https://www.tiktok.com/tiktokstudio/upload"
            log_event("INFO", "ScreenController", f"Navigating to {target_url}")
            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            time.sleep(4)

            # Check if login is needed
            current_url = page.url.lower()
            if "login" in current_url:
                log_event("WARNING", "ScreenController", "TikTok login required! Please log in to your TikTok account in the opened Chrome window. Your login will be saved forever.")
                try:
                    # Wait up to 3 minutes for user to complete login in the opened browser window
                    page.wait_for_url(lambda u: "upload" in u.lower() or "tiktokstudio" in u.lower(), timeout=180000)
                    log_event("INFO", "ScreenController", "Login detected! Continuing with video upload...")
                    time.sleep(5)
                except Exception:
                    log_event("WARNING", "ScreenController", "Login wait timed out.")
                    return False

            # Locate file input element (supports direct and iframe layouts)
            file_input = None
            for selector in ["input[type='file']", "input[accept*='video']"]:
                try:
                    loc = page.locator(selector).first
                    if loc.count() > 0:
                        file_input = loc
                        break
                except Exception:
                    pass

            if not file_input:
                for frame in page.frames:
                    try:
                        loc = frame.locator("input[type='file']").first
                        if loc.count() > 0:
                            file_input = loc
                            break
                    except Exception:
                        pass

            if file_input:
                log_event("INFO", "ScreenController", f"Selecting video file: {resolved_video_path}")
                file_input.set_input_files(resolved_video_path)
                time.sleep(5)
            else:
                log_event("WARNING", "ScreenController", "Could not automatically locate file input element. Upload manually via the opened window.")

            # Enter caption and hashtags
            full_caption = f"{caption} {tags}".strip()
            for selector in [
                "div[contenteditable='true']",
                ".public-DraftEditor-content",
                "[data-placeholder*='caption']",
                "textarea[placeholder*='caption']"
            ]:
                try:
                    field = page.locator(selector).first
                    if field.count() > 0 and field.is_visible():
                        field.click()
                        field.fill(full_caption)
                        log_event("INFO", "ScreenController", "Filled TikTok caption and hashtags successfully!")
                        break
                except Exception:
                    continue

            log_event("INFO", "ScreenController", "Video staged in TikTok Studio! You can review details and click Post in the Chrome window.")
            return True

        except Exception as e:
            log_event("ERROR", "ScreenController", f"TikTok publishing error: {e}")
            return False

    def close(self):
        try:
            if self.context:
                self.context.close()
            if self.pw:
                self.pw.stop()
        except Exception:
            pass
