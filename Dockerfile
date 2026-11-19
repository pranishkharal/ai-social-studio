# ai-social-studio — Kick live-clip engine (Docker image)
#
# Build & run:
#   cp .env.example .env   # then fill in GEMINI_API_KEY (and the alert/auth vars)
#   docker compose up -d --build
#   Dashboard: http://localhost:8500
#
# The Windows flow (start.bat / main.py) is unaffected by these files.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg (rolling stream buffer + clip rendering) and fonts (overlays /
# subtitles). fonts-liberation provides Arial-metric-compatible bold fonts,
# used as the fallback for the Windows Impact / Arial Bold overlay fonts
# (see resolve_display_font_path() in kick_program.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so they stay cached when app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data directories exist even when no volumes are mounted (plain `docker run`).
RUN mkdir -p buffer clips edited subtitles checkpoints overlays

EXPOSE 8500

# NOTE: TikTok upload uses Playwright with a persistent system Chrome/Edge
# profile and does not work inside this container as-is. To enable it,
# uncomment the line below (adds a browser + system deps) and mount a
# persistent chrome_user_data volume in docker-compose.yml:
# RUN playwright install --with-deps chromium

CMD ["uvicorn", "web_dashboard:app", "--host", "0.0.0.0", "--port", "8500"]
