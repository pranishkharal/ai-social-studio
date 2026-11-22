// AI Social Studio v2.0 - Ultra-Lite Progressive Web App Service Worker
// Complies with PRD Phase 4: Shell caching, zero private video caching, offline resilience

const CACHE_NAME = 'ai-social-studio-v2';
const APP_SHELL = [
  '/',
  '/static/manifest.json',
  '/static/app_icon.ico',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/preview_frame.jpg'
];

// Pre-cache core application shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(APP_SHELL).catch((err) => {
        console.warn('[SW] App shell precache notice:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

// Purge obsolete caches on activation
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch handler: Network-only for private data/video, Cache-first for shell assets
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // 1. NEVER cache private video streams, raw clips, or real-time event streams
  if (url.pathname.startsWith('/video/') || url.pathname.startsWith('/api/stream/')) {
    return; // Pass directly to native network
  }

  // 2. API calls: Network-first with graceful offline fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request).catch(async () => {
        const cachedResp = await caches.match(event.request);
        if (cachedResp) return cachedResp;
        return new Response(JSON.stringify({ offline: true, message: "Network offline. Reconnecting..." }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }

  // 3. Static assets & App Shell: Stale-While-Revalidate for instant render
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const fetchPromise = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch(() => cachedResponse);

      return cachedResponse || fetchPromise;
    })
  );
});
