// AI Social Studio v1.0 Service Worker
const CACHE_NAME = 'ai-social-studio-v1';
self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
    // Pass-through network requests to live API
    event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});
