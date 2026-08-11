// Cookieless, first-party funnel/engagement event beacon.
//
// Sibling to PageViewBeacon: posts to the SAME /api/public/hit endpoint (no cookie,
// no persistent ID, no PII — aggregate counts only), but carries an `event` name so
// pre-signup landing-page funnels are visible in the internal traffic report. This
// is the path that survives cookie-consent denial (unlike eventLogger.js, which
// hard-drops when logged out). Fire-and-forget; never throws.
//
// Usage:  import { logPublicEvent } from './lib/publicEvent';
//         logPublicEvent('cta_trial');            // uses current path + URL utm/gclid
//         logPublicEvent('checkout_redirect', { path: '/should-i-sell', utm_campaign: 'preserve' });

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function logPublicEvent(event, opts = {}) {
  if (!event) return;
  try {
    const p = new URLSearchParams(window.location.search);
    fetch(`${API_BASE}/api/public/hit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      keepalive: true,  // survives unload (bounce/exit events)
      body: JSON.stringify({
        path: opts.path || window.location.pathname,
        event,
        ref: document.referrer || null,
        utm_source: opts.utm_source ?? p.get('utm_source'),
        utm_medium: opts.utm_medium ?? p.get('utm_medium'),
        utm_campaign: opts.utm_campaign ?? p.get('utm_campaign'),
        gclid: opts.gclid ?? p.get('gclid'),
      }),
    }).catch(() => {});
  } catch (_) {
    /* never break the app over analytics */
  }
}

// Stash the ad-origin (path + campaign) when a landing visitor commits to signup,
// so the Stripe redirect — which fires later, on /app, after auth — can still be
// attributed back to the landing page that drove it. Ephemeral (sessionStorage),
// per-tab, no cross-site ID.
const ORIGIN_KEY = 'rigacap_ad_origin';

export function stashAdOrigin() {
  try {
    const p = new URLSearchParams(window.location.search);
    sessionStorage.setItem(ORIGIN_KEY, JSON.stringify({
      path: window.location.pathname,
      utm_source: p.get('utm_source'),
      utm_medium: p.get('utm_medium'),
      utm_campaign: p.get('utm_campaign'),
      gclid: p.get('gclid'),
    }));
  } catch { /* ignore */ }
}

// Read + clear the stashed origin (one-shot). Returns null if none.
export function consumeAdOrigin() {
  try {
    const raw = sessionStorage.getItem(ORIGIN_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(ORIGIN_KEY);
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
