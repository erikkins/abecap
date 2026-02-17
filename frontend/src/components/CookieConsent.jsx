import React, { useState, useEffect } from 'react';

const CONSENT_KEY = 'rigacap_cookie_consent';
const GA4_ID = 'G-0QKQRXTFSX';

function loadGA4() {
  if (window.gtag) return;
  const script = document.createElement('script');
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA4_ID}`;
  document.head.appendChild(script);
  window.dataLayer = window.dataLayer || [];
  function gtag() { window.dataLayer.push(arguments); }
  window.gtag = gtag;
  gtag('js', new Date());
  gtag('config', GA4_ID);
}

export default function CookieConsent() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem(CONSENT_KEY);
    if (consent === 'accepted') {
      loadGA4();
    } else if (!consent) {
      setVisible(true);
    }
  }, []);

  const accept = () => {
    localStorage.setItem(CONSENT_KEY, 'accepted');
    loadGA4();
    setVisible(false);
  };

  const decline = () => {
    localStorage.setItem(CONSENT_KEY, 'declined');
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div className="fixed bottom-0 inset-x-0 z-50 bg-blue-950/95 backdrop-blur-sm border-t border-blue-800/50 px-4 py-3">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        <p className="text-sm text-blue-200 text-center sm:text-left">
          We use cookies for analytics to improve your experience.{' '}
          <a href="/privacy" className="text-amber-400 hover:text-amber-300 underline">
            Privacy Policy
          </a>
        </p>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={decline}
            className="px-4 py-1.5 text-sm text-blue-300 hover:text-white border border-blue-700 hover:border-blue-500 rounded-lg transition-colors"
          >
            Decline
          </button>
          <button
            onClick={accept}
            className="px-4 py-1.5 text-sm text-blue-950 bg-amber-400 hover:bg-amber-300 font-medium rounded-lg transition-colors"
          >
            Accept
          </button>
        </div>
      </div>
    </div>
  );
}
