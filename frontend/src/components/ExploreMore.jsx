import React from 'react';
import { Link } from 'react-router-dom';
import { logPublicEvent } from '../lib/publicEvent';

// A soft exploration band for the ad-landing doors (/should-i-sell, /momentum). A lander who
// isn't ready for the hard CTA otherwise dead-ends — this hands them the deep site (the rich
// track-record / methodology / adviser / newsletter pages that do the real selling) instead of
// losing them. Each click fires a cookieless `explore_*` event so we can finally SEE whether
// landers explore, and which page pulls them deeper.

const LINKS = [
  { to: '/track-record', k: 'track', h: 'The track record',
    p: 'Every number, unfiltered — walk-forward, survivorship-free, straight through the crashes.' },
  { to: '/methodology', k: 'method', h: 'How it works',
    p: 'The discipline engine — entries, exits, and the 7-regime filter, explained in plain language.' },
  { to: '/for-advisers', k: 'advisers', h: 'For advisers',
    p: 'Run it as a sleeve — the behavioral case, the risk metrics, and client-ready one-pagers.' },
  { to: '/blog', k: 'blog', h: 'Market, Measured',
    p: 'Our read on the market — the thinking behind the signals, in writing, every week.' },
];

export default function ExploreMore({ heading }) {
  return (
    <section className="py-16">
      <div className="max-w-3xl mx-auto px-4 sm:px-8">
        <div className="flex items-center gap-3 mb-5">
          <span className="inline-block w-6 h-px bg-claret" />
          <span className="font-body text-[0.78rem] font-medium tracking-[0.18em] uppercase text-ink-mute">
            Look closer
          </span>
        </div>
        <h2 className="font-display text-[1.7rem] sm:text-[2.1rem] font-medium leading-[1.15] tracking-tight text-ink"
            style={{ fontVariationSettings: '"opsz" 48' }}>
          {heading || 'Not ready to start? The rest of the story is worth your time.'}
        </h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {LINKS.map((l) => (
            <Link key={l.k} to={l.to} onClick={() => logPublicEvent('explore_' + l.k)}
              className="group block border border-rule rounded bg-paper-card p-6 no-underline hover:border-claret transition-colors">
              <h3 className="font-display text-[1.15rem] font-medium text-ink mb-1 group-hover:text-claret transition-colors"
                  style={{ fontVariationSettings: '"opsz" 32' }}>
                {l.h} <span className="text-claret">&rarr;</span>
              </h3>
              <p className="text-[0.96rem] text-ink-mute leading-[1.5]">{l.p}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
