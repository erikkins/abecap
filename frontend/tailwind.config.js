/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      animation: {
        'fade-in': 'fade-in 0.3s ease-out',
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        body: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'Menlo', 'monospace'],
      },
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        success: {
          50: '#ecfdf5',
          500: '#10b981',
          600: '#059669',
        },
        danger: {
          50: '#fef2f2',
          500: '#ef4444',
          600: '#dc2626',
        },
        paper: {
          DEFAULT: '#F5F1E8',
          deep: '#EDE7D8',
          card: '#FAF7F0',
        },
        ink: {
          DEFAULT: '#141210',
          mute: '#5A544E',
          light: '#8A8279',
        },
        claret: {
          DEFAULT: '#7A2430',
          light: '#9A3444',
        },
        rule: {
          DEFAULT: '#DDD5C7',
          dark: '#C9BFAC',
        },
        positive: '#2D5F3F',
        negative: '#8F2D3D',
      },
    },
  },
  plugins: [],
}
