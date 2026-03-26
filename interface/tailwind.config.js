/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        omnex: {
          bg:          '#050507',
          surface:     '#0a0a0f',
          'surface-2': '#0f0f18',
          'surface-3': '#14141f',
          border:      '#1a1a2e',
          'border-2':  '#252540',
          'border-3':  '#30305a',
          text:        '#e8e8f0',
          'text-2':    '#a0a0b8',
          muted:       '#505068',
          'muted-2':   '#383850',
          accent:      '#7c6af7',
          'accent-2':  '#a78bfa',
          'accent-hover': '#9b8fff',
          'accent-glow':  'rgba(124,106,247,0.15)',
          green:  '#34d399',
          red:    '#f87171',
          amber:  '#fbbf24',
          blue:   '#60a5fa',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'fade-in':     'fadeIn 0.2s ease-out',
        'slide-up':    'slideUp 0.3s cubic-bezier(0.16,1,0.3,1)',
        'slide-in':    'slideIn 0.3s cubic-bezier(0.16,1,0.3,1)',
        'scale-in':    'scaleIn 0.2s cubic-bezier(0.16,1,0.3,1)',
        'glow-pulse':  'glowPulse 3s ease-in-out infinite',
        'ambient':     'ambientPulse 4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:       { from: { opacity: 0 },                       to: { opacity: 1 } },
        slideUp:      { from: { opacity: 0, transform: 'translateY(12px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        slideIn:      { from: { opacity: 0, transform: 'translateX(16px)' }, to: { opacity: 1, transform: 'translateX(0)' } },
        scaleIn:      { from: { opacity: 0, transform: 'scale(0.96)' },      to: { opacity: 1, transform: 'scale(1)' } },
        glowPulse:    {
          '0%,100%': { boxShadow: '0 0 0 0 rgba(124,106,247,0)' },
          '50%':     { boxShadow: '0 0 30px 4px rgba(124,106,247,0.2)' },
        },
        ambientPulse: {
          '0%,100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%':     { opacity: '0.7', transform: 'scale(1.05)' },
        },
      },
      boxShadow: {
        'accent-sm': '0 0 0 1px rgba(124,106,247,0.4)',
        'accent':    '0 0 0 1px rgba(124,106,247,0.3), 0 0 24px rgba(124,106,247,0.12)',
        'card':      '0 1px 3px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02)',
        'card-hover':'0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04)',
        'panel':     '0 0 0 1px rgba(26,26,46,1), 0 24px 64px rgba(0,0,0,0.8)',
        'glow-sm':   '0 0 16px rgba(124,106,247,0.25)',
        'glow':      '0 0 32px rgba(124,106,247,0.3)',
      },
    },
  },
  plugins: [],
}
