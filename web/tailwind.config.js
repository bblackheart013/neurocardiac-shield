/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'nc-bg': '#0a0a0f',
        'nc-bg-secondary': '#12121a',
        'nc-bg-elevated': '#1a1a24',
        'nc-accent': '#6366f1',
        'nc-accent-light': '#818cf8',
        'nc-eeg': '#3b82f6',
        'nc-ecg': '#ef4444',
        'nc-hrv': '#10b981',
        'nc-risk-low': '#10b981',
        'nc-risk-medium': '#f59e0b',
        'nc-risk-high': '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.8s ease-out',
        'fade-in-up': 'fadeInUp 0.8s ease-out',
        'pulse-slow': 'pulse 4s ease-in-out infinite',
        'glow': 'glow 4s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        glow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.1)' },
          '50%': { boxShadow: '0 0 40px rgba(99, 102, 241, 0.2)' },
        },
      },
    },
  },
  plugins: [],
}
