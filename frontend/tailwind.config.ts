import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'cool-white': '#f9fafb',
        'slate-blue': '#64748b',
        'navy': '#1e293b',
        'primary-blue': '#2563eb',
        'indigo': '#4f46e5',
        'success-green': '#10b981',
      },
      fontFamily: {
        sans: ['system-ui', 'sans-serif'],
      },
      spacing: {
        '20': '20px',
        '24': '24px',
      },
      borderRadius: {
        'card': '14px',
        'input': '8px',
      },
      boxShadow: {
        'subtle': '0 1px 2px rgba(0, 0, 0, 0.05)',
        'card': '0 1px 3px rgba(0, 0, 0, 0.08)',
      },
      maxWidth: {
        'container': '1500px',
      },
    },
  },
  plugins: [],
}

export default config
