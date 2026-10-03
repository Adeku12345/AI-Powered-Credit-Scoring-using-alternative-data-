/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0B1220",
          800: "#121B2E",
          700: "#182238",
          600: "#2A3752",
        },
        paper: "#ECEADF",
        muted: "#8C96AC",
        brass: "#C9A15A",
        good: "#5FA777",
        warn: "#D98B3B",
        bad: "#C1584B",
      },
      fontFamily: {
        serif: ["'Source Serif 4'", "ui-serif", "Georgia", "serif"],
        sans: ["'IBM Plex Sans'", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
}
