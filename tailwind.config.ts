import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "oklch(24% 0.018 165)",
        moss: "oklch(55% 0.12 165)",
        grass: "oklch(73% 0.14 135)",
        clay: "oklch(64% 0.15 42)",
        gold: "oklch(68% 0.13 82)",
        mist: "oklch(95% 0.01 185)",
        line: "oklch(88% 0.012 190)",
      },
    },
  },
  plugins: [],
} satisfies Config;
