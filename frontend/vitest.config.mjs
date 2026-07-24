import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.js"],
    css: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      // Stage-by-stage: we'll widen this as each stage lands.
      // Starting narrow so CI coverage % reflects what's actually tested,
      // not a false "0% of everything" or misleading aggregate.
      include: [
        "config/**/*.{js,jsx}",
        "lib/**/*.{js,jsx}",
      ],
    },
  },
});