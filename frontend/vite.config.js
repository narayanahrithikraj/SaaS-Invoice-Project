import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path' // Import the 'path' module

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // This line sets '@' to point to your 'src' folder
      '@': path.resolve(__dirname, './src'),
    },
  },
})

