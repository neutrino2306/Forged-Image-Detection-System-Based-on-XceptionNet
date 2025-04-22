/** @type {import('tailwindcss').Config} */
export default {
  mode: 'jit',
  // content: [],
  content: [
    "./src/**/*.{vue,js,ts,jsx,tsx}", // 这里指定 Tailwind 扫描的文件
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}


