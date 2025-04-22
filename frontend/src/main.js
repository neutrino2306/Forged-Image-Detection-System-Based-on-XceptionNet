import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import i18n from './i18n'

// 创建 Vue 应用
const app = createApp(App)

// 安装 i18n 插件
app.use(i18n)

// 挂载 Vue 应用到 DOM
app.mount('#app')