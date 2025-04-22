<script setup>
import {ref} from "vue";
import { useI18n } from 'vue-i18n'
import {imageSrc} from "@/Store/imageStore.js";
import {logoutUser, userState} from "@/Store/userState.js";
import http from "@/Network/request.js";

const emit = defineEmits(['navigate']);
const currentPage = ref('AboutPage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const { locale } = useI18n()

// const words = ref([
//     { text: 'Vue.js', weight: 2 },
//     { text: 'XceptionNet', weight: 4 },
//     { text: 'TailwindCSS', weight: 1 },
//     { text: 'Deep Learning', weight: 2 },
//     { text: 'Flask', weight: 3 },
//     // 添加更多词汇...
// ]);
//
// const getRandomSize = (weight) => {
//     // 基于权重调整大小，这里简单示例：基本大小为text-sm，权重每增加1，大小增加一级
//     const baseSize = 40; // text-base对应的大小
//     const size = baseSize + weight * 10; // 简单计算示例
//     // return size + 'px'; // 返回计算后的字体大小
//     return size; // 直接返回计算后的字体大小
// };
//
// const getRandomColor = () => {
//     // 定义一组颜色的CSS值
//     const colors = ['#c1f6b6', '#6af1de', '#c5bdf1', '#eedc21','#8ddcfd'];
//     // 随机选择一种颜色
//     const randomIndex = Math.floor(Math.random() * colors.length);
//     return colors[randomIndex];
// };

// const handleLogout = () => {
//     logoutUser();  // 重置用户状态
//     navigateToHomePage();
// };
const handleLogout = async () => {
    try {
        // 向后端发送登出请求
        const response = await http.post('/logout');
        if (response.success) {
            // 登出成功，清除前端的用户状态
            logoutUser();
            navigateToHomePage();
        }
    } catch (error) {
        console.error('Logout error:', error);
    }
};

// 切换到中文
const switchToChinese = () => {
    locale.value = 'zh'
}

// 切换到英文
const switchToEnglish = () => {
    locale.value = 'en'
}
const navigateToHomePage = () => {
    emit('navigate', 'HomePage');
};

const navigateToDetectPage = () => {
    emit('navigate', 'DetectPage');
};

const navigateToUserInfoPage = () => {
    emit('navigate', 'UserInfoPage');
};

const navigateToHistoryPage = () => {
    emit('navigate', 'HistoryPage');
};

const navigateToAboutPage = () => {
    emit('navigate', 'AboutPage');
};

const navigateToLoginPage = () => {
    emit('navigate', 'LoginPage');
};

const navigateToRegisterPage = () => {
    emit('navigate', 'RegisterPage');
};


</script>

<template>
    <nav class="fixed top-6 left-1/2 transform -translate-x-1/2 bg-white bg-opacity-50 shadow-2xl z-50 min-w-[700px] rounded-xl">
        <ul class="flex justify-between px-4 py-6" style="min-width: 950px;">
            <li class="pl-16">
                <button @click="navigateToHomePage" :class="{'active': currentPage === 'HomePage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.home') }}</button>
            </li>
            <li>
                <button @click="navigateToDetectPage" :class="{'active': currentPage === 'DetectPage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.detect') }}</button>
            </li>
            <li>
                <button @click="navigateToAboutPage" :class="{'active': currentPage === 'AboutPage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.about') }}</button>
            </li>
            <li class="relative group pr-16">
                <div class="flex flex-col items-center">
                    <div class="p-4 -m-4 bg-transparent">
                        <button
                            class="font-bold text-xl px-4 py-2 bg-blue-500 text-black hover:text-green-800 group-hover:text-green-800 hover:underline group-hover:underline decoration-green-800 decoration-2 underline-offset-4"
                            :class="{'active': currentPage === 'HistoryDetailPage' || currentPage === 'HistoryPage' || currentPage === 'UserInfoPage'}">{{ $t('message.user') }}
                        </button>
                    </div>
                    <div class="absolute left-1/4 transform -translate-x-1/2 mt-8 w-36 bg-white bg-opacity-75 shadow-md invisible group-hover:opacity-100 group-hover:visible transition-opacity rounded-xl">
                        <ul v-if="userState.isLoggedIn" class="flex flex-col items-center w-full">
                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-tl-xl rounded-tr-xl" @click="navigateToUserInfoPage">{{ $t('message.userInfo') }}</li>
                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center" @click="navigateToHistoryPage">{{ $t('message.userHistory') }}</li>
                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-bl-xl rounded-br-xl" @click="handleLogout">{{ $t('message.logOut') }}</li>
                        </ul>
                        <ul v-else class="flex flex-col items-center w-full">
                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-tl-xl rounded-tr-xl" @click="navigateToLoginPage">{{ $t('message.signIn') }}</li>
                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-bl-xl rounded-br-xl" @click="navigateToRegisterPage">{{ $t('message.signUp') }}</li>
                        </ul>
                    </div>
                </div>
            </li>
        </ul>
    </nav>

<!--    <div class="mt-36 w-[980px] text-center ml-16 flex flex-wrap justify-center">-->
<!--        <div v-for="(word, index) in words" :key="index" :style="{ fontSize: getRandomSize(word.weight) + 'px', color: getRandomColor(), fontWeight: 'bold'  }" class="inline-block mx-4 my-3 text-shadow">-->
<!--            {{ word.text }}-->
<!--        </div>-->
<!--    </div>-->

    <div class="mt-36 ml-72 flex flex-col items-center">
        <!-- 标题部分 -->
        <div class="text-black font-bold text-3xl text-center mb-8">
            {{ $t('message.title') }}
        </div>

        <!-- 白色框部分 -->
        <div class="bg-white bg-opacity-80 w-[500px] h-64 rounded-lg flex justify-center items-center relative shadow-2xl">
            <p class="text-left text-xl p-4">
                {{ $t('message.introduction') }}
            </p>
        </div>
    </div>

    <div class="absolute bottom-7 left-16 flex items-center">
        <button @click="switchToChinese" class="font-bold text-mid text-center text-white">{{ $t('message.zh') }}</button>
        <span class="mx-2 text-white">/</span>
        <button @click="switchToEnglish" class="font-bold text-mid text-center text-white">{{ $t('message.en') }}</button>
    </div>

</template>

<style scoped>
button {
    background: none; /* 移除按钮默认的背景颜色 */
    border: none; /* 移除按钮默认的边框 */
    padding: 0; /* 移除按钮的内边距 */
    margin: 0; /* 移除按钮的外边距 */
    cursor: pointer; /* 添加鼠标指针样式 */
}
/* 修复按钮在点击后保留轮廓的问题 */
button:focus {
    outline: none;
}
.active {
    color: #22543D; /* TailwindCSS的green-600 */
    text-decoration: underline;
}
</style>