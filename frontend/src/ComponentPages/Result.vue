<script setup>
import {ref, onMounted} from "vue";
import { useI18n } from 'vue-i18n'
import { imageSrc, detectionResult } from '../Store/imageStore.js';
import {logoutUser, userState} from "@/Store/userState.js";
import http from "@/Network/request.js";

const emit = defineEmits(['navigate']);
const currentPage = ref('DetectPage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const { locale } = useI18n()

// console.log('imageSrc in Result page:', imageSrc.value);
// console.log('detectionResult in Result page:', detectionResult.isFake);


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
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'HomePage');
};

const navigateToDetectPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'DetectPage');
};

const navigateToAboutPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'AboutPage');
};

const navigateToLoginPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'LoginPage');
};

const navigateToUserInfoPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'UserInfoPage');
};

const navigateToHistoryPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
    emit('navigate', 'HistoryPage');
};

const navigateToRegisterPage = () => {
    imageSrc.value = null;  // 清空图片URL
    detectionResult.isFake = null;  // 可选：同时清空检测结果
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


    <div class="flex flex-col items-center justify-center min-h-screen">
        <!-- 图片预览区域 -->
<!--        <div v-if="imageSrc" class="rounded-lg shadow-lg">-->
<!--            <img :src="imageSrc" alt="Uploaded Image" class="w-[500px] h-72 rounded-lg justify-center mt-16 ml-60"/>-->
<!--        </div>-->

        <div v-if="imageSrc && detectionResult" class="flex items-center mt-16 ml-60 shadow-2xl">
            <!-- 图片预览区域 -->
            <div class="rounded-lg flex-shrink-0">
                <img :src="imageSrc" alt="Uploaded Image" class="w-[500px] h-72 rounded-lg shadow-2xl"/>
            </div>
            <!-- 检测结果显示区域 -->
<!--            <div class="ml-20 py-3 px-6 bg-cyan-400 rounded-full shadow-2xl" style="min-width: 150px;">-->
<!--                <p class="font-bold text-2xl text-center text-white">{{ detectionResult.isFake ? $t('message.fake') : $t('message.authentic') }}</p>-->
<!--            </div>-->
            <div class="flex flex-col ml-20 space-y-16">
                <!-- 检测结果展示区域 -->
                <div class="w-[150px] py-3 bg-cyan-400 rounded-full text-center shadow-2xl" style="align-self: start;">
                    <p class="font-bold text-2xl text-center text-white">
                        {{ detectionResult.isFake ? $t('message.fake') : $t('message.authentic') }}
                    </p>
                </div>
                <!-- 返回按钮 -->
                <div class="w-[150px] py-3 bg-cyan-400 hover:bg-opacity-85 rounded-full text-center shadow-2xl">
                    <button @click="navigateToDetectPage" class="font-bold text-2xl text-center text-white">
                        {{ $t('message.return') }}
                    </button>
                </div>
            </div>
        </div>

        <div v-else class="flex justify-center items-center absolute top-0 left-0 right-0 bottom-0 shadow-2xl">
            <div class="px-4 py-3 bg-white rounded-full shadow-2xl min-w-[300px] bg-opacity-80">
                <p class="font-bold text-4xl text-center">{{ $t('message.noDetectionYet') }}</p>
            </div>
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