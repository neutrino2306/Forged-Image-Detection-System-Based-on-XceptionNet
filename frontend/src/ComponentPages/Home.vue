<script setup>
import {ref} from "vue";
import { useI18n } from 'vue-i18n'
import { userState, loginUser, logoutUser } from "@/Store/userState";
import http from '@/Network/request.js';

const emit = defineEmits(['navigate']);
const currentPage = ref('HomePage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const { locale } = useI18n()

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
<!--            <li class="pl-16">-->
<!--                <button @click="navigateToHomePage" class="font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4">HOME</button>-->
<!--            </li>-->
<!--            <li>-->
<!--                <button @click="navigateToDetectPage" class="font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4">DETECT</button>-->
<!--            </li>-->
<!--            <li>-->
<!--                <button @click="navigateToResultPage" class="font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4">RESULT</button>-->
<!--            </li>-->
<!--            <li class="pr-16">-->
<!--                <button @click="navigateToAboutPage" class="font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4">ABOUT</button>-->
<!--            </li>-->
            <li class="pl-16">
                <button @click="navigateToHomePage" :class="{'active': currentPage === 'HomePage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.home') }}</button>
            </li>
            <li>
                <button @click="navigateToDetectPage" :class="{'active': currentPage === 'DetectPage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.detect') }}</button>
            </li>
<!--            <li>-->
<!--                <button @click="navigateToResultPage" :class="{'active': currentPage === 'ResultPage', 'font-bold text-xl hover:text-green-800 hover:underline decoration-green-800 decoration-2 underline-offset-4': true}">{{ $t('message.result') }}</button>-->
<!--            </li>-->
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
<!--                        <ul class="flex flex-col items-center w-full">-->
<!--                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-tl-xl rounded-tr-xl" @click="navigateToUserInfoPage">{{ $t('message.userInfo') }}</li>-->
<!--                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center" @click="navigateToHistoryPage">{{ $t('message.userHistory') }}</li>-->
<!--                            <li class="hover:bg-purple-200 cursor-pointer text-mid px-4 py-2 w-full text-center rounded-bl-xl rounded-br-xl" @click="navigateToLoginPage">{{ $t('message.logOut') }}</li>-->
<!--                        </ul>-->
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

    <div class="absolute top-40 left-32 flex items-center">
        <img src="/src/img/icon.jpg" alt="Icon" class="h-24 w-24 rounded-3xl mr-8">
        <div class="px-4 py-3 bg-white rounded-full shadow-2xl min-w-[300px] bg-opacity-80">
            <p class="font-bold text-4xl text-center">{{ $t('message.helloFriend') }}</p>
        </div>
    </div>

    <div class="absolute top-72 left-32">
        <div class="px-4 py-3 bg-white rounded-full shadow-2xl bg-opacity-80">
            <p class="font-bold text-4xl text-center">{{ $t('message.availableForWork') }}</p>
        </div>
    </div>

    <div class="absolute top-96 right-40">
        <div class="px-6 py-3 bg-cyan-400 hover:bg-opacity-75 rounded-full shadow-2xl">
            <button @click="navigateToDetectPage" class="font-bold text-2xl text-center text-white">{{ $t('message.letsBegin') }}</button>
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