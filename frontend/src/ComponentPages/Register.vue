<script setup>
import { userState, loginUser, logoutUser } from "@/Store/userState";
import {ref} from "vue";
import {useI18n} from "vue-i18n";
const { locale } = useI18n()
import http from '@/Network/request.js';

const emit = defineEmits(['navigate']);
const username = ref('');
const password = ref('');
const { t } = useI18n();

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
const navigateToLoginPage = () => {
    emit('navigate', 'LoginPage');
};

const handleRegister = async () => {
    // 检查用户名和密码是否为空
    if (!username.value.trim() || !password.value.trim()) {
        alert(t('register.usernameRequired')); // 提示用户名和密码不能为空
        return;
    }

    try {
        const response = await http.post('/register', {
            username: username.value.trim(),
            password: password.value.trim()
        },{
            withCredentials: true
        });

        if (response.success) {
            // 注册成功，更新用户状态
            loginUser(username.value, password.value);
            navigateToHomePage();
            alert(t('register.success')); // 显示注册成功消息
        }
    } catch (error) {
        if (error.response && error.response.data.error) {
            // 根据后端返回的错误信息显示相应的提示
            switch (error.response.data.error) {
                case 'Username already exists':
                    alert(t('register.usernameExists'));
                    break;
                default:
                    alert(t('register.error'));
            }
        } else {
            console.error('register error:', error);
            alert(t('register.error')); // 处理网络错误或其他未知错误
        }
    }
};


</script>

<template>
    <div class="flex items-center h-screen ml-48">
        <div class="relative flex-col bg-white bg-opacity-80 shadow-2xl rounded-2xl p-8 w-96 h-128 ml-40">

            <div class="text-center mb-4">
                <span class="text-3xl font-semibold">{{ $t('register.title') }}</span>
            </div>

            <div class="mb-6">
                <label class="block text-gray-700 text-base font-bold mb-2" for="username">
                    {{ $t('register.username') }}:
                </label>
                <input v-model="username" class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" type="text" :placeholder="$t('register.enterUsername')">
            </div>

            <div class="mb-5">
                <label class="block text-gray-700 text-base font-bold mb-2" for="password">
                    {{ $t('register.password') }}:
                </label>
                <input v-model="password" class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" type="password" :placeholder="$t('register.enterPassword')">
            </div>

            <div class="flex items-center justify-center">
                <button class="bg-yellow-300 hover:bg-yellow-400 text-white font-bold py-2 px-6 rounded focus:shadow-outline mt-1 w-28" @click="handleRegister">
                    {{ $t('register.signUp') }}
                </button>
            </div>
            <div class="text-center mt-5">
                <button class="text-base font-semibold text-black bg-transparent border-none cursor-pointer hover:underline focus:outline-none" @click="navigateToLoginPage">
                    {{ $t('register.haveAccount') }}
                </button>
            </div>
        </div>
    </div>

    <div class="absolute bottom-7 left-16 flex items-center">
        <button @click="switchToChinese" class="font-bold text-mid text-center text-white">{{ $t('message.zh') }}</button>
        <span class="mx-2 text-white">/</span>
        <button @click="switchToEnglish" class="font-bold text-mid text-center text-white">{{ $t('message.en') }}</button>
    </div>
    <div class="absolute bottom-7 right-16">
        <div class="w-[150px] py-3 bg-cyan-400 hover:bg-opacity-85 rounded-full text-center shadow-2xl">
            <button @click="navigateToHomePage" class="font-bold text-2xl text-center text-white">
                {{ $t('message.return') }}
            </button>
        </div>
    </div>
</template>


<style scoped>

</style>