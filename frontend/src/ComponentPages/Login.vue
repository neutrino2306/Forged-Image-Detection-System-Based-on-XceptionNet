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
const navigateToRegisterPage = () => {
    emit('navigate', 'RegisterPage');
};

const handleLogin = async () => {
    // 检查用户名和密码是否为空
    if (!username.value.trim() || !password.value.trim()) {
        alert(t('login.required')); // 提示用户名和密码不能为空
        return;
    }

    try {
        const response = await http.post('/login', {
            username: username.value.trim(),
            password: password.value.trim()
        },{
            withCredentials: true
        });

        if (response.success) {
            // 登录成功，更新用户状态
            loginUser(username.value, password.value);
            navigateToHomePage();
            alert(t('login.success')); // 显示登录成功消息
        }
    } catch (error) {
        if (error.response && error.response.data.error) {
            // 根据后端返回的错误信息显示相应的提示
            switch (error.response.data.error) {
                case 'Invalid username or password':
                    alert(t('login.invalidCredentials'));
                    break;
                default:
                    alert(t('login.error'));
            }
        } else {
            console.error('Login error:', error);
            alert(t('login.error')); // 处理网络错误或其他未知错误
        }
    }
};

</script>

<template>
    <div class="flex items-center h-screen ml-48">
        <div class="relative flex-col bg-white bg-opacity-80 shadow-2xl rounded-2xl p-8 w-96 h-128 ml-40">

            <div class="text-center mb-4">
                <span class="text-3xl font-semibold">{{ $t('login.title') }}</span>
            </div>

            <div class="mb-6">
                <label class="block text-gray-700 text-base font-bold mb-2" for="username">
                    {{ $t('login.username') }}:
                </label>
                <input v-model="username" class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" type="text" :placeholder="$t('login.enterUsername')">
            </div>

            <div class="mb-5">
                <label class="block text-gray-700 text-base font-bold mb-2" for="password">
                    {{ $t('login.password') }}:
                </label>
                <input v-model="password" class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" type="password" :placeholder="$t('login.enterPassword')">
            </div>

            <div class="flex items-center justify-center">
                <button class="bg-yellow-300 hover:bg-yellow-400 text-white font-bold py-2 px-6 rounded focus:shadow-outline mt-1 w-28" @click="handleLogin">
                    {{ $t('login.signIn') }}
                </button>
            </div>
            <div class="text-center mt-5">
                <button class="text-base font-semibold text-black bg-transparent border-none cursor-pointer hover:underline focus:outline-none" @click="navigateToRegisterPage">
                    {{ $t('login.noAccount') }}
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