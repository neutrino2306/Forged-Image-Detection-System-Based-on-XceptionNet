<script setup>
import {ref, reactive, onMounted} from "vue";
import { useI18n } from 'vue-i18n'
import {userState, loginUser, logoutUser} from '@/Store/userState.js';
import http from '@/Network/request.js';

const emit = defineEmits(['navigate']);
const currentPage = ref('UserInfoPage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const { locale } = useI18n()
const { t } = useI18n();
const isEditing = ref(false);
const editableUserInfo = reactive({
    username: '',
    password: '',
});

const toggleEdit = () => {
    isEditing.value = !isEditing.value;
    if (isEditing.value) {
        // 进入编辑模式时，从userState复制userInfo
        editableUserInfo.username = '';
        editableUserInfo.password = '';
    }
};

const saveChanges = async () => {
    let updatedUserInfo = {};

    if (editableUserInfo.username.trim()) {
        updatedUserInfo.username = editableUserInfo.username.trim();
    }

    if (editableUserInfo.password.trim()) {
        updatedUserInfo.password = editableUserInfo.password.trim();
    }

    // 确保至少有一个字段被更新
    if (!updatedUserInfo.username && !updatedUserInfo.password) {
        alert(t('errors.updateAtLeastOne'));
        return;
    }

    try {
        const response = await http.put('/user/update', updatedUserInfo,{
            withCredentials: true
        });

        if (response && response.success) {
            console.log('User updated successfully');
            alert(t('message.userUpdatedSuccessfully'));
            if (updatedUserInfo.username) {
                loginUser(updatedUserInfo.username, userState.userInfo.password);
            } else {
                loginUser(userState.userInfo.username, updatedUserInfo.password);
            }
            isEditing.value = false;
        } else {
            console.error(response.error);
            alert(t('errors.updateFailed'));
        }
    } catch (error) {
        if (error.response && error.response.data && error.response.data.error) {
            // 根据后端返回的错误信息显示相应的提示
            switch (error.response.data.error) {
                case 'Username already exists':
                    alert(t('errors.usernameExists')); // "用户名已存在"
                    break;
                default:
                    alert(t('errors.updateFailed')); // "更新失败"
            }
        } else {
            console.error('Error updating user:', error);
            alert(t('errors.networkError')); // "网络错误或未知错误"
        }
    }
};

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

    <div class="flex justify-center mt-40 ml-72">
        <div class="bg-white bg-opacity-65 w-[500px] h-72 rounded-lg shadow-2xl flex justify-center items-center">
            <div v-if="!isEditing" class="flex flex-col gap-8 items-start">
                <div class="flex items-center gap-4">
                    <label class="text-xl w-24 -ml-16">{{ t('message.account') }}:</label>
                    <span class="text-xl">{{ userState.userInfo.username }}</span>
                </div>

                <div class="flex items-center gap- mt-2">
                    <label class="text-xl w-24 -ml-16">{{ t('message.password') }}:</label>
                    <span class="text-xl ml-4">******</span>
                </div>

                <div class="w-[100px] py-2 bg-cyan-400 hover:bg-opacity-65 rounded-lg text-center shadow-2xl self-center mt-2">
                    <button @click="toggleEdit" class="font-bold text-xl text-center text-white">{{ t('message.edit') }}</button>
                </div>
            </div>

            <!-- 编辑模式 -->
            <div v-else class="flex flex-col gap-6 items-center">
                <div class="flex items-center gap-4 mt-3">
                    <label class="text-xl w-24">{{ t('message.account') }}:</label>
                    <input v-model="editableUserInfo.username" class="outline-none text-lg rounded-xl pl-5 bg-gray-100 h-12" :placeholder="t('message.newUsernamePlaceholder')" />
                </div>

                <div class="flex items-center gap-4">
                    <label class="text-xl w-24">{{ t('message.password') }}:</label>
                    <input v-model="editableUserInfo.password" type="password" class="outline-none text-lg rounded-xl pl-5 bg-gray-100 h-12" :placeholder="t('message.newPasswordPlaceholder')" />
                </div>

                <div class="flex space-x-16 mt-3 justify-center">
                    <div class="w-[100px] py-2 bg-cyan-400 hover:bg-opacity-65 rounded-lg text-center shadow-2xl">
                        <button @click="saveChanges" class="font-bold text-xl text-center text-white">{{ t('message.confirm') }}</button>
                    </div>
                    <div class="w-[100px] py-2 bg-cyan-400 hover:bg-opacity-65 rounded-lg text-center shadow-2xl">
                        <button @click="toggleEdit" class="font-bold text-xl text-center text-white">{{ t('message.Cancel') }}</button>
                    </div>
                </div>
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