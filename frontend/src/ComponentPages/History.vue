<script setup>
import {ref, onMounted} from "vue";
import { useI18n } from 'vue-i18n'
import {imageSrc} from "@/Store/imageStore.js";
import http from '@/Network/request.js';
import {logoutUser, userState} from "@/Store/userState.js";

const emit = defineEmits(['navigate']);
const currentPage = ref('HistoryPage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const records = ref([]);
const { t, locale } = useI18n() // 使用useI18n()同时获取t函数和locale响应式引用

const fetchUserImageHistory = async () => {
    try {
        const response = await http.get('/history',{
            withCredentials: true
        });
        console.log("Received response:", response);
        if (response && Array.isArray(response)) {
            records.value = response.map(record => ({
                ID: record.ID,
                isFake: record.is_fake ? 1 : 0,
                name: record.filename,
                time: record.uploaded_at,
                url: record.url,
                userID: record.userID
            }));
        }
    } catch (error) {
        if (error.response && error.response.data && error.response.data.error) {
            // 根据后端返回的具体错误信息处理
            switch (error.response.data.error) {
                case 'Authentication required':
                    alert(t('message.authenticationRequired'));
                    break;
                default:
                    alert(t('message.failedToLoadImages'));
                    break;
            }
        } else {
            console.error('获取用户图片历史失败:', error);
            alert(t('errors.networkError'));
        }
    }
};

onMounted(fetchUserImageHistory);


// async function fetchRecords() {
//     // 模拟从后端获取数据的过程
//     return [
//         {
//             id: 1,
//             name: '20240313204026_000_003_14.jpg',
//             isFake: 0,
//             time: '2024-03-13 12:40:28',
//             userId: 1,
//             url: 'http://localhost:5000/uploads/20240313204026_000_003_14.jpg',
//         },
//         // 添加更多记录...
//     ];
// }

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

    <div class="mt-36 ml-20 flex justify-center">
        <div class="shadow-2xl border border-gray-200 rounded-lg overflow-hidden" style="width: 910px;">
            <div class="custom-scrollbar overflow-y-auto" style="max-height: 320px;">
                <table class="w-full">
                <thead class="bg-gray-100 sticky top-0">
                <tr>
                    <th class="px-5 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 130px;">图片ID</th>
                    <th class="px-4 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 200px;">图片名称</th>
                    <th class="px-2 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 120px;">真伪情况</th>
                    <th class="px-4 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 170px;">检测时间</th>
                    <th class="px-2 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 120px;">用户ID</th>
                    <th class="px-4 py-4 text-center text-mid font-medium text-gray-500 uppercase tracking-wider" style="width: 140px;">图片URL</th>
                </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                <tr v-for="record in records" :key="record.ID" class="hover:bg-gray-100">
                    <td class="px-4 py-4 text-center whitespace-nowrap">{{ record.ID }}</td>
                    <td class="px-4 py-4 text-center whitespace-nowrap">{{ record.name }}</td>
                    <td class="px-4 py-4 text-center whitespace-nowrap" :class="{'text-green-500': record.isFake === 0, 'text-red-500': record.isFake === 1}">
                        {{ record.isFake === 0 ? '真实' : '伪造' }}
                    </td>
                    <td class="px-4 py-4 text-center whitespace-nowrap">{{ record.time }}</td>
                    <td class="px-4 py-4 text-center whitespace-nowrap">{{ record.userID }}</td>
                    <td class="px-4 py-4 text-center whitespace-nowrap">
                        <a :href="record.url" target="_blank" class="text-blue-500 hover:text-blue-600">查看图片</a>
                    </td>
                </tr>
                </tbody>
                </table>
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
.custom-scrollbar {
    scrollbar-width: none; /* 对 Firefox */
    -ms-overflow-style: none; /* 对 IE 10+ 和 Edge */
}
.custom-scrollbar::-webkit-scrollbar {
    width: 0; /* 对 Chrome, Safari 和 Opera */
    height: 0; /* 对横向滚动条 */
}
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