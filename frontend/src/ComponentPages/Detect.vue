<script setup>
import {ref} from "vue";
import { useI18n } from 'vue-i18n'
import { imageSrc, detectionResult } from '../Store/imageStore.js';
import http from "@/Network/request.js";
import {logoutUser, userState} from "@/Store/userState.js";

const emit = defineEmits(['navigate']);
const currentPage = ref('DetectPage'); // 当前页面变量，用于确定哪个导航按钮应该是激活状态
const { t, locale } = useI18n() // 使用useI18n()同时获取t函数和locale响应式引用
const fileInput = ref(null);
const currentFile = ref(null); // 用于存储当前处理的文件
const triggerFileInput = () => {
    fileInput.value.click();
};

// 处理文件选择
const handleFiles = (event) => {
    const file = event.target.files[0];
    processAndPreviewImage(file);
};

// 处理拖拽上传
const handleDrop = (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        processAndPreviewImage(file);
    }
};

// 读取并预览图片
const processAndPreviewImage = (file) => {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imageSrc.value = e.target.result; // 更新全局状态中的图片数据
            currentFile.value = file;
        };
        reader.readAsDataURL(file);
    } else {
        alert('请上传图片文件');
    }
};

// 上传图片到后端
// const uploadImage = () => {
//     if (imageSrc.value) {
//         const file = fileInput.value.files[0];
//         const formData = new FormData();
//         formData.append('image', file);
//
//         // 使用axios实例发送POST请求
//         http.post('/upload', formData,{
//             withCredentials: true
//         })
//             .then(data => {
//                 console.log('Upload successful:', data);
//                 detectionResult.isFake = data.isFake; // 直接更新isFake属性
//                 alert(t('message.detectSuccess')); // 弹出检测完成的提示
//                 navigateToResultPage(); // 检测完成后自动跳转到结果页面
//             })
//             .catch(error => {
//                 console.error('Error uploading image:', error);
//             });
//     } else {
//         alert(t('message.detectAlert'));
//     }
// };

const uploadImage = async () => {
    // if (!imageSrc.value) {
    //     alert(t('message.detectAlert'));
    //     return;
    // }
    if (!currentFile.value) {
        alert(t('message.detectAlert'));
        return;
    }

    const file = fileInput.value.files[0];
    const formData = new FormData();
    // formData.append('image', file);
    formData.append('image', currentFile.value);
    try {
        const response = await http.post('/upload', formData, {
            withCredentials: true
        });

        if (response) {
            console.log('Upload successful:', response);
            detectionResult.isFake = response.isFake;
            alert(t('message.detectSuccess'));
            navigateToResultPage();
        }
    } catch (error) {
        if (error.response && error.response && error.response.error) {
            switch (error.response.error) {
                // case 'Authentication required':
                //     alert(t('message.authenticationRequired'));
                //     break;
                default:
                    alert(t('message.uploadError'));
            }
        } else {
            console.error('Error uploading image:', error);
            alert(t('errors.networkError'));
        }
    }
};

const detectImage = () => {
    console.log('imageSrc in detect page:', imageSrc.value);
    uploadImage(); // 调用上传图片到后端的函数
};

const cancelUpload = () => {
    // 检查是否已选择图片
    if (imageSrc.value) {
        console.log('Canceling upload...');  // 清空图片预览或其他状态重置
        imageSrc.value = null;  // 清除预览
        fileInput.value.value = '';  // 重置文件输入
        detectionResult.isFake = null; // 重置检测结果
    } else {
        // 如果没有选择图片，显示提示消息
        alert(t('message.cancelAlert'));
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

const navigateToResultPage = () => {
    emit('navigate', 'ResultPage');
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

<!--    <div class="absolute left-1/4 top-40 bg-white bg-opacity-80 w-[500px] h-72 rounded-lg flex justify-center items-center"-->
<!--         @dragover.prevent-->
<!--         @dragenter.prevent-->
<!--         @drop="handleDrop">-->
<!--        <p v-if="!imageSrc">拖拽图片到这里上传</p>-->
<!--        <img v-if="imageSrc" :src="imageSrc" class="w-[500px] h-72 rounded-lg" />-->
<!--    </div>-->

    <div class="flex items-start justify-center mt-40 ml-60 space-x-4">
        <div class="bg-white bg-opacity-80 w-[500px] h-72 rounded-lg flex justify-center items-center relative shadow-2xl"
             @dragover.prevent
             @dragenter.prevent
             @drop="handleDrop">
            <div v-if="!imageSrc" @click="triggerFileInput" class="cursor-pointer">
                <!-- 使用SVG作为加号图标 -->
                <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                </svg>
            </div>
            <img v-if="imageSrc" :src="imageSrc" alt="图片预览" class="shadow-xl w-[500px] h-72 rounded-lg"/>
            <input type="file" ref="fileInput" @change="handleFiles" class="hidden" accept="image/*" />
        </div>

        <div style="margin-left: 5rem; display: flex; flex-direction: column; gap: 4rem; margin-top: 4rem;">
            <div class="w-[120px] py-3 bg-cyan-400 hover:bg-opacity-75 rounded-full text-center shadow-2xl">
                <button @click="detectImage" class="font-bold text-2xl text-center text-white">{{ $t('message.Detect') }}</button>
            </div>
            <div class="w-[120px] py-3 bg-cyan-400 hover:bg-opacity-75 rounded-full text-center shadow-2xl">
                <button @click="cancelUpload" class="font-bold text-2xl text-center text-white">{{ $t('message.Cancel') }}</button>
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