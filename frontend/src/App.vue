<script setup>
import { ref } from 'vue';
import HomePage from "@/ComponentPages/Home.vue";
import DetectPage from "@/ComponentPages/Detect.vue";
import ResultPage from "@/ComponentPages/Result.vue";
import AboutPage from "@/ComponentPages/About.vue";
import UserInfoPage from "@/ComponentPages/UserInfo.vue";
import HistoryPage from "@/ComponentPages/History.vue";
import LoginPage from "@/ComponentPages/Login.vue";
import RegisterPage from "@/ComponentPages/Register.vue";

const currentPage = ref('HomePage');

const navigateToPage = (pageName) => {
    currentPage.value = pageName;
};
</script>

<template>

    <HomePage v-if="currentPage === 'HomePage'" @navigate="navigateToPage" />
    <DetectPage v-if="currentPage === 'DetectPage'" @navigate="navigateToPage" />
    <ResultPage v-if="currentPage === 'ResultPage'" @navigate="navigateToPage" />
    <AboutPage v-if="currentPage === 'AboutPage'" @navigate="navigateToPage" />
    <UserInfoPage v-if="currentPage === 'UserInfoPage'" @navigate="navigateToPage" />
    <HistoryPage v-if="currentPage === 'HistoryPage'" @navigate="navigateToPage" />
    <LoginPage v-if="currentPage === 'LoginPage'" @navigate="navigateToPage" />
    <RegisterPage v-if="currentPage === 'RegisterPage'" @navigate="navigateToPage" />


</template>

<style scoped>
.core::-webkit-scrollbar {
    display: none;
}
</style>
