import { reactive } from 'vue';

export const userState = reactive({
    userInfo: {
        username: '',
        password: '',
    },
    isLoggedIn: false
});


export const loginUser = (username, password) => {
    userState.userInfo.username = username;
    userState.userInfo.password = password;
    userState.isLoggedIn = true;
};


export const logoutUser = () => {
    userState.userInfo.username = '';
    userState.userInfo.password = '';
    userState.isLoggedIn = false;
};