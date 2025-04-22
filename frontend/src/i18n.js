import { createI18n } from 'vue-i18n'

// 定义词条
const messages = {
    en: { // 英文词条
        register: {
            title: 'Sign Up',
            username: 'Username',
            password: 'Password',
            enterUsername: 'Please enter your username',
            enterPassword: 'Please enter your password',
            signUp: 'Register',
            haveAccount: 'Already have an account? Sign in here!',
            success: "Registration successful!",
            error: "Registration failed.",
            usernameRequired: "Username and password are required.",
            usernameExists: "Username already exists."
        },
        errors: {
            updateAtLeastOne: "At least one field must be updated.",
            updateFailed: "Failed to update user.",
            usernameExists: "Username already exists.",
            networkError: "Network error or unknown error."
        },
        login: {
            title: 'Sign In',
            username: 'Username',
            password: 'Password',
            enterUsername: 'Please enter your username',
            enterPassword: 'Please enter your password',
            signIn: 'Login',
            noAccount: 'Don\'t have an account? Register now!',
            success: "Logged in successfully!",
            error: "Login failed.",
            invalidCredentials: "Invalid username or password.",
            required: "Username and password are required."
        },
        message: {
            signIn: 'Sign In',
            signUp: 'Sign Up',
            home: 'HOME',
            detect: 'DETECT',
            about: 'ABOUT',
            user: 'USER',
            userInfo: 'Info',
            userHistory: 'History',
            logOut: 'Log Out',
            helloFriend: 'Hello friend!',
            availableForWork: 'I’m available for detection work.',
            letsBegin: 'Let\'s begin',
            zh: '中文',
            en: 'English',
            Detect: 'Detect',
            Cancel: 'Cancel',
            account: "Account",
            password: "Password",
            edit: "Edit",
            confirm: "Confirm",
            userUpdatedSuccessfully: "User updated successfully.",
            newUsernamePlaceholder: "New Username",
            newPasswordPlaceholder: "New Password",
            noDetectionYet: 'No image has been detected; no results are available.',
            fake: 'Fake',
            authentic: 'Authentic',
            return: 'Return',
            uploadError: 'File upload failed',
            authenticationRequired: 'Authentication required',
            failedToLoadImages: 'Failed to load images, please try again later!',
            detectAlert: 'Detection cannot proceed, please upload an image first!',
            cancelAlert: 'Cancellation cannot proceed, please upload an image first!',
            detectSuccess: 'Detection is completed, please check the result!',
            title: 'A Little Bit About The System',
            introduction: 'This system is developed based on the XceptionNet deep learning model, implementing a counterfeit image recognition feature. Combining a user-friendly web service built with Vue 3 and Tailwind CSS for the frontend and Flask for the backend, it enables users to easily upload and verify the authenticity of images.'
        },
    },
    zh: { // 中文词条
        register: {
            title: '注册',
            username: '账号',
            password: '密码',
            enterUsername: '请输入账号',
            enterPassword: '请输入密码',
            signUp: '注册',
            haveAccount: '已经有账号了？直接登录吧！',
            success: "注册成功！",
            error: "注册失败！",
            usernameRequired: "用户名和密码不能为空!",
            usernameExists: "用户名已存在!"
        },
        errors: {
            updateAtLeastOne: "至少需要更新一个字段!",
            updateFailed: "用户更新失败!",
            usernameExists: "用户名已存在!",
            networkError: "网络错误或未知错误!"
        },
        login: {
            title: '登录',
            username: '账号',
            password: '密码',
            enterUsername: '请输入账号',
            enterPassword: '请输入密码',
            signIn: '登录',
            noAccount: '还没有账号吗？注册一个吧！',
            success: "登录成功！",
            error: "登录失败!",
            invalidCredentials: "用户名或密码无效!",
            required: "用户名和密码不能为空!"
        },
        message: {
            signIn: '登录',
            signUp: '注册',
            home: '首页',
            detect: '检测',
            about: '关于',
            user: '用户',
            userInfo: '个人信息',
            userHistory: '检测历史',
            logOut: '退出登录',
            helloFriend: '你好，朋友！',
            availableForWork: '我可以开始进行检测工作。',
            letsBegin: '开始',
            zh: '中文',
            en: 'English',
            Detect: '检测',
            Cancel: '取消',
            userUpdatedSuccessfully: "用户更新成功。",
            newUsernamePlaceholder: "新的用户名",
            newPasswordPlaceholder: "新的密码",
            noDetectionYet: '还没有检测图片，还没有结果。',
            fake: '伪造',
            authentic: '真实',
            return: '返回',
            account: "账号",
            password: "密码",
            edit: "修改",
            confirm: "确定",
            uploadError: '文件上传失败！',
            authenticationRequired: '请先登录！',
            failedToLoadImages: '加载历史图片失败，请稍后再试!',
            detectAlert: '无法检测，请先上传图片！',
            cancelAlert: '无法取消，请先上传图片！',
            detectSuccess: '检测完毕，请查看结果！',
            title: '系统介绍',
            introduction: '本系统基于XceptionNet深度学习模型开发，实现了伪造图像识别功能。通过结合Vue 3和Tailwind CSS构建的前端以及Flask后端，提供了一个用户友好的Web服务，使用户能够轻松上传并验证图像的真伪。'
        }

    }
}

// 创建 i18n 实例并配置语言环境和词条
const i18n = createI18n({
    legacy: false, // 对于 Vue 3，这个值必须设置为 false
    globalInjection: true, // 允许在应用的全局范围内使用 $t 方法
    locale: 'en', // 默认语言环境
    fallbackLocale: 'en', // 如果没有找到当前语言环境的词条，将回退到该语言环境
    messages, // 设置语言环境信息
})

export default i18n