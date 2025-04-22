import axios from 'axios';

// 创建axios实例
const http = axios.create({
    baseURL: 'http://localhost:5000/api', // Flask后端基础 URL
    timeout: 10000, // 请求超时时间
    withCredentials: true, // 允许跨域请求时携带凭证（cookies等）
});

// 请求拦截器
http.interceptors.request.use(
    config => {
        // 根据请求类型动态设置Content-Type
        if (config.url.includes('upload')) {
            // 文件上传请求，使用multipart/form-data
            config.headers['Content-Type'] = 'multipart/form-data';
        } else {
            // 其他API请求，默认使用application/json
            config.headers['Content-Type'] = 'application/json';
        }

        // 在这里可以对请求头进行其他设置，例如添加 token
        // config.headers['Authorization'] = 'Bearer your-token';

        return config;
    },
    error => {
        // 对请求错误进行处理
        console.error('Request Error:', error);
        return Promise.reject(error);
    }
);

// 响应拦截器
http.interceptors.response.use(
    response => {
        // 对响应数据进行处理
        // 可以在这里根据后端返回的状态码来统一处理错误
        if (response.status === 200) {
            // 如果返回的状态码为200，说明接口请求成功，可以正常拿到数据
            return response.data; // 只返回数据部分
        } else {
            // 否则的话抛出错误
            return Promise.reject(response);
        }
    },
    error => {
        // 对响应错误进行处理
        console.error('Response Error:', error);
        return Promise.reject(error);
    }
);

// 导出 http 实例
export default http;