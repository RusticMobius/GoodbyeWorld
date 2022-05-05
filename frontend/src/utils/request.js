import Vue from 'vue'
import axios from 'axios'
import {VueAxios} from './axios'
import {notification, message} from 'ant-design-vue'
import store from '@/store'
import {getToken} from './auth'
import router from '../router'

// 创建 axios 实例
const service = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? 'http://139.196.173.197:8080' : 'http://localhost:8080',
  withCredentials: true
})
console.log(process.env.NODE_ENV)

const err = (error) => {
  if (error.response) {
    const data = error.response.data
    const token = Vue.ls.get('ACCESS_TOKEN')
    if (error.response.status === 403) {
      notification.error({
        message: 'Forbidden',
        description: data.message
      })
    }
    if (error.response.status === 401 && !(data.result && data.result.isLogin)) {
      notification.error({
        message: 'Unauthorized',
        description: 'Authorization verification failed'
      })
      if (token) {
        store.dispatch('Logout').then(() => {
          setTimeout(() => {
            window.location.reload()
          }, 1500)
        })
      }
    }
  }
  return Promise.reject(error)
}

//request incerceptor
//拦截请求
service.interceptors.request.use(
  (config) => {
    const requestConfig = {
      ...config,
      url: `${config.url}`,
    }
    // console.log(requestConfig.url)
    if (localStorage.getItem("token")) {
      const token = localStorage.getItem("token");
      console.log("TokenFromLocalStore ", token)
      requestConfig.headers.token = token;
    }
    return requestConfig
  }, err);

//拦截响应
service.interceptors.response.use((response) => {

  switch (response.status) {
    case 200:
      return response.data;
    case 404:
      return false;
    case 401:
      // 返回 401 清除token信息并跳转到登录页面
      store.commit('logout');
      router.replace({
        path: 'login',
        query: {redirect: router.currentRoute.fullPath}
      });
      break;
    default:
  }
})

const installer = {
  vm: {},
  install(Vue) {
    Vue.use(VueAxios, service)
  }
}

export {
  installer as VueAxios,
  service as axios
}
