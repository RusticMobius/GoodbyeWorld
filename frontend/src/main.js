// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import store from './store'

import App from './App'
import router from './router'
import 'ant-design-vue/dist/antd.css';
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/antd.less'


Vue.config.productionTip = false

Vue.use(Antd);


import infiniteScroll from "vue-infinite-scroll";
Vue.use(infiniteScroll);


router.afterEach(route => {
  window.scroll(0, 0);
});

new Vue({
  router,//引入router
  store,//引vuex
  render: h => h(App)
}).$mount('#app');
//挂在到app,进行渲染
