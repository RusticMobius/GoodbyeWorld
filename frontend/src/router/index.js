import Vue from 'vue'
// const Vue = ()=>import('vue')
import VueRouter from 'vue-router'
// const VueRouter = ()=> import( 'vue-router')
import question from "../components/question";
import Qs from 'qs';
// const Qs = ()=>import("qs")


Vue.use(VueRouter);
Vue.prototype.Qs = Qs;
const routes = [
  {
    path: "/",
    name: "question",
    component: question,
  }


]

const createRouter = () => new VueRouter({
  // mode: 'history',
  scrollBehavior: () => ({y: 0}),
  routes
});

const router = createRouter()


export function resetRouter() {
  const newRouter = createRouter();
  router.matcher = newRouter.matcher // reset router
}

export default router
