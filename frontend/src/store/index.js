import Vue from 'vue'
import Vuex from 'vuex'
import getters from './getters'
import question from "./modules/question";

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    question,
  },
  state: {},
  mutations: {},
  actions: {},
  getters
})

