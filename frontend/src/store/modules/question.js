import {matchCaseInfoAPI, matchCaseJudgementAPI, matchCauseTypeAPI, matchLegalItemAPI} from '@/api/question'

const file = {
  state: {
    chatMessages: [
      {
        type: 1,
        key: 1,
        message: "欢迎使用司法问答系统",
        from: 2,
        timestamp: new Date(),
        displayedTime: '',
        questionType: -1
      },
      {
        type: 1,
        key: 1,
        message: "请在下方选择您想要咨询的问题类型",
        from: 2,
        timestamp: new Date(),
        displayedTime: '',
        questionType: -1
      },
    ],
    qType: 0

  },
  mutations: {
    set_qType: (state, data) => {
      state.qType = data
    },

  },
  actions: {
    question: async ({state, commit}, data) => {
      console.log(state.qType)
      if(state.qType == 1){
        return await matchCauseTypeAPI(data);
      }
      if(state.qType == 2){
        return await matchLegalItemAPI(data);
      }
      if(state.qType == 3){
        return await matchCaseJudgementAPI(data);
      }
      if(state.qType == 4){
        return await matchCaseInfoAPI(data);
      }
    }
  }
}
export default file
