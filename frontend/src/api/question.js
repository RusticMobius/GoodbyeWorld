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
      // console.log(state.qType)
      if (state.qType == 1) {
        const res = await matchCauseTypeAPI(data);
        return "您描述的案件类型可能为 "+res+" 。";
      }
      if (state.qType == 2) {
        const res = await matchLegalItemAPI(data);
        return "与您描述的案情相关的法条为 "+Object.values(res)+"。";
      }
      if (state.qType == 3) {
        const res = await matchCaseJudgementAPI(data);
        return "以下为与您的描述相似的案件的判决结果，可作为参考：\n"+Object.values(Object.values(res)[0])[2];
      }
      if (state.qType == 4) {
        const res = await matchCaseInfoAPI(data);
        return "以下为与您的描述相似的案件：\n"+Object.keys(res) + "\n 其案件基本情况为："+Object.values(res);
      }
    }
  }
}
export default file
