import {axios} from '@/utils/request'

// const api = {
//   qPre: '/api/question'
// };

export function matchCauseTypeAPI(data) {
  return axios({
    url: `http://localhost:8802/matchCauseType`,
    method: 'post',
    params: {info: data}
  })
}

export function matchCaseInfoAPI(data) {
  return axios({
    url: `http://localhost:8802/matchCaseInfo`,
    method: 'post',
    params: {info: data}
  })
}

export function matchCaseJudgementAPI(data) {
  return axios({
    url: `http://localhost:8802/matchCaseJudgement`,
    method: 'post',
    params: {info: data}
  })
}

export function matchLegalItemAPI(data) {
  return axios({
    url: `http://localhost:8802/matchLegalItem`,
    method: 'post',
    params: {info: data}
  })
}


// export function selectAPI(data) {
//   return axios({
//     url: `${api.qPre}/selectQuestionType/${data}`,
//     method: 'post'
//   })
// }
