import Cookies from 'js-cookie'

const TokenKey = 'NJUSE-TOKEN'
const SecretTokenKey = 'token'

export function getToken() {
  return Cookies.get(TokenKey)
}

export function setToken(token) {
  return Cookies.set(TokenKey, token)
}

export function removeToken() {
  return Cookies.remove(TokenKey)
}

export function getSecretToken() {
  return Cookies.get(SecretTokenKey);
}

export function setSecretToken(secretToken) {
  localStorage.setItem("token", secretToken)
  return Cookies.set(SecretTokenKey, secretToken);
}

export function removeSecretToken() {
  localStorage.removeItem("token")
  return Cookies.remove(SecretTokenKey)
}
