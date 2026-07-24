import axios from 'axios'

export const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
})

// This gets set once, from a component that has access to Clerk's useAuth()
let getTokenFn: (() => Promise<string | null>) | null = null

export function setTokenGetter(fn: () => Promise<string | null>) {
  getTokenFn = fn
}

api.interceptors.request.use(async (config) => {
  if (getTokenFn) {
    const token = await getTokenFn()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
  }
  return config
})