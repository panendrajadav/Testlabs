'use client'

import { useState, useEffect } from 'react'

const DEMO_USER = 'Panendra'
const DEMO_PASS = '16062005'
const AUTH_KEY  = 'tl_auth'

export interface AuthUser { username: string }

let _authCache: AuthUser | null | undefined = undefined

function _load(): AuthUser | null {
  if (_authCache !== undefined) return _authCache
  try {
    const raw = localStorage.getItem(AUTH_KEY)
    _authCache = raw ? JSON.parse(raw) : null
  } catch { _authCache = null }
  return _authCache!
}

export function login(username: string, password: string): boolean {
  if (username === DEMO_USER && password === DEMO_PASS) {
    const user: AuthUser = { username }
    _authCache = user
    localStorage.setItem(AUTH_KEY, JSON.stringify(user))
    window.dispatchEvent(new Event('authUpdated'))
    return true
  }
  return false
}

export function logout() {
  _authCache = null
  localStorage.removeItem(AUTH_KEY)
  localStorage.removeItem('currentDataset')
  window.dispatchEvent(new Event('authUpdated'))
  window.dispatchEvent(new Event('datasetUpdated'))
}

export function getAuth(): AuthUser | null {
  return _load()
}

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(() => _load())

  useEffect(() => {
    const sync = () => {
      _authCache = undefined
      setUser(_load())
    }
    window.addEventListener('authUpdated', sync)
    window.addEventListener('storage', sync)
    return () => {
      window.removeEventListener('authUpdated', sync)
      window.removeEventListener('storage', sync)
    }
  }, [])

  return { user, isLoggedIn: !!user }
}
