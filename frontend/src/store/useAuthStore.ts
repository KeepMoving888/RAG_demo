import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from '@/api/types';
import { clearToken, setToken } from '@/api/client';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  setAuth: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (patch: Partial<User>) => void;
}

/**
 * Global client-side auth store (Zustand + persist to localStorage).
 * Server state is managed by React Query; this only holds identity/token.
 */
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      setAuth: (user, token) => {
        setToken(token);
        set({ user, token, isAuthenticated: true });
      },
      logout: () => {
        clearToken();
        set({ user: null, token: null, isAuthenticated: false });
      },
      updateUser: (patch) => {
        const cur = get().user;
        if (cur) set({ user: { ...cur, ...patch } });
      },
    }),
    {
      name: 'rag_kb_auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
