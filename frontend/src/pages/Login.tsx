import { useState, type FormEvent } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { AlertCircle, KeyRound, Mail, Network } from 'lucide-react';
import { authApi } from '@/api';
import { useAuthStore } from '@/store/useAuthStore';
import Button from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

/** Login page with email/password form and seed-credentials hint. */
export default function Login() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const setAuth = useAuthStore((s) => s.setAuth);
  const [email, setEmail] = useState('hezh@semitech.cn');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const expired = params.get('expired') === '1';

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!email || !password) {
      setError('请输入邮箱和密码');
      return;
    }
    setLoading(true);
    try {
      const { access_token, user } = await authApi.login({ email, password });
      setAuth(user, access_token);
      navigate('/documents', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : '登录失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-8 text-center lg:text-left">
        <div className="mb-4 flex justify-center lg:justify-start">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-200">
            <Network className="h-6 w-6 text-white" strokeWidth={2} />
          </div>
        </div>
        <h2 className="text-2xl font-bold text-slate-800">欢迎回来</h2>
        <p className="mt-1 text-sm text-slate-500">
          登录企业知识库，开启智能检索与问答
        </p>
      </div>

      {expired && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          登录状态已过期，请重新登录。
        </div>
      )}

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="邮箱"
          type="email"
          name="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@company.com"
          icon={Mail}
          autoComplete="email"
        />
        <Input
          label="密码"
          type="password"
          name="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="请输入密码"
          icon={KeyRound}
          autoComplete="current-password"
        />

        <div className="flex items-center justify-between text-xs">
          <label className="flex items-center gap-1.5 text-slate-500">
            <input type="checkbox" className="rounded border-slate-300 text-primary-600 focus:ring-primary-500" />
            记住我
          </label>
          <a href="#" className="font-medium text-primary-600 hover:text-primary-700">
            忘记密码？
          </a>
        </div>

        <Button type="submit" loading={loading} className="w-full" size="lg">
          {loading ? '登录中…' : '登录'}
        </Button>
      </form>

      <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-3">
        <p className="text-xs font-medium text-slate-600">种子账号（离线模式可用）</p>
        <p className="mt-1 text-xs text-slate-500">
          邮箱：<code className="rounded bg-white px-1.5 py-0.5 text-primary-700">hezh@semitech.cn</code>
          {' '}密码：<code className="rounded bg-white px-1.5 py-0.5 text-primary-700">admin123</code>
        </p>
        <p className="mt-1 text-[11px] text-slate-400">
          后端不可用时自动启用离线模式，使用内置种子数据演示全部功能。
        </p>
      </div>
    </div>
  );
}
