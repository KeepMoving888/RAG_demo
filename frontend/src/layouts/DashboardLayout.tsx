import { useEffect, useRef, useState } from 'react';
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Bell,
  ChevronsLeft,
  ChevronsRight,
  FileText,
  LayoutDashboard,
  Library,
  LogOut,
  Menu,
  MessagesSquare,
  Network,
  Search,
  Settings,
  ShieldCheck,
  User as UserIcon,
  type LucideIcon,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/store/useAuthStore';

interface NavItem {
  label: string;
  to: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { label: '智能问答', to: '/qa', icon: MessagesSquare },
  { label: '知识库中心', to: '/hub', icon: Library },
  { label: '文档管理', to: '/documents', icon: FileText },
  { label: '知识图谱', to: '/graph', icon: Network },
  { label: '检索评估', to: '/evaluation', icon: ShieldCheck },
  { label: '管理后台', to: '/admin', icon: LayoutDashboard },
];

/** Resolve a human-readable breadcrumb from the current pathname. */
function useBreadcrumb(): string[] {
  const { pathname } = useLocation();
  const map: Record<string, string> = {
    hub: '知识库中心',
    documents: '文档管理',
    qa: '智能问答',
    graph: '知识图谱',
    evaluation: '检索评估',
    admin: '管理后台',
  };
  const seg = pathname.split('/').filter(Boolean);
  return ['控制台', ...(seg.map((s) => map[s]).filter(Boolean) as string[])];
}

/** Enterprise dashboard shell: dark sidebar + light topbar + content. */
export default function DashboardLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const notifRef = useRef<HTMLDivElement>(null);
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const breadcrumb = useBreadcrumb();

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setNotifOpen(false);
      }
    }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* ---------- Sidebar ---------- */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex flex-col bg-slate-900 text-slate-300 transition-all duration-300 lg:static lg:translate-x-0',
          collapsed ? 'w-[68px]' : 'w-60',
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center gap-2.5 border-b border-slate-800 px-4">
          <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-900/50">
            <Network className="h-5 w-5 text-white" strokeWidth={2} />
          </div>
          {!collapsed && (
            <div className="min-w-0 animate-fade-in">
              <p className="truncate text-sm font-bold text-white">企业知识库</p>
              <p className="truncate text-[11px] text-slate-500">Enterprise RAG</p>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="scrollbar-dark flex-1 space-y-1 overflow-y-auto px-2.5 py-4">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} item={item} collapsed={collapsed} />
          ))}
        </nav>

        {/* User footer */}
        <div className="border-t border-slate-800 p-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-primary-600 text-sm font-semibold text-white">
              {user?.name?.[0] ?? 'U'}
            </div>
            {!collapsed && (
              <div className="min-w-0 flex-1 animate-fade-in">
                <p className="truncate text-sm font-medium text-white">
                  {user?.name ?? '用户'}
                </p>
                <p className="truncate text-[11px] text-slate-500">
                  {user?.email}
                </p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/50 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ---------- Main column ---------- */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Topbar */}
        <header className="z-20 flex h-16 items-center gap-4 border-b border-slate-200 bg-white/90 px-4 backdrop-blur lg:px-6">
          <button
            onClick={() => setMobileOpen(true)}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:hidden"
            aria-label="打开菜单"
          >
            <Menu className="h-5 w-5" />
          </button>
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="hidden rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:block"
            aria-label="折叠侧边栏"
          >
            {collapsed ? (
              <ChevronsRight className="h-5 w-5" />
            ) : (
              <ChevronsLeft className="h-5 w-5" />
            )}
          </button>

          {/* Breadcrumb */}
          <nav className="hidden items-center gap-1.5 text-sm md:flex">
            {breadcrumb.map((c, i) => (
              <span key={c} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-slate-300">/</span>}
                <span
                  className={cn(
                    i === breadcrumb.length - 1
                      ? 'font-medium text-slate-800'
                      : 'text-slate-400'
                  )}
                >
                  {c}
                </span>
              </span>
            ))}
          </nav>

          {/* Search */}
          <div className="relative ml-auto hidden max-w-xs flex-1 sm:block">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="搜索文档、实体、问答…"
              className="focus-ring w-full rounded-lg border border-slate-200 bg-slate-50 py-2 pl-9 pr-3 text-sm placeholder:text-slate-400"
            />
          </div>

          {/* Notifications */}
          <div className="relative" ref={notifRef}>
            <button
              onClick={() => setNotifOpen((o) => !o)}
              className="focus-ring relative rounded-lg p-2 text-slate-500 hover:bg-slate-100"
              aria-label="通知"
            >
              <Bell className="h-5 w-5" />
              <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-rose-500 ring-2 ring-white" />
            </button>
            {notifOpen && (
              <div className="absolute right-0 mt-2 w-80 animate-fade-in overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card-hover">
                <div className="border-b border-slate-100 px-4 py-3 text-sm font-semibold text-slate-700">
                  系统通知
                </div>
                <ul className="max-h-72 divide-y divide-slate-100 overflow-y-auto">
                  {[
                    { t: '文档解析完成', d: '《A 系列供应商准入标准》已就绪', c: 'text-emerald-600' },
                    { t: '解析失败', d: '《固件版本发布说明》编码不兼容', c: 'text-rose-600' },
                    { t: '图谱更新', d: '新增 12 个实体节点', c: 'text-primary-600' },
                  ].map((n, i) => (
                    <li key={i} className="px-4 py-3 hover:bg-slate-50">
                      <p className={cn('text-sm font-medium', n.c)}>{n.t}</p>
                      <p className="mt-0.5 text-xs text-slate-500">{n.d}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* User menu */}
          <div className="relative" ref={userMenuRef}>
            <button
              onClick={() => setUserMenuOpen((o) => !o)}
              className="focus-ring flex items-center gap-2 rounded-lg p-1 hover:bg-slate-100"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary-600 text-sm font-semibold text-white">
                {user?.name?.[0] ?? 'U'}
              </div>
            </button>
            {userMenuOpen && (
              <div className="absolute right-0 mt-2 w-56 animate-fade-in overflow-hidden rounded-xl border border-slate-200 bg-white shadow-card-hover">
                <div className="border-b border-slate-100 px-4 py-3">
                  <p className="text-sm font-medium text-slate-800">{user?.name}</p>
                  <p className="truncate text-xs text-slate-500">{user?.email}</p>
                </div>
                <div className="py-1">
                  <button className="flex w-full items-center gap-2 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50">
                    <UserIcon className="h-4 w-4" /> 个人中心
                  </button>
                  <button className="flex w-full items-center gap-2 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50">
                    <Settings className="h-4 w-4" /> 偏好设置
                  </button>
                </div>
                <div className="border-t border-slate-100 py-1">
                  <button
                    onClick={handleLogout}
                    className="flex w-full items-center gap-2 px-4 py-2 text-sm text-rose-600 hover:bg-rose-50"
                  >
                    <LogOut className="h-4 w-4" /> 退出登录
                  </button>
                </div>
              </div>
            )}
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto scrollbar-thin">
          <div className="mx-auto max-w-[1600px] p-4 lg:p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

/** Single sidebar nav entry with active highlighting. */
function NavLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const { pathname } = useLocation();
  const active = pathname.startsWith(item.to);
  const Icon = item.icon;
  return (
    <Link
      to={item.to}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
        active
          ? 'bg-primary-600 text-white shadow-sm'
          : 'text-slate-400 hover:bg-slate-800 hover:text-white'
      )}
    >
      <Icon className="h-5 w-5 flex-shrink-0" strokeWidth={2} />
      {!collapsed && <span className="animate-fade-in">{item.label}</span>}
    </Link>
  );
}
