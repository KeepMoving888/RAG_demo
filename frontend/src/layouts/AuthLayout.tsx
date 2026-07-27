import { Network, FileText, MessagesSquare, ShieldCheck } from 'lucide-react';

interface AuthLayoutProps {
  children: React.ReactNode;
}

const CAPABILITIES = [
  {
    icon: FileText,
    title: '文档智能解析',
    desc: '支持 PDF / Word / Excel / Markdown 多格式，自动分块与标题路径提取',
  },
  {
    icon: MessagesSquare,
    title: '检索增强问答',
    desc: 'BM25 + 向量 + RRF 融合 + 重排，答案溯源到原文段落',
  },
  {
    icon: Network,
    title: '知识图谱',
    desc: '实体关系抽取与力导向可视化，自然语言转 Cypher 查询',
  },
  {
    icon: ShieldCheck,
    title: '检索评估',
    desc: '消融实验对比 Recall / MRR / NDCG，量化检索管线效果',
  },
];

/** Split-screen authentication layout: brand showcase on the left, form on the right. */
export default function AuthLayout({ children }: AuthLayoutProps) {
  return (
    <div className="flex min-h-screen">
      {/* Brand panel */}
      <div className="relative hidden w-1/2 flex-col justify-between overflow-hidden bg-slate-900 p-12 lg:flex">
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage:
              'radial-gradient(circle at 20% 30%, rgba(79,70,229,0.4), transparent 45%), radial-gradient(circle at 80% 70%, rgba(16,185,129,0.25), transparent 40%)',
          }}
        />
        <div className="relative">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-900/50">
              <Network className="h-6 w-6 text-white" strokeWidth={2} />
            </div>
            <div>
              <p className="text-lg font-bold text-white">企业知识库</p>
              <p className="text-xs text-slate-400">Enterprise RAG Knowledge Base</p>
            </div>
          </div>
        </div>

        <div className="relative space-y-8">
          <div>
            <h1 className="text-3xl font-bold leading-tight text-white">
              让企业的每一份知识，
              <br />
              都能被<span className="text-primary-400">精准检索</span>与
              <span className="text-emerald-400">智能问答</span>
            </h1>
            <p className="mt-4 max-w-md text-sm leading-relaxed text-slate-400">
              面向企业内部场景的检索增强生成（RAG）知识库平台，融合混合检索、
              答案溯源、知识图谱与量化评估，构建可信赖的企业级知识中枢。
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {CAPABILITIES.map((c) => {
              const Icon = c.icon;
              return (
                <div
                  key={c.title}
                  className="rounded-xl border border-slate-800 bg-slate-800/40 p-4 backdrop-blur transition-colors hover:border-primary-700/60 hover:bg-slate-800/70"
                >
                  <Icon className="h-5 w-5 text-primary-400" strokeWidth={2} />
                  <p className="mt-2 text-sm font-semibold text-white">{c.title}</p>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">
                    {c.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <div className="relative text-xs text-slate-500">
          © 2026 Enterprise RAG Knowledge Base · 企业级生产部署
        </div>
      </div>

      {/* Form panel */}
      <div className="flex w-full items-center justify-center bg-slate-50 p-6 lg:w-1/2">
        <div className="w-full max-w-sm">{children}</div>
      </div>
    </div>
  );
}
