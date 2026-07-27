import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronDown,
  Clock,
  Copy,
  Check,
  Lightbulb,
  Loader2,
  MessageSquarePlus,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from 'lucide-react';
import { qaApi } from '@/api';
import type { QAMessage, QASession, RetrievalExplain } from '@/api/types';
import { cn, formatLatency } from '@/lib/utils';
import Button from '@/components/ui/Button';
import Skeleton from '@/components/ui/Skeleton';
import CitationCard from '@/components/CitationCard';
import RetrievalDetail from '@/components/RetrievalDetail';

const SUGGESTIONS = [
  '车规 eMMC 的 AEC-Q100 认证流程是怎样的？',
  'NAND Flash 的核心供应商有哪些',
  '通过 IATF 16949 认证的封测厂',
  '存储产品可靠性测试包含哪些项目',
];

/** 核心页面: 检索增强问答, 带答案溯源与检索可解释性. */
export default function KnowledgeQA() {
  const qc = useQueryClient();
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [messages, setMessages] = useState<QAMessage[]>([]);
  const [activeRetrieval, setActiveRetrieval] =
    useState<RetrievalExplain | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['qa-sessions'],
    queryFn: qaApi.sessions,
  });

  // 默认选中第一个会话
  useEffect(() => {
    if (!activeSession && sessions && sessions.length > 0) {
      setActiveSession(sessions[0].id);
    }
  }, [sessions, activeSession]);

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['qa-messages', activeSession],
    queryFn: () => (activeSession ? qaApi.messages(activeSession) : Promise.resolve([])),
    enabled: !!activeSession,
  });

  useEffect(() => {
    if (history) {
      setMessages(history);
      const lastAssistant = [...history].reverse().find((m) => m.role === 'assistant');
      setActiveRetrieval(lastAssistant?.retrieval ?? null);
    }
  }, [history]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const createSession = async () => {
    const s = await qaApi.createSession('新会话 ' + new Date().toLocaleTimeString());
    qc.invalidateQueries({ queryKey: ['qa-sessions'] });
    setActiveSession(s.id);
    setMessages([]);
    setActiveRetrieval(null);
  };

  const send = async (text?: string) => {
    const query = (text ?? input).trim();
    if (!query || streaming) return;
    let sessionId = activeSession;
    if (!sessionId) {
      const s = await qaApi.createSession(query.slice(0, 20));
      sessionId = s.id;
      setActiveSession(s.id);
      qc.invalidateQueries({ queryKey: ['qa-sessions'] });
    }
    setInput('');

    const userMsg: QAMessage = {
      id: `u-${Date.now()}`,
      session_id: sessionId,
      role: 'user',
      content: query,
      created_at: new Date().toISOString(),
    };
    const placeholder: QAMessage = {
      id: `a-${Date.now()}`,
      session_id: sessionId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    };
    setMessages((m) => [...m, userMsg, placeholder]);
    setStreaming(true);

    try {
      let acc = '';
      for await (const delta of qaApi.stream(sessionId, query)) {
        acc += delta;
        setMessages((m) =>
          m.map((msg) => (msg.id === placeholder.id ? { ...msg, content: acc } : msg))
        );
      }
      const finalized = await qaApi.ask({ session_id: sessionId, query: '' }).catch(() => null);
      if (finalized && acc) {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === placeholder.id
              ? {
                  ...msg,
                  content: acc,
                  citations: finalized.citations,
                  retrieval: finalized.retrieval,
                  latency_ms: finalized.latency_ms,
                  cache_hit: finalized.cache_hit,
                }
              : msg
          )
        );
        setActiveRetrieval(finalized.retrieval ?? null);
      }
    } finally {
      setStreaming(false);
      qc.invalidateQueries({ queryKey: ['qa-sessions'] });
    }
  };

  const regenerate = async () => {
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUser) {
      setMessages((m) => m.filter((msg) => msg.id !== lastUser.id));
      await send(lastUser.content);
    }
  };

  const onFeedback = async (msg: QAMessage, feedback: 'up' | 'down') => {
    setMessages((m) =>
      m.map((mm) => (mm.id === msg.id ? { ...mm, feedback } : mm))
    );
    await qaApi.feedback({ message_id: msg.id, feedback });
  };

  const copyMsg = (content: string) => {
    navigator.clipboard?.writeText(content);
  };

  return (
    <div className="flex h-[calc(100vh-7.5rem)] gap-4">
      {/* ---------- 会话列表 ---------- */}
      <aside className="hidden w-60 flex-shrink-0 flex-col rounded-xl border border-slate-200 bg-white shadow-sm md:flex">
        <div className="border-b border-slate-100 p-3">
          <Button icon={MessageSquarePlus} className="w-full" onClick={createSession}>
            新建会话
          </Button>
        </div>
        <div className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto p-2">
          {sessionsLoading &&
            Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          {sessions?.map((s) => (
            <SessionItem
              key={s.id}
              session={s}
              active={s.id === activeSession}
              onClick={() => {
                setActiveSession(s.id);
                setMessages([]);
                setActiveRetrieval(null);
              }}
            />
          ))}
        </div>
        {/* 底部能力标签 */}
        <div className="border-t border-slate-100 p-3">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Zap className="h-3 w-3 text-amber-400" />
            <span>混合检索 · 答案溯源 · 可解释</span>
          </div>
        </div>
      </aside>

      {/* ---------- 对话区 ---------- */}
      <section className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        {/* 对话头部 */}
        <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700">
              <Sparkles className="h-4 w-4 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-800">智能问答</h3>
              <p className="text-[11px] text-slate-400">基于企业知识库的 RAG 问答引擎</p>
            </div>
          </div>
          <button
            onClick={() => setPanelOpen((o) => !o)}
            className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            title="切换可解释性面板"
          >
            {panelOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* 消息流 */}
        <div ref={scrollRef} className="scrollbar-thin flex-1 space-y-6 overflow-y-auto px-4 py-5">
          {historyLoading && (
            <div className="space-y-3">
              <Skeleton className="h-16 w-2/3" />
              <Skeleton className="h-24 w-3/4" />
            </div>
          )}
          {!historyLoading && messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-50 to-primary-100 ring-1 ring-primary-200/50">
                <Sparkles className="h-8 w-8 text-primary-600" />
              </div>
              <p className="mt-4 text-base font-semibold text-slate-700">
                向知识库提问，获取带溯源的精准答案
              </p>
              <p className="mt-1.5 text-sm text-slate-400">
                支持混合检索 · 答案溯源 · 检索可解释
              </p>
              <div className="mt-6 grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="group flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-left text-sm text-slate-600 transition-all hover:border-primary-300 hover:bg-primary-50/50 hover:text-primary-700 hover:shadow-sm"
                  >
                    <Lightbulb className="h-3.5 w-3.5 flex-shrink-0 text-amber-400 transition-colors group-hover:text-amber-500" />
                    <span className="line-clamp-1">{s}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              streaming={streaming && msg.content === ''}
              onShowRetrieval={() => setActiveRetrieval(msg.retrieval ?? null)}
              onFeedback={(f) => onFeedback(msg, f)}
              onCopy={() => copyMsg(msg.content)}
              onRegenerate={regenerate}
              isActiveRetrieval={!!msg.retrieval && activeRetrieval?.query === msg.retrieval.query}
            />
          ))}
        </div>

        {/* 输入区 */}
        <div className="border-t border-slate-100 p-3">
          <div className="mb-2 flex flex-wrap gap-1.5">
            {SUGGESTIONS.slice(0, 3).map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-500 transition-all hover:border-primary-300 hover:bg-primary-50 hover:text-primary-700"
              >
                <Lightbulb className="mr-1 inline h-3 w-3 text-amber-400" />
                {s}
              </button>
            ))}
          </div>
          <div className="flex items-end gap-2">
            <div className="relative flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                rows={1}
                placeholder="输入问题，Enter 发送，Shift+Enter 换行"
                className="focus-ring max-h-32 w-full resize-none rounded-xl border border-slate-200 bg-slate-50/50 py-2.5 pl-4 pr-3 text-sm placeholder:text-slate-400"
              />
            </div>
            <Button
              icon={Send}
              onClick={() => send()}
              loading={streaming}
              disabled={!input.trim() && !streaming}
              className="h-[42px]"
            >
              发送
            </Button>
          </div>
        </div>
      </section>

      {/* ---------- 检索可解释性面板 ---------- */}
      {panelOpen && (
        <aside className="hidden w-80 flex-shrink-0 flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm xl:flex">
          <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-primary-50">
              <Search className="h-3.5 w-3.5 text-primary-600" />
            </div>
            <h3 className="text-sm font-semibold text-slate-800">检索可解释性</h3>
          </div>
          <div className="scrollbar-thin flex-1 overflow-y-auto p-4">
            {activeRetrieval ? (
              <RetrievalDetail retrieval={activeRetrieval} />
            ) : (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-50">
                  <Search className="h-6 w-6 text-slate-300" />
                </div>
                <p className="mt-3 text-sm font-medium text-slate-500">等待检索</p>
                <p className="mt-1 text-xs text-slate-400">
                  发送问题后，此处展示<br />检索重写、阶段延迟与召回片段
                </p>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}

/** 会话列表项 */
function SessionItem({
  session,
  active,
  onClick,
}: {
  session: QASession;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full rounded-lg px-3 py-2.5 text-left transition-all',
        active
          ? 'bg-primary-50 ring-1 ring-primary-200/50'
          : 'hover:bg-slate-50'
      )}
    >
      <p className={cn(
        'truncate text-sm font-medium',
        active ? 'text-primary-700' : 'text-slate-600'
      )}>
        {session.title}
      </p>
      <div className="mt-1 flex items-center gap-2">
        <p className="text-[11px] text-slate-400">
          {session.message_count} 条消息
        </p>
        {active && (
          <span className="h-1 w-1 rounded-full bg-primary-400" />
        )}
      </div>
    </button>
  );
}

/** 消息气泡 */
function MessageBubble({
  msg,
  streaming,
  onShowRetrieval,
  onFeedback,
  onCopy,
  onRegenerate,
  isActiveRetrieval,
}: {
  msg: QAMessage;
  streaming: boolean;
  onShowRetrieval: () => void;
  onFeedback: (f: 'up' | 'down') => void;
  onCopy: () => void;
  onRegenerate: () => void;
  isActiveRetrieval: boolean;
}) {
  const [citeOpen, setCiteOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const isUser = msg.role === 'user';

  const handleCopy = () => {
    onCopy();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-md bg-gradient-to-br from-primary-600 to-primary-700 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      {/* AI 头像 */}
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary-500 to-primary-700 text-white shadow-sm">
        <Sparkles className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        {/* 回答主体 */}
        <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
          {msg.content ? (
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
              {msg.content}
              {streaming && <span className="typewriter-cursor" />}
            </p>
          ) : (
            <div className="flex items-center gap-2.5 py-1.5 text-sm text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin text-primary-500" />
              <span>正在检索知识库…</span>
            </div>
          )}
        </div>

        {/* 引用溯源 */}
        {msg.citations && msg.citations.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setCiteOpen((o) => !o)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50"
            >
              <ChevronDown
                className={cn('h-3 w-3 text-slate-400 transition-transform', citeOpen && 'rotate-180')}
              />
              答案溯源
              <span className="rounded-full bg-primary-50 px-1.5 py-0.5 text-[10px] font-semibold text-primary-600">
                {msg.citations.length}
              </span>
            </button>
            {citeOpen && (
              <div className="mt-2 space-y-2 animate-fade-in">
                {msg.citations.map((c, i) => (
                  <CitationCard key={c.chunk_id + i} citation={c} index={i} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* 操作工具栏 */}
        {!streaming && msg.content && (
          <div className="mt-2 flex items-center gap-0.5">
            {msg.retrieval && (
              <button
                onClick={onShowRetrieval}
                className={cn(
                  'rounded-md px-2 py-1 text-[11px] font-medium transition-colors',
                  isActiveRetrieval
                    ? 'bg-primary-50 text-primary-700 ring-1 ring-primary-200/50'
                    : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                )}
              >
                <Search className="mr-1 inline h-3 w-3" />
                检索详情
              </button>
            )}
            <ToolBtn onClick={handleCopy} title="复制">
              {copied ? (
                <Check className="h-3.5 w-3.5 text-emerald-500" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </ToolBtn>
            <ToolBtn onClick={onRegenerate} title="重新生成">
              <RefreshCw className="h-3.5 w-3.5" />
            </ToolBtn>
            <span className="mx-1 h-3 w-px bg-slate-200" />
            <ToolBtn
              onClick={() => onFeedback('up')}
              title="有用"
              active={msg.feedback === 'up'}
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </ToolBtn>
            <ToolBtn
              onClick={() => onFeedback('down')}
              title="无用"
              active={msg.feedback === 'down'}
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </ToolBtn>
            {msg.latency_ms && (
              <span className="ml-auto flex items-center gap-1 text-[11px] text-slate-400">
                <Clock className="h-3 w-3" />
                {formatLatency(msg.latency_ms)}
                {msg.cache_hit && (
                  <span className="rounded bg-emerald-50 px-1 py-0.5 text-[10px] font-medium text-emerald-600">
                    缓存命中
                  </span>
                )}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** 工具栏按钮 */
function ToolBtn({
  children,
  onClick,
  title,
  active,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  active?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        'rounded-md p-1.5 transition-colors',
        active
          ? 'bg-primary-50 text-primary-600'
          : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'
      )}
    >
      {children}
    </button>
  );
}
