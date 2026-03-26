'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Database, Users, Clock, Settings, Mic, MicOff, ArrowUp, Loader2, X, ChevronRight, Zap, HardDrive, FileText, Image, Video, Music, Code2, File, Sparkles, Activity, Search, FolderOpen, Volume2 } from 'lucide-react'
import ResultGrid from '@/components/ResultGrid/ResultGrid'
import PreviewPane from '@/components/PreviewPane/PreviewPane'
import IngestionPanel from '@/components/IngestionPanel/IngestionPanel'
import IdentityManager from '@/components/IdentityManager/IdentityManager'
import SetupScreen from '@/components/SetupScreen/SetupScreen'

export interface QueryResult {
  chunk_id:      string
  score:         number
  file_type:     string
  source_path:   string
  text:          string | null
  metadata:      Record<string, unknown>
  thumbnail_url: string | null
}

export interface QueryResponse {
  query:                 string
  results:               QueryResult[]
  total:                 number
  llm_response:          string | null
  suggested_refinements: string[]
  session_id:            string | null
}

type NavView = 'recall' | 'ingest' | 'people' | 'timeline' | 'remote'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  results?: QueryResult[]
  total?: number
  refinements?: string[]
  timestamp: Date
}

interface IndexStats {
  total:     number
  images:    number
  documents: number
  videos:    number
  audio:     number
  code:      number
}

const MODEL_DISPLAY: Record<string, string> = {
  minilm:   'Language Model',
  clip:     'Vision Model',
  codebert: 'Code Model',
  whisper:  'Audio Model',
  deepface: 'Identity Model',
}

export default function Home() {
  const [setupDone,     setSetupDone]     = useState<boolean>(true)
  const [modelStatuses, setModelStatuses] = useState<{id: string, name: string, status: string}[]>([])
  const [view,          setView]          = useState<NavView>('recall')
  const [selectedChunk, setSelectedChunk] = useState<QueryResult | null>(null)
  const [showIdentity,  setShowIdentity]  = useState(false)
  const [loading,       setLoading]       = useState(false)
  const [llmLoading,    setLlmLoading]    = useState(false)
  const [messages,      setMessages]      = useState<ChatMessage[]>([])
  const [input,         setInput]         = useState('')
  const [listening,     setListening]     = useState(false)
  const [voiceSupport,  setVoiceSupport]  = useState(false)
  const [ttsSupport,    setTtsSupport]    = useState(false)
  const [ttsEnabled,    setTtsEnabled]    = useState(false)
  const [stats,         setStats]         = useState<IndexStats | null>(null)
  const [apiReady,      setApiReady]      = useState<boolean | null>(null)
  const [focused,       setFocused]       = useState(false)
  const [sessionId,     setSessionId]     = useState<string | null>(null)
  const [streamingText, setStreamingText] = useState('')
  const inputRef       = useRef<HTMLTextAreaElement>(null)
  const recognRef      = useRef<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const ttsRef         = useRef<SpeechSynthesisUtterance | null>(null)

  const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
  const [apiKey, setApiKey] = useState<string | null>(null)

  function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
    const headers: Record<string, string> = { ...(init.headers as Record<string, string> || {}) }
    if (apiKey) headers['X-API-Key'] = apiKey
    return fetch(url, { ...init, headers })
  }

  // Init: restore or create session, detect browser capabilities
  useEffect(() => {
    if (typeof window === 'undefined') return

    // Fetch API key first, then init session with it
    fetch(`${API}/setup/tunnel`)
      .then(r => r.json())
      .then(d => {
        const key = d.api_key || null
        if (key) setApiKey(key)

        const stored = localStorage.getItem('omnex_session_id')
        if (stored) {
          setSessionId(stored)
        } else {
          const headers: Record<string, string> = {}
          if (key) headers['X-API-Key'] = key
          fetch(`${API}/query/sessions`, { method: 'POST', headers })
            .then(r => r.json())
            .then(d => {
              if (d.session_id) {
                setSessionId(d.session_id)
                localStorage.setItem('omnex_session_id', d.session_id)
              }
            })
            .catch(() => {})
        }
      })
      .catch(() => {
        // No tunnel/auth — still init session
        const stored = localStorage.getItem('omnex_session_id')
        if (!stored) {
          fetch(`${API}/query/sessions`, { method: 'POST' })
            .then(r => r.json())
            .then(d => { if (d.session_id) { setSessionId(d.session_id); localStorage.setItem('omnex_session_id', d.session_id) } })
            .catch(() => {})
        } else {
          setSessionId(stored)
        }
      })

    const hasSpeechRec = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window
    setVoiceSupport(hasSpeechRec)

    const hasTts = 'speechSynthesis' in window
    setTtsSupport(hasTts)
    // Default TTS on if available
    if (hasTts) setTtsEnabled(true)
  }, [])

  // Fetch stats on mount + poll every 10s to keep connection indicator live
  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  async function fetchStats() {
    try {
      const res = await apiFetch(`${API}/stats`, { signal: AbortSignal.timeout(3000) })
      if (!res.ok) { setApiReady(false); return }
      const data = await res.json()
      setApiReady(true)
      setStats({
        total:     data.total_chunks       || 0,
        images:    data.by_type?.image     || 0,
        documents: data.by_type?.document  || 0,
        videos:    data.by_type?.video     || 0,
        audio:     data.by_type?.audio     || 0,
        code:      data.by_type?.code      || 0,
      })
      apiFetch(`${API}/setup/status`)
        .then(r => r.json())
        .then(d => {
          if (Array.isArray(d.models) && d.models.length > 0) setModelStatuses(d.models)
          if (typeof d.ready === 'boolean') setSetupDone(d.ready)
        })
        .catch(() => {})
    } catch { setApiReady(false) }
  }

  const speakText = useCallback((text: string) => {
    if (!ttsEnabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.rate  = 1.0
    utt.pitch = 1.0
    ttsRef.current = utt
    window.speechSynthesis.speak(utt)
  }, [ttsEnabled])

  async function handleQuery(query: string) {
    if (!query.trim() || loading) return
    setLoading(true)
    setSelectedChunk(null)
    setStreamingText('')

    const userMsg: ChatMessage = { role: 'user', content: query, timestamp: new Date() }
    setMessages((m) => [...m, userMsg])
    setInput('')
    if (inputRef.current) inputRef.current.style.height = 'auto'

    // Build conversation history for LLM (last 10 turns, text-only)
    const history = messages
      .filter(m => m.content)
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }))

    try {
      const res  = await apiFetch(`${API}/query`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ query, top_k: 20, session_id: sessionId }),
      })
      const data: QueryResponse = await res.json()

      // Update session_id if server returned one (auto-created)
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id)
        localStorage.setItem('omnex_session_id', data.session_id)
      }

      setLoading(false)

      // Add assistant message placeholder with results
      setMessages((m) => [...m, {
        role:        'assistant',
        content:     '',
        results:     data.results,
        total:       data.total,
        refinements: data.suggested_refinements,
        timestamp:   new Date(),
      }])

      // Stream LLM response via /api/chat — always, even with no results
      {
        const context = data.results?.length
          ? data.results.slice(0, 5).map((r, i) =>
              `[${i + 1}] ${r.file_type.toUpperCase()} — ${r.source_path}\n${(r.text || '').slice(0, 400)}`
            ).join('\n\n')
          : ''

        setLlmLoading(true)
        let fullText = ''

        const chatRes = await fetch('/api/chat', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ query, context, messages: history }),
        })

        const reader = chatRes.body?.getReader()
        const decoder = new TextDecoder()

        if (reader) {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            const raw = decoder.decode(value, { stream: true })
            for (const line of raw.split('\n')) {
              const trimmed = line.trim()
              if (!trimmed || trimmed === 'data: [DONE]') continue
              // SSE format: "data: text"
              if (trimmed.startsWith('data: ')) {
                fullText += trimmed.slice(6)
              } else {
                // Plain text stream — append directly
                fullText += line
              }
            }
            setStreamingText(fullText)
          }
        }

        // Finalize: write streamed text into last assistant message
        setMessages((m) => {
          const updated = [...m]
          const lastIdx = updated.map((msg, i) => msg.role === 'assistant' ? i : -1).filter(i => i >= 0).pop()
          if (lastIdx !== undefined) updated[lastIdx] = { ...updated[lastIdx], content: fullText }
          return updated
        })
        setStreamingText('')
        setLlmLoading(false)

        // Voice output
        if (fullText) speakText(fullText)
      }
    } catch (err) {
      console.error('Query failed:', err)
      setMessages((m) => [...m, {
        role: 'assistant',
        content: 'Could not reach Omnex API. Make sure the backend is running on port 8000.',
        timestamp: new Date(),
      }])
    } finally {
      setLoading(false)
    }
  }

  function submit() {
    handleQuery(input.trim())
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  function toggleVoice() {
    if (listening) { recognRef.current?.stop(); setListening(false); return }
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    const recog: any = new SR()
    recog.lang = 'en-US'
    recog.interimResults = true
    recog.continuous = false
    recog.onresult = (e: any) => {
      setInput(Array.from(e.results).map((r: any) => r[0].transcript).join(''))
    }
    recog.onend = () => {
      setListening(false)
      const q = inputRef.current?.value.trim()
      if (q) handleQuery(q)
    }
    recog.onerror = () => setListening(false)
    recognRef.current = recog
    recog.start()
    setListening(true)
  }

  // The last assistant message index for streaming
  const lastAssistantIdx = messages.map((m, i) => m.role === 'assistant' ? i : -1).filter((i) => i >= 0).pop()

  // Show setup screen until models are confirmed ready
  if (setupDone === false) {
    return <SetupScreen onComplete={() => {
      setSetupDone(true)
      apiFetch(`${API}/setup/status`).then(r => r.json()).then(d => { if (d.models) setModelStatuses(d.models) }).catch(() => {})
    }} />
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-omnex-bg hex-pattern">

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        style={{
          width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column',
          borderRight: '1px solid #1a1a2e',
          background: 'rgba(5,5,7,0.95)',
          backdropFilter: 'blur(20px)',
          position: 'relative', zIndex: 20,
        }}
      >
        {/* Logo */}
        <div style={{ padding: '20px 16px 12px', borderBottom: '1px solid #1a1a2e' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8,
              background: 'linear-gradient(135deg, rgba(124,106,247,0.3) 0%, rgba(124,106,247,0.1) 100%)',
              border: '1px solid rgba(124,106,247,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 12px rgba(124,106,247,0.2)',
            }}>
              <Brain size={13} color="#a78bfa" />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#e8e8f0', letterSpacing: '0.04em' }}>Omnex</div>
              <div style={{ fontSize: 9, color: '#505068', letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: 1 }}>Agentic Memory</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '8px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <NavItem icon={<Search size={14} />} label="Recall" active={view === 'recall'} onClick={() => setView('recall')} />
          <NavItem icon={<Database size={14} />} label="Ingest" active={view === 'ingest'} onClick={() => setView('ingest')} />
          <NavItem icon={<Users size={14} />} label="People" active={view === 'people'} onClick={() => { setView('people'); setShowIdentity(true) }} />
          <NavItem icon={<Clock size={14} />} label="Timeline" active={view === 'timeline'} onClick={() => setView('timeline')} badge="soon" />
          <NavItem icon={<Zap size={14} />} label="Remote Access" active={view === 'remote'} onClick={() => setView('remote' as NavView)} />

          <div style={{ margin: '8px 0', borderTop: '1px solid #1a1a2e' }} />

          <NavItem icon={<Settings size={14} />} label="Settings" active={false} onClick={() => {}} />
        </nav>

        {/* Index stats panel */}
        <div style={{ padding: '12px 12px', borderTop: '1px solid #1a1a2e', marginBottom: 4 }}>
          <div style={{ fontSize: 9, letterSpacing: '0.15em', color: '#505068', textTransform: 'uppercase', marginBottom: 8 }}>Index</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            {[
              { icon: <Image size={10} />, label: 'Photos',    count: stats?.images    ?? '—', color: '#60a5fa' },
              { icon: <Video size={10} />, label: 'Video',     count: stats?.videos    ?? '—', color: '#a78bfa' },
              { icon: <Music size={10} />, label: 'Audio',     count: stats?.audio     ?? '—', color: '#34d399' },
              { icon: <FileText size={10} />, label: 'Docs',   count: stats?.documents ?? '—', color: '#fbbf24' },
              { icon: <Code2 size={10} />, label: 'Code',      count: stats?.code      ?? '—', color: '#f87171' },
            ].map(({ icon, label, count, color }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#505068' }}>
                  <span style={{ color }}>{icon}</span>
                  <span style={{ fontSize: 11, color: '#505068' }}>{label}</span>
                </div>
                <span style={{ fontSize: 11, color: '#383850', fontFamily: 'JetBrains Mono, monospace' }}>
                  {typeof count === 'number' ? count.toLocaleString() : count}
                </span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #1a1a2e' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{
                width: 5, height: 5, borderRadius: '50%',
                background: apiReady === null ? '#505068' : apiReady ? '#34d399' : '#f87171',
                boxShadow: apiReady === true ? '0 0 5px #34d399' : apiReady === false ? '0 0 5px #f87171' : 'none',
                transition: 'background 0.3s, box-shadow 0.3s',
              }} />
              <span style={{ fontSize: 10, color: '#505068' }}>
                {apiReady === null ? 'Connecting…' : apiReady ? `${(stats?.total ?? 0).toLocaleString()} memories` : 'API offline'}
              </span>
            </div>
          </div>
        </div>

        {/* Model status panel — always shown */}
        <div style={{ padding: '10px 12px 18px', borderTop: '1px solid #1a1a2e' }}>
          <div style={{ fontSize: 9, letterSpacing: '0.15em', color: '#505068', textTransform: 'uppercase', marginBottom: 6 }}>System Models</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {modelStatuses.length > 0
              ? modelStatuses.map((m) => (
                <div key={m.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 10, color: '#383850' }}>{MODEL_DISPLAY[m.id] || m.name}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{
                      width: 4, height: 4, borderRadius: '50%',
                      background: m.status === 'ready' ? '#34d399' : m.status === 'downloading' ? '#a78bfa' : '#383850',
                      boxShadow: m.status === 'ready' ? '0 0 4px #34d399' : m.status === 'downloading' ? '0 0 4px #a78bfa' : 'none',
                      transition: 'background 0.3s',
                    }} />
                    <span style={{ fontSize: 9, color: '#383850', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      {m.status === 'ready' ? 'ready' : m.status === 'downloading' ? 'loading' : 'pending'}
                    </span>
                  </div>
                </div>
              ))
              : Object.entries(MODEL_DISPLAY).map(([id, label]) => (
                <div key={id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 10, color: '#383850' }}>{label}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#252540' }} />
                    <span style={{ fontSize: 9, color: '#252540', textTransform: 'uppercase', letterSpacing: '0.06em' }}>—</span>
                  </div>
                </div>
              ))
            }
          </div>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* Ambient background glow */}
        <div style={{
          position: 'absolute', top: -120, left: '30%', width: 600, height: 400,
          background: 'radial-gradient(ellipse, rgba(124,106,247,0.06) 0%, transparent 70%)',
          pointerEvents: 'none', zIndex: 0,
        }} />

        <AnimatePresence mode="wait">
          {/* ── INGEST VIEW ─────────────────────────────────────────── */}
          {view === 'ingest' ? (
            <motion.div
              key="ingest"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              style={{ flex: 1, overflow: 'hidden', position: 'relative', zIndex: 1 }}
            >
              <IngestionPanel onDone={() => { setView('recall'); fetchStats() }} />
            </motion.div>

          ) : view === 'recall' ? (
            /* ── RECALL VIEW ──────────────────────────────────────── */
            <motion.div
              key="recall"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative', zIndex: 1 }}
            >
              {/* Top bar */}
              <div style={{
                flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '0 20px', height: 44,
                borderBottom: '1px solid #1a1a2e',
                background: 'rgba(5,5,7,0.8)',
                backdropFilter: 'blur(12px)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Search size={12} color="#505068" />
                  <span style={{ fontSize: 12, color: '#505068' }}>
                    {messages.length === 0 ? 'Memory Recall' : `${messages.filter(m => m.role === 'user').length} quer${messages.filter(m => m.role === 'user').length === 1 ? 'y' : 'ies'} this session`}
                  </span>
                </div>
                {messages.length > 0 && (
                  <button
                    onClick={() => { setMessages([]); setSelectedChunk(null) }}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      padding: '4px 10px', borderRadius: 6,
                      background: 'transparent', border: '1px solid #1a1a2e',
                      color: '#505068', fontSize: 11, cursor: 'pointer',
                      fontFamily: 'inherit',
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#e8e8f0'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#252540' }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#505068'; (e.currentTarget as HTMLButtonElement).style.borderColor = '#1a1a2e' }}
                  >
                    <X size={10} /> New session
                  </button>
                )}
              </div>

              {/* Messages + Results area */}
              <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

                {/* Chat column */}
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: messages.length === 0 ? '0' : '16px 24px' }}>

                    {/* Empty state */}
                    {messages.length === 0 && (
                      <div style={{
                        minHeight: '100%', width: '100%', display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        padding: '32px 48px 24px', position: 'relative', boxSizing: 'border-box',
                      }}>
                        {/* Ambient glow behind everything */}
                        <div style={{
                          position: 'absolute', top: '20%', left: '50%', transform: 'translateX(-50%)',
                          width: 500, height: 300, pointerEvents: 'none',
                          background: 'radial-gradient(ellipse, rgba(124,106,247,0.08) 0%, transparent 70%)',
                          filter: 'blur(40px)',
                        }} />
                        {/* Fine grid */}
                        <div style={{
                          position: 'absolute', inset: 0, pointerEvents: 'none',
                          backgroundImage: 'linear-gradient(rgba(124,106,247,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(124,106,247,0.02) 1px, transparent 1px)',
                          backgroundSize: '56px 56px',
                          maskImage: 'radial-gradient(ellipse 80% 70% at 50% 40%, black, transparent)',
                        }} />

                        {/* Orbital animation */}
                        <div style={{ position: 'relative', marginBottom: 32, width: 110, height: 110, flexShrink: 0 }}>
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 36, repeat: Infinity, ease: 'linear' }}
                            style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '1px dashed rgba(124,106,247,0.12)' }}
                          >
                            <div style={{
                              position: 'absolute', top: -3, left: '50%', marginLeft: -3,
                              width: 6, height: 6, borderRadius: '50%', background: '#7c6af7',
                              boxShadow: '0 0 8px #7c6af7, 0 0 18px rgba(124,106,247,0.5)',
                            }} />
                          </motion.div>
                          <motion.div
                            animate={{ rotate: -360 }}
                            transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                            style={{ position: 'absolute', inset: 18, borderRadius: '50%', border: '1px solid rgba(96,165,250,0.1)' }}
                          >
                            <div style={{
                              position: 'absolute', bottom: -2, left: '50%', marginLeft: -2,
                              width: 4, height: 4, borderRadius: '50%', background: '#60a5fa',
                              boxShadow: '0 0 6px #60a5fa',
                            }} />
                          </motion.div>
                          <motion.div
                            animate={{ boxShadow: [
                              '0 0 28px rgba(124,106,247,0.22), 0 0 56px rgba(124,106,247,0.07)',
                              '0 0 52px rgba(124,106,247,0.48), 0 0 90px rgba(124,106,247,0.14)',
                              '0 0 28px rgba(124,106,247,0.22), 0 0 56px rgba(124,106,247,0.07)',
                            ]}}
                            transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
                            style={{
                              position: 'absolute', inset: 30, borderRadius: '50%',
                              background: 'radial-gradient(circle at 35% 30%, rgba(192,168,255,0.28) 0%, rgba(124,106,247,0.1) 50%, rgba(5,5,7,0.9) 100%)',
                              border: '1px solid rgba(124,106,247,0.38)',
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                            }}
                          >
                            <Brain size={18} color="#c4b5fd" strokeWidth={1.2} />
                          </motion.div>
                        </div>

                        {/* Wordmark */}
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0, duration: 0.5 }}
                          style={{
                            fontSize: 10, letterSpacing: '0.4em', textTransform: 'uppercase',
                            color: 'rgba(124,106,247,0.4)', fontWeight: 600, marginBottom: 24,
                          }}
                        >
                          Memory that thinks. Data that speaks.
                        </motion.div>

                        {/* Headline */}
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                          style={{ textAlign: 'center', marginBottom: 18, width: '100%', maxWidth: 580 }}
                        >
                          <h1 style={{
                            fontSize: 'clamp(2rem, 3.5vw, 3.4rem)',
                            fontWeight: 900, letterSpacing: '-0.05em', lineHeight: 1.08, margin: 0,
                          }}>
                            {/* <span style={{ color: '#e8e8f4' }}>Everything you know,</span><br />
                            <span style={{
                              background: 'linear-gradient(110deg, #c4b5fd 0%, #a78bfa 40%, #60a5fa 100%)',
                              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'inline',
                            }}>
                              always within reach.
                            </span> */}
                          </h1>
                        </motion.div>

                        {/* Sub */}
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.3, duration: 0.6 }}
                          style={{
                            fontSize: 14, color: '#40405e', textAlign: 'center',
                            lineHeight: 1.8, maxWidth: 420, marginBottom: 40, letterSpacing: '-0.01em',
                          }}
                        >
                          Every photo, file, recording, and thought — indexed by meaning.<br />
                          <span style={{ color: '#54547a' }}>Ask anything. In plain English. Right now.</span>
                        </motion.p>

                        {/* Suggestion chips */}
                        <motion.div
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.35, duration: 0.5 }}
                          style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 540 }}
                        >
                          {[
                            'Photos from last summer',
                            'Show me everything from last week',
                            'That meeting about the product launch',
                            'All my invoices and receipts',
                            'Code that handles authentication',
                          ].map((s, i) => (
                            <motion.button
                              key={s}
                              initial={{ opacity: 0, scale: 0.95 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.4 + i * 0.06, duration: 0.3 }}
                              whileHover={{ scale: 1.03, y: -2 }}
                              whileTap={{ scale: 0.97 }}
                              onClick={() => handleQuery(s)}
                              style={{
                                padding: '8px 16px', borderRadius: 22,
                                border: '1px solid rgba(37,37,64,0.9)',
                                background: 'rgba(10,10,20,0.7)',
                                color: '#7070a0', fontSize: 12,
                                cursor: 'pointer', fontFamily: 'inherit',
                                transition: 'all 0.15s',
                                backdropFilter: 'blur(8px)',
                              }}
                              onMouseEnter={(e) => {
                                const b = e.currentTarget as HTMLButtonElement
                                b.style.borderColor = 'rgba(124,106,247,0.5)'
                                b.style.color = '#e8e8f0'
                                b.style.background = 'rgba(124,106,247,0.08)'
                                b.style.boxShadow = '0 0 16px rgba(124,106,247,0.1)'
                              }}
                              onMouseLeave={(e) => {
                                const b = e.currentTarget as HTMLButtonElement
                                b.style.borderColor = 'rgba(37,37,64,0.9)'
                                b.style.color = '#7070a0'
                                b.style.background = 'rgba(10,10,20,0.7)'
                                b.style.boxShadow = 'none'
                              }}
                            >
                              {s}
                            </motion.button>
                          ))}
                        </motion.div>
                      </div>
                    )}

                    {/* Message thread */}
                    {messages.length > 0 && (
                      <div style={{ maxWidth: 760, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
                        <AnimatePresence initial={false}>
                          {messages.map((msg, idx) => (
                            <motion.div
                              key={idx}
                              initial={{ opacity: 0, y: 12 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                            >
                              {msg.role === 'user' ? (
                                /* User message */
                                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                                  <div style={{
                                    maxWidth: '72%',
                                    background: 'rgba(124,106,247,0.12)',
                                    border: '1px solid rgba(124,106,247,0.2)',
                                    borderRadius: '16px 16px 4px 16px',
                                    padding: '10px 16px',
                                    color: '#e8e8f0', fontSize: 14, lineHeight: 1.6,
                                  }}>
                                    {msg.content}
                                  </div>
                                </div>
                              ) : (
                                /* Assistant message */
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                  {/* LLM text response */}
                                  {(idx === lastAssistantIdx && (streamingText || llmLoading)) ? (
                                    <div style={{
                                      background: 'rgba(10,10,15,0.8)',
                                      border: '1px solid #1a1a2e',
                                      borderRadius: '4px 16px 16px 16px',
                                      padding: '12px 16px',
                                    }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                                        <div style={{
                                          width: 16, height: 16, borderRadius: 4,
                                          background: 'linear-gradient(135deg, rgba(124,106,247,0.4), rgba(124,106,247,0.1))',
                                          border: '1px solid rgba(124,106,247,0.3)',
                                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        }}>
                                          <Brain size={9} color="#a78bfa" />
                                        </div>
                                        <span style={{ fontSize: 10, color: '#505068', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Omnex</span>
                                        {llmLoading && (
                                          <div style={{ display: 'flex', gap: 3, marginLeft: 4 }}>
                                            {[0,1,2].map((i) => (
                                              <motion.div key={i}
                                                style={{ width: 3, height: 3, borderRadius: '50%', background: '#7c6af7' }}
                                                animate={{ opacity: [0.3, 1, 0.3] }}
                                                transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                                              />
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                      <p style={{ fontSize: 14, lineHeight: 1.7, color: '#e8e8f0' }}>
                                        {streamingText}
                                        {llmLoading && <span style={{ display: 'inline-block', width: 2, height: 14, background: '#7c6af7', marginLeft: 2, verticalAlign: 'middle', animation: 'blink 1s step-end infinite' }} />}
                                      </p>
                                    </div>
                                  ) : msg.content ? (
                                    <div style={{
                                      background: 'rgba(10,10,15,0.8)',
                                      border: '1px solid #1a1a2e',
                                      borderRadius: '4px 16px 16px 16px',
                                      padding: '12px 16px',
                                    }}>
                                      <p style={{ fontSize: 14, lineHeight: 1.7, color: '#e8e8f0' }}>{msg.content}</p>
                                    </div>
                                  ) : null}

                                  {/* Results */}
                                  {msg.results && msg.results.length > 0 && (
                                    <div>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                                        <div style={{ height: 1, flex: 1, background: '#1a1a2e' }} />
                                        <span style={{ fontSize: 10, color: '#505068', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                                          {msg.total} result{msg.total !== 1 ? 's' : ''}
                                        </span>
                                        <div style={{ height: 1, flex: 1, background: '#1a1a2e' }} />
                                      </div>
                                      <ResultGrid results={msg.results} onSelect={setSelectedChunk} selected={selectedChunk} />
                                    </div>
                                  )}

                                  {/* Refinement pills */}
                                  {msg.refinements && msg.refinements.length > 0 && (
                                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                      {msg.refinements.map((r) => (
                                        <button
                                          key={r}
                                          onClick={() => handleQuery(r)}
                                          style={{
                                            padding: '5px 12px', borderRadius: 20,
                                            border: '1px solid #1a1a2e',
                                            background: 'transparent',
                                            color: '#505068', fontSize: 11,
                                            cursor: 'pointer', fontFamily: 'inherit',
                                            transition: 'all 0.15s',
                                          }}
                                          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = '#252540'; (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
                                          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = '#1a1a2e'; (e.currentTarget as HTMLButtonElement).style.color = '#505068' }}
                                        >
                                          → {r}
                                        </button>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        {/* Loading state */}
                        {loading && (
                          <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0' }}
                          >
                            <div style={{ display: 'flex', gap: 4 }}>
                              {[0,1,2].map((i) => (
                                <motion.div key={i}
                                  style={{ width: 4, height: 4, borderRadius: '50%', background: '#7c6af7' }}
                                  animate={{ opacity: [0.2, 1, 0.2], scale: [0.8, 1.2, 0.8] }}
                                  transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
                                />
                              ))}
                            </div>
                            <span style={{ fontSize: 12, color: '#505068' }}>Searching memory…</span>
                          </motion.div>
                        )}
                        <div ref={messagesEndRef} />
                      </div>
                    )}
                  </div>

                  {/* Input bar */}
                  <div style={{
                    flexShrink: 0, padding: '12px 24px 20px',
                    background: 'linear-gradient(to top, rgba(5,5,7,1) 0%, rgba(5,5,7,0.9) 60%, transparent 100%)',
                  }}>
                    <div style={{ maxWidth: 760, margin: '0 auto' }}>
                      <InputBar
                        value={input}
                        onChange={setInput}
                        onSubmit={submit}
                        onKey={(e) => handleKey(e as any)}
                        onFocus={() => setFocused(true)}
                        onBlur={() => setFocused(false)}
                        focused={focused}
                        loading={loading}
                        listening={listening}
                        voiceSupport={voiceSupport}
                        onVoice={toggleVoice}
                        inputRef={inputRef}
                        ttsSupport={ttsSupport}
                        ttsEnabled={ttsEnabled}
                        onTtsToggle={() => {
                          setTtsEnabled(v => !v)
                          if (ttsEnabled) window.speechSynthesis?.cancel()
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Preview pane */}
                <AnimatePresence>
                  {selectedChunk && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                      style={{
                        width: 380, flexShrink: 0,
                        borderLeft: '1px solid #1a1a2e',
                        overflow: 'hidden',
                      }}
                    >
                      <PreviewPane chunk={selectedChunk} onClose={() => setSelectedChunk(null)} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

          ) : view === 'timeline' ? (
            /* ── TIMELINE (coming soon) ─────────────────────────── */
            <motion.div
              key="timeline"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', zIndex: 1 }}
            >
              <div style={{ textAlign: 'center' }}>
                <Clock size={40} color="#252540" strokeWidth={1} style={{ marginBottom: 16 }} />
                <p style={{ fontSize: 14, color: '#383850', marginBottom: 6 }}>Timeline view coming soon</p>
                <p style={{ fontSize: 12, color: '#252540' }}>Your memories, organized by time and place.</p>
              </div>
            </motion.div>

          ) : view === 'remote' ? (
            /* ── REMOTE ACCESS ───────────────────────────────────── */
            <motion.div
              key="remote"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ flex: 1, overflow: 'auto', padding: '32px 32px', position: 'relative', zIndex: 1 }}
            >
              <RemoteAccessPanel api={API} />
            </motion.div>

          ) : null}
        </AnimatePresence>
      </div>

      {/* Identity manager modal */}
      <AnimatePresence>
        {showIdentity && (
          <IdentityManager onClose={() => { setShowIdentity(false); setView('recall') }} />
        )}
      </AnimatePresence>
    </div>
  )
}

/* ── Remote Access Panel ───────────────────────────────────────────────────── */
function RemoteAccessPanel({ api }: { api: string }) {
  const [tunnel, setTunnel]     = useState<any>(null)
  const [copied, setCopied]     = useState<string | null>(null)
  const [, setPolling]          = useState(true)

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>
    async function poll() {
      try {
        const res  = await fetch(`${api}/setup/tunnel`)
        const data = await res.json()
        setTunnel(data)
        if (data.status === 'active' || data.status === 'error' || data.status === 'disabled') {
          setPolling(false)
          clearInterval(interval)
        }
      } catch {}
    }
    poll()
    interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [])

  function copy(text: string, key: string) {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(key)
      setTimeout(() => setCopied(null), 1800)
    })
  }

  const url     = tunnel?.url
  const enabled = tunnel?.auth_enabled
  const apiKey  = tunnel?.api_key || 'YOUR_KEY'
  const curlCmd = url
    ? `curl -H "X-API-Key: ${apiKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query":"show me recent photos"}' \\\n  ${url}/query`
    : ''

  const mcpConfig = url ? JSON.stringify({
    mcpServers: {
      omnex: {
        url:     `${url}/mcp`,
        headers: { "X-API-Key": apiKey },
      }
    }
  }, null, 2) : ''

  return (
    <div style={{ maxWidth: 680 }}>
      <div style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>Remote Access</h2>
        <p style={{ fontSize: 13, color: '#505068' }}>
          Expose Omnex to the internet via an ngrok tunnel. Set <code style={{ color: '#a78bfa', fontSize: 12 }}>NGROK_AUTHTOKEN</code> in your docker-compose env to activate.
        </p>
      </div>

      {/* Tunnel status card */}
      <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '20px 24px', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: tunnel?.status === 'active' ? '#34d399' : tunnel?.status === 'starting' ? '#fbbf24' : tunnel?.status === 'error' ? '#f87171' : '#383850',
            boxShadow: tunnel?.status === 'active' ? '0 0 8px #34d399' : 'none',
          }} />
          <span style={{ fontSize: 13, fontWeight: 500, color: '#e8e8f0' }}>
            {tunnel?.status === 'active'   ? 'Tunnel active'
           : tunnel?.status === 'starting' ? 'Opening tunnel…'
           : tunnel?.status === 'error'    ? 'Tunnel error'
           : tunnel?.status === 'disabled' ? 'Tunnel disabled'
           : 'Checking…'}
          </span>
        </div>

        {url && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <code style={{ flex: 1, fontSize: 13, color: '#a78bfa', background: 'rgba(124,106,247,0.08)', padding: '8px 12px', borderRadius: 8, wordBreak: 'break-all' }}>
              {url}
            </code>
            <button onClick={() => copy(url, 'url')} style={{ padding: '8px 14px', borderRadius: 8, background: 'rgba(124,106,247,0.12)', border: '1px solid rgba(124,106,247,0.2)', color: '#a78bfa', fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap' }}>
              {copied === 'url' ? 'Copied!' : 'Copy URL'}
            </button>
          </div>
        )}

        {tunnel?.status === 'disabled' && (
          <div style={{ fontSize: 12, color: '#505068', lineHeight: 1.7 }}>
            Add to your <code style={{ color: '#a78bfa' }}>docker-compose.yml</code> environment:<br />
            <code style={{ display: 'block', marginTop: 8, padding: '10px 12px', background: 'rgba(124,106,247,0.06)', borderRadius: 8, color: '#e8e8f0', fontSize: 11 }}>
              NGROK_AUTHTOKEN: your_token_here<br />
              OMNEX_API_KEY: your_secret_key
            </code>
          </div>
        )}

        {tunnel?.error && (
          <p style={{ fontSize: 12, color: '#f87171', marginTop: 8 }}>{tunnel.error}</p>
        )}
      </div>

      {/* Auth status */}
      <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '16px 24px', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: enabled ? '#34d399' : '#fbbf24' }} />
          <span style={{ fontSize: 13, color: '#e8e8f0', fontWeight: 500 }}>
            API Key Auth — {enabled ? 'enabled' : 'disabled (local mode)'}
          </span>
        </div>
        <p style={{ fontSize: 12, color: '#505068', marginTop: 4 }}>
          {enabled ? 'All external requests require X-API-Key header.' : 'Set OMNEX_API_KEY env var to require authentication.'}
        </p>
      </div>

      {/* Connect instructions */}
      {url && (
        <>
          <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '16px 24px', marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 12, color: '#505068', textTransform: 'uppercase', letterSpacing: '0.1em' }}>REST API</span>
              <button onClick={() => copy(curlCmd, 'curl')} style={{ padding: '4px 10px', borderRadius: 6, background: 'transparent', border: '1px solid #1a1a2e', color: '#505068', fontSize: 11, cursor: 'pointer' }}>
                {copied === 'curl' ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <pre style={{ fontSize: 11, color: '#a78bfa', margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{curlCmd}</pre>
          </div>

          <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '16px 24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 12, color: '#505068', textTransform: 'uppercase', letterSpacing: '0.1em' }}>MCP (Claude / Cursor / Windsurf)</span>
              <button onClick={() => copy(mcpConfig, 'mcp')} style={{ padding: '4px 10px', borderRadius: 6, background: 'transparent', border: '1px solid #1a1a2e', color: '#505068', fontSize: 11, cursor: 'pointer' }}>
                {copied === 'mcp' ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <pre style={{ fontSize: 11, color: '#34d399', margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{mcpConfig}</pre>
            <p style={{ fontSize: 11, color: '#383850', marginTop: 10 }}>Paste into your Claude Desktop / Cursor mcp.json config.</p>
          </div>
        </>
      )}
    </div>
  )
}


/* ── Nav item ──────────────────────────────────────────────────────────────── */
function NavItem({ icon, label, active, onClick, badge }: {
  icon: React.ReactNode
  label: string
  active: boolean
  onClick: () => void
  badge?: string
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 9,
        padding: '7px 10px', borderRadius: 8,
        border: 'none', cursor: 'pointer', width: '100%', textAlign: 'left',
        background: active ? 'rgba(124,106,247,0.1)' : 'transparent',
        color: active ? '#a78bfa' : '#505068',
        fontSize: 13, fontFamily: 'inherit',
        transition: 'all 0.12s',
        position: 'relative',
      }}
      onMouseEnter={(e) => { if (!active) (e.currentTarget as HTMLButtonElement).style.color = '#e8e8f0' }}
      onMouseLeave={(e) => { if (!active) (e.currentTarget as HTMLButtonElement).style.color = '#505068' }}
    >
      {active && (
        <div style={{
          position: 'absolute', left: 0, top: '20%', bottom: '20%',
          width: 2, borderRadius: 2, background: '#7c6af7',
          boxShadow: '0 0 6px #7c6af7',
        }} />
      )}
      {icon}
      <span>{label}</span>
      {badge && (
        <span style={{
          marginLeft: 'auto', fontSize: 9, padding: '2px 5px',
          borderRadius: 4, background: 'rgba(124,106,247,0.1)',
          border: '1px solid rgba(124,106,247,0.15)',
          color: '#505068', letterSpacing: '0.08em', textTransform: 'uppercase',
        }}>{badge}</span>
      )}
    </button>
  )
}

/* ── Input bar ─────────────────────────────────────────────────────────────── */
function InputBar({
  value, onChange, onSubmit, onKey, onFocus, onBlur,
  focused, loading, listening, voiceSupport, onVoice, inputRef,
  ttsSupport, ttsEnabled, onTtsToggle,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onKey: (e: React.KeyboardEvent) => void
  onFocus: () => void
  onBlur: () => void
  focused: boolean
  loading: boolean
  listening: boolean
  voiceSupport: boolean
  onVoice: () => void
  inputRef: React.RefObject<HTMLTextAreaElement>
  ttsSupport?: boolean
  ttsEnabled?: boolean
  onTtsToggle?: () => void
}) {
  const hasValue = value.trim().length > 0
  const borderColor = listening
    ? 'rgba(248,113,113,0.45)'
    : focused
    ? 'rgba(124,106,247,0.45)'
    : 'rgba(37,37,64,0.8)'

  return (
    <motion.div
      animate={{
        boxShadow: focused || listening
          ? `0 0 0 1px ${borderColor}, 0 0 0 4px ${listening ? 'rgba(248,113,113,0.06)' : 'rgba(124,106,247,0.06)'}, 0 16px 48px rgba(0,0,0,0.6)`
          : '0 0 0 1px rgba(26,26,46,0.9), 0 8px 32px rgba(0,0,0,0.5)',
      }}
      transition={{ duration: 0.18 }}
      style={{
        borderRadius: 20,
        background: 'rgba(12,12,18,0.98)',
        display: 'flex', flexDirection: 'column',
        padding: '4px 8px 0 18px',
        position: 'relative',
      }}
    >
      {/* Main row: textarea + right-side buttons */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6 }}>
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKey}
          onFocus={onFocus}
          onBlur={onBlur}
          placeholder={listening ? 'Listening…' : 'Ask anything about your data…'}
          rows={1}
          style={{
            flex: 1, paddingTop: 16, paddingBottom: 10, paddingRight: 8,
            background: 'transparent', border: 'none', outline: 'none', resize: 'none',
            color: '#e8e8f0', fontSize: 15, lineHeight: 1.6,
            minHeight: 56, maxHeight: 200, overflowY: 'auto',
            fontFamily: 'inherit',
          }}
          onInput={(e) => {
            const el = e.currentTarget
            el.style.height = 'auto'
            el.style.height = `${el.scrollHeight}px`
          }}
        />

        {/* Button cluster */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, paddingBottom: 5, flexShrink: 0 }}>
          {voiceSupport && (
            <motion.button
              whileTap={{ scale: 0.88 }}
              onClick={onVoice}
              title={listening ? 'Stop listening' : 'Voice input'}
              style={{
                width: 34, height: 34, borderRadius: 12, flexShrink: 0,
                background: listening ? 'rgba(248,113,113,0.1)' : 'transparent',
                border: listening ? '1px solid rgba(248,113,113,0.25)' : '1px solid transparent',
                cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: listening ? '#f87171' : '#383850',
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => { if (!listening) (e.currentTarget as HTMLButtonElement).style.color = '#7c6af7' }}
              onMouseLeave={(e) => { if (!listening) (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
            >
              {listening ? <MicOff size={15} /> : <Mic size={15} />}
            </motion.button>
          )}

          {ttsSupport && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.92 }}
              onClick={onTtsToggle}
              title={ttsEnabled ? 'Voice output on' : 'Voice output off'}
              style={{
                width: 34, height: 34, borderRadius: 12, flexShrink: 0,
                background: ttsEnabled ? 'rgba(52,211,153,0.08)' : 'transparent',
                border: ttsEnabled ? '1px solid rgba(52,211,153,0.2)' : '1px solid transparent',
                cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: ttsEnabled ? '#34d399' : '#383850',
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => { if (!ttsEnabled) (e.currentTarget as HTMLButtonElement).style.color = '#34d399' }}
              onMouseLeave={(e) => { if (!ttsEnabled) (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
            >
              <Volume2 size={15} />
            </motion.button>
          )}

          <motion.button
            whileHover={hasValue && !loading ? { scale: 1.05 } : {}}
            whileTap={hasValue && !loading ? { scale: 0.92 } : {}}
            onClick={onSubmit}
            disabled={!hasValue || loading}
            style={{
              width: 34, height: 34, borderRadius: 12, flexShrink: 0,
              background: hasValue && !loading
                ? 'linear-gradient(145deg, #8b7cf8, #6b5ce7)'
                : 'rgba(124,106,247,0.08)',
              border: hasValue && !loading
                ? '1px solid rgba(124,106,247,0.3)'
                : '1px solid rgba(124,106,247,0.08)',
              cursor: (!hasValue || loading) ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: hasValue && !loading ? 'white' : 'rgba(124,106,247,0.3)',
              boxShadow: hasValue && !loading ? '0 0 20px rgba(124,106,247,0.35), inset 0 1px 0 rgba(255,255,255,0.15)' : 'none',
              transition: 'all 0.15s',
            }}
          >
            {loading
              ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
              : <ArrowUp size={15} strokeWidth={2.5} />
            }
          </motion.button>
        </div>
      </div>

      {/* Fixed-height hint — always present to prevent layout shift */}
      <div style={{ paddingBottom: 10 }}>
        <span style={{ fontSize: 10, color: focused || hasValue ? 'transparent' : '#252540', transition: 'color 0.2s' }}>
          Enter to send · Shift+Enter for new line{voiceSupport ? ' · Mic for voice' : ''}
        </span>
      </div>
    </motion.div>
  )
}
