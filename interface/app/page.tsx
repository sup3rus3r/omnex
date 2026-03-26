'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Database, Users, Clock, Settings, Mic, ArrowUp, Loader2, X, ChevronRight, Zap, FileText, Image, Video, Music, Code2, Search } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
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

type NavView = 'recall' | 'ingest' | 'people' | 'timeline' | 'remote' | 'settings'

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
  const [isSpeaking,    setIsSpeaking]    = useState(false)
  const [alwaysListen,  setAlwaysListen]  = useState(false)
  const [micLevel,      setMicLevel]      = useState(0)   // 0-1 amplitude
  const mediaRecorderRef  = useRef<MediaRecorder | null>(null)
  const audioChunksRef    = useRef<Blob[]>([])
  const analyserRef       = useRef<AnalyserNode | null>(null)
  const micStreamRef      = useRef<MediaStream | null>(null)
  const vadTimerRef       = useRef<ReturnType<typeof setTimeout> | null>(null)
  const vadActiveRef      = useRef(false)
  const alwaysListenRef   = useRef(false)
  const [stats,         setStats]         = useState<IndexStats | null>(null)
  const [apiReady,      setApiReady]      = useState<boolean | null>(null)
  const [focused,       setFocused]       = useState(false)
  const [sessionId,     setSessionId]     = useState<string | null>(null)
  const [streamingText, setStreamingText] = useState('')
  const [ingestToast,   setIngestToast]   = useState<{path: string, pct: number, eta: number | null, fpm: number | null} | null>(null)
  const [expandNudge,   setExpandNudge]   = useState(false)
  const [nudgeDismissed, setNudgeDismissed] = useState(false)
  const inputRef       = useRef<HTMLTextAreaElement>(null)
const messagesEndRef = useRef<HTMLDivElement>(null)

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

    const hasMediaRecorder = 'MediaRecorder' in window && 'mediaDevices' in navigator
    setVoiceSupport(hasMediaRecorder)

    setTtsSupport(true)
    setTtsEnabled(false)
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
      const total = data.total_chunks || 0
      setStats({
        total,
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

      // Poll active ingestion for toast
      apiFetch(`${API}/ingest/status`)
        .then(r => r.json())
        .then(d => {
          const active = (d.ingestion || []).find((r: any) => r.status === 'running' || r.status === 'processing')
          if (active && active.total_files > 0) {
            const pct = Math.round((active.processed / active.total_files) * 100)
            const name = active.source_path?.split(/[/\\]/).pop() || active.source_path
            setIngestToast({ path: name, pct, eta: active.eta_seconds ?? null, fpm: active.files_per_minute ?? null })
          } else {
            setIngestToast(null)
            // Show drive expansion nudge if index has something but looks like just one folder
            if (total > 0 && total < 5000 && !nudgeDismissed) {
              setExpandNudge(true)
            }
          }
        })
        .catch(() => {})
    } catch { setApiReady(false) }
  }

  const speakText = useCallback(async (text: string) => {
    if (!ttsEnabled) return
    try {
      const res = await apiFetch(`${API}/voice/speak`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
      })
      if (!res.ok) return
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const audio = new Audio(url)
      setIsSpeaking(true)
      audio.onended = () => { setIsSpeaking(false); URL.revokeObjectURL(url) }
      audio.onerror = () => { setIsSpeaking(false); URL.revokeObjectURL(url) }
      audio.play()
    } catch (e) {
      console.warn('TTS failed:', e)
      setIsSpeaking(false)
    }
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

      // Use backend LLM response if available (already filtered + answered)
      // Fall back to /api/chat stream only for pure conversation (no search results)
      {
        setLlmLoading(true)
        let fullText = ''

        if (data.llm_response) {
          // Backend already did LLM filtering — use that response directly
          fullText = data.llm_response
          setStreamingText(fullText)
        } else {
          // Pure conversation — stream via /api/chat
          const chatRes = await fetch('/api/chat', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ query, context: '', messages: history }),
          })

          const reader = chatRes.body?.getReader()
          const decoder = new TextDecoder()

          if (reader) {
            while (true) {
              const { done, value } = await reader.read()
              if (done) break
              fullText += decoder.decode(value, { stream: true })
              setStreamingText(fullText)
            }
          }
        }

        // Finalize: write text into last assistant message
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

  // Start mic stream + level analyser (shared between push-to-talk and always-listen)
  async function _startMicStream(): Promise<MediaStream | null> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      micStreamRef.current = stream

      // Hook up Web Audio analyser for level meter
      const ctx = new AudioContext()
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // Animate mic level
      const buf = new Uint8Array(analyser.frequencyBinCount)
      const tick = () => {
        if (!analyserRef.current) return
        analyserRef.current.getByteTimeDomainData(buf)
        let sum = 0
        for (let i = 0; i < buf.length; i++) sum += Math.abs(buf[i] - 128)
        setMicLevel(Math.min(sum / buf.length / 64, 1))
        requestAnimationFrame(tick)
      }
      tick()
      return stream
    } catch (e) {
      console.warn('Microphone access denied:', e)
      return null
    }
  }

  function _stopMicStream() {
    micStreamRef.current?.getTracks().forEach(t => t.stop())
    micStreamRef.current = null
    analyserRef.current = null
    setMicLevel(0)
  }

  // Pick best supported audio MIME type once
  function _mimeType() {
    return ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'].find(
      (m) => MediaRecorder.isTypeSupported(m)
    ) || ''
  }

  // Send recorded blob to Whisper and run the query
  async function _transcribeAndQuery(blob: Blob) {
    if (blob.size < 1000) return
    const formData = new FormData()
    formData.append('audio', blob, 'recording.webm')
    try {
      const res = await apiFetch(`${API}/voice/transcribe`, { method: 'POST', body: formData })
      if (!res.ok) return
      const data = await res.json()
      const text = (data.text || '').trim()
      if (text) { setInput(text); setTimeout(() => handleQuery(text), 200) }
    } catch {}
  }

  // Push-to-talk: click to start, click again to stop
  async function startPushToTalk() {
    if (listening) {
      mediaRecorderRef.current?.stop()
      return
    }
    const stream = await _startMicStream()
    if (!stream) return

    const mime = _mimeType()
    audioChunksRef.current = []
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    mediaRecorderRef.current = recorder
    recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
    recorder.onstop = async () => {
      _stopMicStream()
      setListening(false)
      const blob = new Blob(audioChunksRef.current, { type: mime || 'audio/webm' })
      await _transcribeAndQuery(blob)
      if (alwaysListenRef.current) setTimeout(startVADLoop, 1500)
    }
    recorder.start()
    setListening(true)
  }

  // VAD loop: always-listen mode — auto-detect speech, record, transcribe, repeat
  async function startVADLoop() {
    if (!alwaysListenRef.current) return
    const stream = await _startMicStream()
    if (!stream) return

    const mime = _mimeType()
    audioChunksRef.current = []
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    mediaRecorderRef.current = recorder
    recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
    recorder.onstop = async () => {
      _stopMicStream()
      setListening(false)
      const blob = new Blob(audioChunksRef.current, { type: mime || 'audio/webm' })
      await _transcribeAndQuery(blob)
      if (alwaysListenRef.current) setTimeout(startVADLoop, 1200)
    }

    const buf = new Uint8Array(analyserRef.current?.frequencyBinCount ?? 128)
    let speaking = false
    let silenceFrames = 0
    const THRESHOLD = 0.025
    const SILENCE_FRAMES = 45  // ~0.75s at 60fps

    const detect = () => {
      if (!alwaysListenRef.current || !analyserRef.current) {
        if (recorder.state === 'recording') recorder.stop()
        return
      }
      analyserRef.current.getByteTimeDomainData(buf)
      let sum = 0
      for (let i = 0; i < buf.length; i++) sum += Math.abs(buf[i] - 128)
      const level = sum / buf.length / 128

      if (level > THRESHOLD) {
        silenceFrames = 0
        if (!speaking && recorder.state === 'inactive') {
          speaking = true
          audioChunksRef.current = []
          recorder.start(100)
          setListening(true)
        }
      } else if (speaking) {
        silenceFrames++
        if (silenceFrames > SILENCE_FRAMES) {
          speaking = false
          silenceFrames = 0
          if (recorder.state === 'recording') { recorder.stop(); return }
        }
      }
      requestAnimationFrame(detect)
    }
    requestAnimationFrame(detect)
  }

  // Toggle always-listen (called from MicOrb long-press)
  async function toggleAlwaysListen() {
    const next = !alwaysListenRef.current
    alwaysListenRef.current = next
    setAlwaysListen(next)
    if (next) {
      startVADLoop()
    } else {
      mediaRecorderRef.current?.stop()
      _stopMicStream()
      setListening(false)
    }
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
          <NavItem icon={<Users size={14} />} label="People" active={view === 'people'} onClick={() => setView('people')} />
          <NavItem icon={<Clock size={14} />} label="Timeline" active={view === 'timeline'} onClick={() => setView('timeline')} />
          <NavItem icon={<Zap size={14} />} label="Remote Access" active={view === 'remote'} onClick={() => setView('remote' as NavView)} />

          <div style={{ margin: '8px 0', borderTop: '1px solid #1a1a2e' }} />

          <NavItem icon={<Settings size={14} />} label="Settings" active={view === 'settings'} onClick={() => setView('settings')} />
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

          ) : view === 'people' ? (
            /* ── PEOPLE VIEW ──────────────────────────────────────── */
            <motion.div
              key="people"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ flex: 1, overflow: 'auto', padding: '32px 32px', position: 'relative', zIndex: 1 }}
            >
              <PeoplePanel api={API} />
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
                      stats?.total === 0
                        ? <EmptyIndexState onIngest={() => setView('ingest')} />
                        : <ReadyState stats={stats} onQuery={handleQuery} />
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
                                      <div style={{ fontSize: 14, lineHeight: 1.7, color: '#e8e8f0' }} className="md-prose">
                                        <ReactMarkdown>{streamingText}</ReactMarkdown>
                                        {llmLoading && <span style={{ display: 'inline-block', width: 2, height: 14, background: '#7c6af7', marginLeft: 2, verticalAlign: 'middle', animation: 'blink 1s step-end infinite' }} />}
                                      </div>
                                    </div>
                                  ) : msg.content ? (
                                    <div style={{
                                      background: 'rgba(10,10,15,0.8)',
                                      border: '1px solid #1a1a2e',
                                      borderRadius: '4px 16px 16px 16px',
                                      padding: '12px 16px',
                                    }}>
                                      <div style={{ fontSize: 14, lineHeight: 1.7, color: '#e8e8f0' }} className="md-prose">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                      </div>
                                    </div>
                                  ) : null}

                                  {/* Results — collapsed by default, expandable */}
                                  {msg.results && msg.results.length > 0 && (
                                    <ExpandableResults results={msg.results} total={msg.total ?? 0} onSelect={setSelectedChunk} selected={selectedChunk} />
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
                        onVoice={startPushToTalk}
                        inputRef={inputRef}
                        ttsSupport={ttsSupport}
                        ttsEnabled={ttsEnabled}
                        onTtsToggle={() => setTtsEnabled(v => !v)}
                        isSpeaking={isSpeaking}
                        micLevel={micLevel}
                        alwaysListen={alwaysListen}
                        onAlwaysListen={toggleAlwaysListen}
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
                      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <div style={{ padding: '10px 12px', borderBottom: '1px solid #1a1a2e', display: 'flex', justifyContent: 'flex-end' }}>
                          <button
                            onClick={async () => {
                              if (!selectedChunk) return
                              if (!confirm('Remove this item from your memory index?')) return
                              const res = await apiFetch(`${API}/ingest/chunk/${selectedChunk.chunk_id}`, { method: 'DELETE' })
                              if (res.ok) {
                                setMessages(m => m.map(msg => ({
                                  ...msg,
                                  results: msg.results?.filter(r => r.chunk_id !== selectedChunk.chunk_id)
                                })))
                                setSelectedChunk(null)
                              }
                            }}
                            style={{
                              display: 'flex', alignItems: 'center', gap: 5,
                              padding: '5px 10px', borderRadius: 6,
                              background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
                              color: '#f87171', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                            }}
                          >
                            <X size={11} /> Remove from index
                          </button>
                        </div>
                        <div style={{ flex: 1, overflow: 'hidden' }}>
                          <PreviewPane chunk={selectedChunk} onClose={() => setSelectedChunk(null)} />
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

          ) : view === 'timeline' ? (
            /* ── TIMELINE ────────────────────────────────────────── */
            <motion.div
              key="timeline"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ flex: 1, overflow: 'auto', padding: '32px 32px', position: 'relative', zIndex: 1 }}
            >
              <TimelinePanel api={API} onSelect={setSelectedChunk} selected={selectedChunk} />
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

          ) : view === 'settings' ? (
            /* ── SETTINGS ────────────────────────────────────────── */
            <motion.div
              key="settings"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ flex: 1, overflow: 'auto', padding: '32px 32px', position: 'relative', zIndex: 1 }}
            >
              <SettingsPanel api={API} />
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

      {/* ── Ingestion progress toast ─────────────────────────────────── */}
      <AnimatePresence>
        {ingestToast && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.96 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: 'fixed', bottom: 28, right: 28, zIndex: 100,
              background: 'rgba(10,10,15,0.96)', backdropFilter: 'blur(16px)',
              border: '1px solid rgba(124,106,247,0.25)', borderRadius: 14,
              padding: '12px 16px', minWidth: 260, maxWidth: 340,
              boxShadow: '0 8px 40px rgba(0,0,0,0.6)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                style={{ flexShrink: 0 }}
              >
                <Loader2 size={13} color="#7c6af7" />
              </motion.div>
              <span style={{ fontSize: 12, color: '#e8e8f0', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Indexing <span style={{ color: '#a78bfa' }}>{ingestToast.path}</span>
              </span>
              <span style={{ fontSize: 11, color: '#505068', fontFamily: 'JetBrains Mono, monospace', flexShrink: 0 }}>
                {ingestToast.pct}%
              </span>
            </div>
            {/* Progress bar */}
            <div style={{ height: 3, borderRadius: 2, background: 'rgba(124,106,247,0.12)', overflow: 'hidden', marginBottom: 6 }}>
              <motion.div
                animate={{ width: `${ingestToast.pct}%` }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
                style={{ height: '100%', background: 'linear-gradient(90deg, #7c6af7, #a78bfa)', borderRadius: 2 }}
              />
            </div>
            {/* ETA row */}
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 10, color: '#383850' }}>
                {ingestToast.eta != null
                  ? ingestToast.eta > 60
                    ? `~${Math.round(ingestToast.eta / 60)} min remaining`
                    : `~${ingestToast.eta}s remaining`
                  : 'Calculating…'}
              </span>
              {ingestToast.fpm != null && (
                <span style={{ fontSize: 10, color: '#252540', fontFamily: 'JetBrains Mono, monospace' }}>
                  {ingestToast.fpm} files/min
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Drive expansion nudge ────────────────────────────────────── */}
      <AnimatePresence>
        {expandNudge && !nudgeDismissed && !ingestToast && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.96 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: 'fixed', bottom: 28, right: 28, zIndex: 100,
              background: 'rgba(10,10,15,0.96)', backdropFilter: 'blur(16px)',
              border: '1px solid rgba(124,106,247,0.2)', borderRadius: 14,
              padding: '14px 16px', minWidth: 280, maxWidth: 340,
              boxShadow: '0 8px 40px rgba(0,0,0,0.6)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <div style={{
                width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                background: 'rgba(124,106,247,0.1)', border: '1px solid rgba(124,106,247,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Database size={13} color="#a78bfa" />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: '#e8e8f0', marginBottom: 4 }}>
                  Expand your memory
                </div>
                <div style={{ fontSize: 11, color: '#505068', lineHeight: 1.6 }}>
                  Add more folders to give Omnex a fuller picture — photos, documents, code, audio.
                </div>
              </div>
              <button
                onClick={() => { setNudgeDismissed(true); setExpandNudge(false) }}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 2, color: '#383850', flexShrink: 0 }}
              >
                <X size={12} />
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button
                onClick={() => { setNudgeDismissed(true); setExpandNudge(false); setView('ingest') }}
                style={{
                  flex: 1, padding: '6px 0', borderRadius: 8,
                  background: 'rgba(124,106,247,0.12)', border: '1px solid rgba(124,106,247,0.25)',
                  color: '#a78bfa', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Add folders
              </button>
              <button
                onClick={() => { setNudgeDismissed(true); setExpandNudge(false) }}
                style={{
                  padding: '6px 14px', borderRadius: 8,
                  background: 'transparent', border: '1px solid #1a1a2e',
                  color: '#505068', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                Dismiss
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ── Timeline Panel ─────────────────────────────────────────────────────────── */
const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const TYPE_ICONS: Record<string, React.ReactNode> = {
  image: <Image size={11} />, video: <Video size={11} />, audio: <Music size={11} />,
  document: <FileText size={11} />, code: <Code2 size={11} />,
}

function TimelinePanel({ api, onSelect, selected }: { api: string, onSelect: (c: any) => void, selected: any }) {
  const [years,    setYears]    = useState<{year: number, count: number}[]>([])
  const [selYear,  setSelYear]  = useState<number | null>(null)
  const [months,   setMonths]   = useState<{month: number, count: number, types: string[]}[]>([])
  const [selMonth, setSelMonth] = useState<number | null>(null)
  const [chunks,   setChunks]   = useState<any[]>([])
  const [total,    setTotal]    = useState(0)
  const [page,     setPage]     = useState(1)
  const [loading,  setLoading]  = useState(false)
  const [typeFilter, setTypeFilter] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${api}/timeline/years`).then(r => r.json()).then(d => {
      setYears(d)
      if (d.length > 0) setSelYear(d[0].year)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selYear) return
    fetch(`${api}/timeline/months?year=${selYear}`).then(r => r.json()).then(d => {
      setMonths(d)
      setSelMonth(d.length > 0 ? d[0].month : null)
    }).catch(() => {})
  }, [selYear])

  useEffect(() => {
    if (!selYear || !selMonth) return
    setLoading(true)
    const params = new URLSearchParams({ year: String(selYear), month: String(selMonth), page: String(page), limit: '40' })
    if (typeFilter) params.set('file_type', typeFilter)
    fetch(`${api}/timeline?${params}`).then(r => r.json()).then(d => {
      setChunks(d.results || [])
      setTotal(d.total || 0)
    }).catch(() => {}).finally(() => setLoading(false))
  }, [selYear, selMonth, page, typeFilter])

  const pillStyle = (active: boolean): React.CSSProperties => ({
    padding: '4px 10px', borderRadius: 20, cursor: 'pointer', fontSize: 11,
    border: `1px solid ${active ? '#7c6af7' : '#1a1a2e'}`,
    background: active ? 'rgba(124,106,247,0.12)' : 'transparent',
    color: active ? '#a78bfa' : '#505068', fontFamily: 'inherit',
  })

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>Timeline</h2>
        <p style={{ fontSize: 13, color: '#505068' }}>Your memories organised by time.</p>
      </div>

      {/* Year selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {years.map(y => (
          <button key={y.year} style={pillStyle(selYear === y.year)} onClick={() => { setSelYear(y.year); setPage(1) }}>
            {y.year} <span style={{ opacity: 0.5 }}>({y.count})</span>
          </button>
        ))}
      </div>

      {/* Month selector */}
      {months.length > 0 && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          {months.map(m => (
            <button key={m.month} style={pillStyle(selMonth === m.month)} onClick={() => { setSelMonth(m.month); setPage(1) }}>
              {MONTH_NAMES[m.month - 1]} <span style={{ opacity: 0.5 }}>({m.count})</span>
            </button>
          ))}
        </div>
      )}

      {/* Type filter */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {['image','video','audio','document','code'].map(t => (
          <button key={t} style={pillStyle(typeFilter === t)} onClick={() => { setTypeFilter(typeFilter === t ? null : t); setPage(1) }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>{TYPE_ICONS[t]} {t}</span>
          </button>
        ))}
      </div>

      {/* Results grid */}
      {loading ? (
        <div style={{ color: '#505068', fontSize: 13 }}>Loading…</div>
      ) : chunks.length === 0 ? (
        <div style={{ color: '#383850', fontSize: 13 }}>No items for this period.</div>
      ) : (
        <>
          <div style={{ fontSize: 11, color: '#505068', marginBottom: 12 }}>{total} items</div>
          <ResultGrid results={chunks} onSelect={onSelect} selected={selected} />
          {total > 40 && (
            <div style={{ display: 'flex', gap: 8, marginTop: 16, alignItems: 'center' }}>
              <button disabled={page === 1} onClick={() => setPage(p => p - 1)}
                style={{ ...pillStyle(false), opacity: page === 1 ? 0.3 : 1 }}>← Prev</button>
              <span style={{ fontSize: 11, color: '#505068' }}>Page {page} of {Math.ceil(total / 40)}</span>
              <button disabled={page >= Math.ceil(total / 40)} onClick={() => setPage(p => p + 1)}
                style={{ ...pillStyle(false), opacity: page >= Math.ceil(total / 40) ? 0.3 : 1 }}>Next →</button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

/* ── People Panel ────────────────────────────────────────────────────────────── */
function PeoplePanel({ api }: { api: string }) {
  const [identities, setIdentities] = useState<any[]>([])
  const [selected,   setSelected]   = useState<any | null>(null)
  const [photos,     setPhotos]     = useState<any[]>([])
  const [loading,    setLoading]    = useState(false)
  const [editing,    setEditing]    = useState<string | null>(null)
  const [editName,   setEditName]   = useState('')

  useEffect(() => {
    fetch(`${api}/identity/clusters`).then(r => r.json()).then(setIdentities).catch(() => {})
  }, [])

  function selectPerson(identity: any) {
    setSelected(identity)
    setLoading(true)
    fetch(`${api}/identity/photos/${identity.cluster_id}`).then(r => r.json()).then(d => {
      setPhotos(d.photos || [])
    }).catch(() => {}).finally(() => setLoading(false))
  }

  async function saveName(clusterId: string, name: string) {
    await fetch(`${api}/identity/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cluster_id: clusterId, label: name }),
    })
    setIdentities(ids => ids.map(i => i.cluster_id === clusterId ? { ...i, label: name } : i))
    setEditing(null)
  }

  const unlabelled = identities.filter(i => !i.label)
  const labelled   = identities.filter(i => i.label)

  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>People</h2>
        <p style={{ fontSize: 13, color: '#505068' }}>
          {identities.length} {identities.length === 1 ? 'person' : 'people'} detected in your photos.
          {unlabelled.length > 0 && <span style={{ color: '#fbbf24' }}> {unlabelled.length} unnamed.</span>}
        </p>
      </div>

      <div style={{ display: 'flex', gap: 20 }}>
        {/* Identity list */}
        <div style={{ width: 200, flexShrink: 0 }}>
          {labelled.length > 0 && (
            <>
              <div style={{ fontSize: 10, color: '#505068', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>Named</div>
              {labelled.map(i => (
                <button key={i.cluster_id} onClick={() => selectPerson(i)} style={{
                  display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                  padding: '8px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
                  background: selected?.cluster_id === i.cluster_id ? 'rgba(124,106,247,0.1)' : 'transparent',
                  color: selected?.cluster_id === i.cluster_id ? '#a78bfa' : '#e8e8f0',
                  fontSize: 13, fontFamily: 'inherit', textAlign: 'left', marginBottom: 2,
                }}>
                  <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(124,106,247,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 600, color: '#a78bfa', flexShrink: 0 }}>
                    {i.label[0].toUpperCase()}
                  </div>
                  <div>
                    <div style={{ fontSize: 12 }}>{i.label}</div>
                    <div style={{ fontSize: 10, color: '#505068' }}>{i.photo_count} photos</div>
                  </div>
                </button>
              ))}
            </>
          )}

          {unlabelled.length > 0 && (
            <>
              <div style={{ fontSize: 10, color: '#505068', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '16px 0 8px' }}>Unnamed</div>
              {unlabelled.map((i, idx) => (
                <button key={i.cluster_id} onClick={() => selectPerson(i)} style={{
                  display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                  padding: '8px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
                  background: selected?.cluster_id === i.cluster_id ? 'rgba(124,106,247,0.1)' : 'transparent',
                  color: selected?.cluster_id === i.cluster_id ? '#a78bfa' : '#505068',
                  fontSize: 13, fontFamily: 'inherit', textAlign: 'left', marginBottom: 2,
                }}>
                  <div style={{ width: 28, height: 28, borderRadius: '50%', background: '#1a1a2e', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#383850', flexShrink: 0 }}>
                    ?
                  </div>
                  <div>
                    <div style={{ fontSize: 12 }}>Person {idx + 1}</div>
                    <div style={{ fontSize: 10, color: '#383850' }}>{i.photo_count} photos</div>
                  </div>
                </button>
              ))}
            </>
          )}

          {identities.length === 0 && (
            <div style={{ fontSize: 12, color: '#383850' }}>No faces detected yet. Ingest some photos first.</div>
          )}
        </div>

        {/* Photo grid + name editor */}
        {selected && (
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              {editing === selected.cluster_id ? (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    autoFocus
                    value={editName}
                    onChange={e => setEditName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') saveName(selected.cluster_id, editName); if (e.key === 'Escape') setEditing(null) }}
                    style={{ background: 'rgba(124,106,247,0.06)', border: '1px solid #7c6af7', borderRadius: 6, color: '#e8e8f0', fontSize: 14, padding: '6px 10px', fontFamily: 'inherit', outline: 'none' }}
                    placeholder="Enter name…"
                  />
                  <button onClick={() => saveName(selected.cluster_id, editName)} style={{ padding: '6px 12px', borderRadius: 6, background: 'rgba(124,106,247,0.15)', border: '1px solid rgba(124,106,247,0.3)', color: '#a78bfa', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit' }}>Save</button>
                  <button onClick={() => setEditing(null)} style={{ padding: '6px 12px', borderRadius: 6, background: 'transparent', border: '1px solid #1a1a2e', color: '#505068', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit' }}>Cancel</button>
                </div>
              ) : (
                <>
                  <span style={{ fontSize: 16, fontWeight: 600, color: '#e8e8f0' }}>{selected.label || 'Unnamed person'}</span>
                  <button onClick={() => { setEditing(selected.cluster_id); setEditName(selected.label || '') }}
                    style={{ padding: '4px 10px', borderRadius: 6, background: 'transparent', border: '1px solid #1a1a2e', color: '#505068', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' }}>
                    {selected.label ? 'Rename' : 'Name this person'}
                  </button>
                </>
              )}
            </div>

            {loading ? (
              <div style={{ color: '#505068', fontSize: 13 }}>Loading photos…</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: 6 }}>
                {photos.map((p: any) => (
                  <div key={p.chunk_id} style={{ aspectRatio: '1', borderRadius: 8, overflow: 'hidden', background: '#0a0a0f', border: '1px solid #1a1a2e' }}>
                    {p.thumbnail_url
                      ? <img src={`${api}${p.thumbnail_url}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      : <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Image size={20} color="#252540" /></div>
                    }
                  </div>
                ))}
                {photos.length === 0 && <div style={{ fontSize: 12, color: '#383850' }}>No photos found.</div>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Settings Panel ─────────────────────────────────────────────────────────── */
function SettingsPanel({ api }: { api: string }) {
  const [cfg,     setCfg]     = useState<any>(null)
  const [stats,   setStats]   = useState<any>(null)
  const [copied,  setCopied]  = useState<string | null>(null)

  useEffect(() => {
    fetch(`${api}/setup/config`).then(r => r.json()).then(setCfg).catch(() => {})
    fetch(`${api}/stats`).then(r => r.json()).then(setStats).catch(() => {})
  }, [])

  function copy(key: string, value: string) {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(key)
      setTimeout(() => setCopied(null), 1800)
    })
  }

  const Section = ({ title, children }: { title: string, children: React.ReactNode }) => (
    <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '20px 24px', marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: '#505068', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>{title}</div>
      {children}
    </div>
  )

  const Row = ({ label, hint, children }: { label: string, hint?: string, children: React.ReactNode }) => (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, gap: 16 }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, color: '#e8e8f0' }}>{label}</div>
        {hint && <div style={{ fontSize: 11, color: '#505068', marginTop: 2 }}>{hint}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  )

  const StatusDot = ({ on, onColor = '#34d399', offColor = '#383850' }: { on: boolean, onColor?: string, offColor?: string }) => (
    <div style={{ width: 6, height: 6, borderRadius: '50%', background: on ? onColor : offColor, boxShadow: on ? `0 0 5px ${onColor}` : 'none' }} />
  )

  const EnvVar = ({ name, value, copyKey }: { name: string, value: string, copyKey: string }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, padding: '8px 10px', background: 'rgba(124,106,247,0.04)', borderRadius: 6, border: '1px solid #1a1a2e' }}>
      <code style={{ flex: 1, fontSize: 11, color: '#7c6af7', fontFamily: 'JetBrains Mono, monospace', wordBreak: 'break-all' }}>
        {name}={value}
      </code>
      <button
        onClick={() => copy(copyKey, `${name}=${value}`)}
        style={{ padding: '3px 8px', borderRadius: 4, background: 'transparent', border: '1px solid #252540', color: '#505068', fontSize: 10, cursor: 'pointer', fontFamily: 'inherit', whiteSpace: 'nowrap' }}
      >
        {copied === copyKey ? 'Copied!' : 'Copy'}
      </button>
    </div>
  )

  return (
    <div style={{ maxWidth: 680 }}>
      <div style={{ marginBottom: 28 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>Settings</h2>
        <p style={{ fontSize: 13, color: '#505068' }}>
          Settings are configured via environment variables in <code style={{ color: '#a78bfa', fontSize: 12 }}>.env</code>. Edit that file and restart the container to apply changes.
        </p>
      </div>

      {/* Index stats */}
      <Section title="Index">
        <Row label="Total memories" hint="Chunks currently indexed">
          <span style={{ fontSize: 14, fontWeight: 600, color: '#a78bfa', fontFamily: 'JetBrains Mono, monospace' }}>
            {stats?.total_chunks?.toLocaleString() ?? '—'}
          </span>
        </Row>
        {stats?.by_type && Object.keys(stats.by_type).length > 0 && (
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {Object.entries(stats.by_type as Record<string, number>).map(([type, count]) => (
              <div key={type} style={{ fontSize: 11, color: '#505068' }}>
                <span style={{ color: '#383850' }}>{count.toLocaleString()}</span> {type}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* LLM */}
      <Section title="LLM Provider">
        <Row label="Active provider" hint={`Set LLM_PROVIDER in .env`}>
          <span style={{ fontSize: 12, color: '#e8e8f0', textTransform: 'capitalize' }}>{cfg?.llm_provider ?? '—'}</span>
        </Row>
        <div style={{ fontSize: 11, color: '#505068', marginBottom: 6 }}>Copy to .env:</div>
        <EnvVar name="LLM_PROVIDER" value="anthropic" copyKey="llm_anthropic" />
        <EnvVar name="LLM_PROVIDER" value="openai"    copyKey="llm_openai" />
        <EnvVar name="LLM_PROVIDER" value="local"     copyKey="llm_local" />
        <EnvVar name="ANTHROPIC_MODEL" value="claude-sonnet-4-6" copyKey="model_sonnet" />
        <EnvVar name="ANTHROPIC_MODEL" value="claude-opus-4-6"   copyKey="model_opus" />
      </Section>

      {/* TTS */}
      <Section title="Voice (TTS)">
        <Row label="Kokoro voice (CPU)" hint={`Current: ${cfg?.tts_kokoro_voice ?? '—'}`}>
          <span style={{ fontSize: 12, color: '#e8e8f0', fontFamily: 'monospace' }}>{cfg?.tts_kokoro_voice ?? '—'}</span>
        </Row>
        <Row label="Qwen voice (GPU)" hint={`Current: ${cfg?.tts_qwen_voice ?? '—'}`}>
          <span style={{ fontSize: 12, color: '#e8e8f0', fontFamily: 'monospace' }}>{cfg?.tts_qwen_voice ?? '—'}</span>
        </Row>
        <div style={{ fontSize: 11, color: '#505068', marginBottom: 6 }}>Kokoro voice options:</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {['af_heart','af_bella','af_sarah','af_nicole','am_adam','am_michael','bm_george','bf_emma'].map(v => (
            <button key={v} onClick={() => copy(`kokoro_${v}`, `TTS_KOKORO_VOICE=${v}`)}
              style={{ padding: '3px 8px', borderRadius: 4, background: 'transparent', border: '1px solid #1a1a2e', color: copied === `kokoro_${v}` ? '#34d399' : '#505068', fontSize: 10, cursor: 'pointer', fontFamily: 'monospace' }}>
              {copied === `kokoro_${v}` ? '✓' : ''} {v}
            </button>
          ))}
        </div>
      </Section>

      {/* GPU */}
      <Section title="GPU & Hardware">
        <Row label="GPU acceleration" hint={cfg?.gpu_enabled ? 'CUDA active' : 'Set GPU_ENABLED=true in .env'}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <StatusDot on={cfg?.gpu_enabled} />
            <span style={{ fontSize: 12, color: cfg?.gpu_enabled ? '#34d399' : '#505068' }}>
              {cfg?.gpu_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </Row>
        {!cfg?.gpu_enabled && <EnvVar name="GPU_ENABLED" value="true" copyKey="gpu_enable" />}
      </Section>

      {/* Auth */}
      <Section title="API Authentication">
        <Row label="API key auth" hint={cfg?.auth_enabled ? 'X-API-Key required on all requests' : 'Set OMNEX_API_KEY to enable'}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <StatusDot on={cfg?.auth_enabled} onColor="#34d399" offColor="#fbbf24" />
            <span style={{ fontSize: 12, color: cfg?.auth_enabled ? '#34d399' : '#fbbf24' }}>
              {cfg?.auth_enabled ? 'Enabled' : 'Disabled (local only)'}
            </span>
          </div>
        </Row>
        {!cfg?.auth_enabled && <EnvVar name="OMNEX_API_KEY" value="your_secret_key_here" copyKey="api_key" />}
      </Section>
    </div>
  )
}

/* ── Expandable Results ─────────────────────────────────────────────────────── */
function ExpandableResults({ results, total, onSelect, selected }: {
  results: QueryResult[]
  total: number
  onSelect: (c: QueryResult) => void
  selected: QueryResult | null
}) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'transparent', border: 'none', cursor: 'pointer',
          padding: '4px 0', color: '#505068', fontSize: 11,
          letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'inherit',
        }}
      >
        <ChevronRight size={12} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
        {total} source{total !== 1 ? 's' : ''} retrieved
      </button>
      {open && (
        <div style={{ marginTop: 8 }}>
          <ResultGrid results={results} onSelect={onSelect} selected={selected} />
        </div>
      )}
    </div>
  )
}

/* ── Remote Access Panel ───────────────────────────────────────────────────── */
function RemoteAccessPanel({ api }: { api: string }) {
  const [tunnel, setTunnel]     = useState<any>(null)
  const [fuse,   setFuse]       = useState<any>(null)
  const [copied, setCopied]     = useState<string | null>(null)
  const [, setPolling]          = useState(true)

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>
    async function poll() {
      try {
        const res  = await fetch(`${api}/setup/tunnel`)
        const data = await res.json()
        setTunnel(data)
        fetch(`${api}/setup/fuse`).then(r => r.json()).then(setFuse).catch(() => {})
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
        headers: {
          "X-API-Key":  apiKey,
          "X-Agent-ID": "YOUR_AGENT_ID",
        },
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

      {/* FUSE virtual filesystem status */}
      <div style={{ background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e', borderRadius: 12, padding: '16px 24px', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: fuse?.mounted ? '#34d399' : '#383850', boxShadow: fuse?.mounted ? '0 0 6px #34d399' : 'none' }} />
          <span style={{ fontSize: 13, color: '#e8e8f0', fontWeight: 500 }}>
            FUSE Virtual Drive — {fuse == null ? 'Checking…' : fuse.mounted ? 'Mounted' : 'Not mounted'}
          </span>
        </div>
        {fuse && (
          <>
            <p style={{ fontSize: 12, color: '#505068', marginTop: 4, marginBottom: 8, fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {fuse.mount_path}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {fuse.dirs?.map((d: string) => (
                <span key={d} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(124,106,247,0.08)', border: '1px solid rgba(124,106,247,0.12)', color: '#7c6af7', fontFamily: 'monospace' }}>
                  /{d}
                </span>
              ))}
            </div>
          </>
        )}
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
            <p style={{ fontSize: 11, color: '#383850', marginTop: 10, lineHeight: 1.6 }}>
              Paste into your Claude Desktop / Cursor mcp.json config.<br />
              Replace <code style={{ color: '#a78bfa' }}>YOUR_AGENT_ID</code> with the ID returned from{' '}
              <code style={{ color: '#a78bfa' }}>POST {url}/agents</code>.<br />
              Tools available: <code style={{ color: '#505068' }}>omnex_search</code> · <code style={{ color: '#505068' }}>omnex_remember</code> · <code style={{ color: '#505068' }}>omnex_stats</code>
            </p>
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

/* ── Brain Orb — TTS indicator ─────────────────────────────────────────────── */
function BrainOrb({ enabled, speaking, onClick }: { enabled: boolean, speaking: boolean, onClick: () => void }) {
  return (
    <motion.button
      onClick={onClick}
      title={enabled ? (speaking ? 'Speaking…' : 'Voice output on — click to mute') : 'Voice output off — click to enable'}
      whileTap={{ scale: 0.88 }}
      style={{
        width: 34, height: 34, borderRadius: 12, flexShrink: 0,
        background: 'transparent', border: '1px solid transparent',
        cursor: 'pointer', position: 'relative',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 0,
      }}
    >
      {/* Outer pulse rings — only when speaking */}
      {speaking && (
        <>
          <motion.div
            animate={{ scale: [1, 2.0], opacity: [0.4, 0] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
            style={{
              position: 'absolute', inset: 4, borderRadius: '50%',
              border: '1px solid rgba(124,106,247,0.6)', pointerEvents: 'none',
            }}
          />
          <motion.div
            animate={{ scale: [1, 1.6], opacity: [0.3, 0] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut', delay: 0.3 }}
            style={{
              position: 'absolute', inset: 4, borderRadius: '50%',
              border: '1px solid rgba(167,139,250,0.5)', pointerEvents: 'none',
            }}
          />
        </>
      )}

      {/* Core orb */}
      <motion.div
        animate={speaking
          ? { boxShadow: [
              '0 0 8px rgba(124,106,247,0.4), 0 0 16px rgba(124,106,247,0.2)',
              '0 0 20px rgba(167,139,250,0.8), 0 0 36px rgba(124,106,247,0.4)',
              '0 0 8px rgba(124,106,247,0.4), 0 0 16px rgba(124,106,247,0.2)',
            ] }
          : { boxShadow: enabled
              ? '0 0 6px rgba(124,106,247,0.25)'
              : 'none'
          }
        }
        transition={speaking ? { duration: 0.9, repeat: Infinity, ease: 'easeInOut' } : { duration: 0.3 }}
        style={{
          width: 26, height: 26, borderRadius: '50%',
          background: enabled
            ? 'radial-gradient(circle at 35% 30%, rgba(192,168,255,0.35) 0%, rgba(124,106,247,0.18) 50%, rgba(5,5,7,0.85) 100%)'
            : 'radial-gradient(circle at 35% 30%, rgba(80,80,104,0.2) 0%, rgba(26,26,46,0.6) 100%)',
          border: `1px solid ${enabled ? 'rgba(124,106,247,0.45)' : 'rgba(37,37,64,0.6)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <Brain size={11} color={enabled ? '#c4b5fd' : '#383850'} strokeWidth={1.4} />
      </motion.div>
    </motion.button>
  )
}

/* ── Mic Orb — waveform visualiser + push-to-talk + always-listen toggle ───── */
function MicOrb({
  listening, micLevel, alwaysListen, onPress, onLongPress,
}: {
  listening: boolean
  micLevel: number
  alwaysListen: boolean
  onPress: () => void
  onLongPress: () => void
}) {
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const didLongPress = useRef(false)

  function handlePointerDown() {
    didLongPress.current = false
    holdTimer.current = setTimeout(() => {
      didLongPress.current = true
      onLongPress()
    }, 600)
  }

  function handlePointerUp() {
    if (holdTimer.current) clearTimeout(holdTimer.current)
    if (!didLongPress.current) onPress()
  }

  const bars = 5
  const color = alwaysListen ? '#a78bfa' : listening ? '#f87171' : '#383850'
  const activeColor = alwaysListen ? '#a78bfa' : '#f87171'

  return (
    <motion.button
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={() => { if (holdTimer.current) clearTimeout(holdTimer.current) }}
      title={alwaysListen ? 'Always-listening on (hold to toggle)' : listening ? 'Recording… click to stop' : 'Click to speak · Hold to toggle always-listen'}
      whileTap={{ scale: 0.88 }}
      style={{
        width: 34, height: 34, borderRadius: 12, flexShrink: 0,
        background: listening
          ? 'rgba(248,113,113,0.08)'
          : alwaysListen ? 'rgba(124,106,247,0.08)' : 'transparent',
        border: listening
          ? '1px solid rgba(248,113,113,0.25)'
          : alwaysListen ? '1px solid rgba(124,106,247,0.25)' : '1px solid transparent',
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: 2, padding: 0, position: 'relative',
      }}
    >
      {/* Always-listen pulse ring */}
      {alwaysListen && !listening && (
        <motion.div
          animate={{ scale: [1, 1.5], opacity: [0.3, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }}
          style={{
            position: 'absolute', inset: 2, borderRadius: 10,
            border: '1px solid rgba(124,106,247,0.4)', pointerEvents: 'none',
          }}
        />
      )}

      {/* Waveform bars — animated by micLevel when active */}
      {(listening || alwaysListen) ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 2, height: 20 }}>
          {Array.from({ length: bars }).map((_, i) => {
            const phase = i / (bars - 1)
            // Centre bar tallest, edges shortest
            const base = 0.25 + 0.5 * Math.sin(Math.PI * phase)
            const height = listening
              ? Math.max(3, (base + micLevel * (1 - base)) * 18)
              : 3 + base * 6  // idle when always-listen but not currently recording
            return (
              <motion.div
                key={i}
                animate={{ height }}
                transition={{ duration: 0.08, ease: 'easeOut' }}
                style={{
                  width: 2.5, borderRadius: 2,
                  background: activeColor,
                  opacity: listening ? 0.9 : 0.4,
                }}
              />
            )
          })}
        </div>
      ) : (
        <Mic size={15} color={color} />
      )}
    </motion.button>
  )
}

/* ── Empty Index State — shown when no data has been ingested yet ───────────── */
function EmptyIndexState({ onIngest }: { onIngest: () => void }) {
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: '0 40px', minHeight: '60vh', textAlign: 'center',
    }}>
      {/* Icon */}
      <motion.div
        animate={{ scale: [1, 1.04, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          width: 64, height: 64, borderRadius: 20, marginBottom: 28,
          background: 'radial-gradient(circle at 35% 30%, rgba(124,106,247,0.3) 0%, rgba(5,5,7,0.9) 100%)',
          border: '1px solid rgba(124,106,247,0.25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 0 32px rgba(124,106,247,0.12)',
        }}
      >
        <Database size={28} color="#7c6af7" strokeWidth={1.4} />
      </motion.div>

      <h2 style={{ fontSize: 22, fontWeight: 600, color: '#e8e8f0', marginBottom: 10, letterSpacing: '-0.01em' }}>
        Your memory is empty
      </h2>
      <p style={{ fontSize: 14, color: '#505068', lineHeight: 1.7, maxWidth: 420, marginBottom: 32 }}>
        Omnex has nothing to recall yet. Drop in a folder of documents, photos, audio, video, or code and let it index everything automatically.
      </p>

      {/* What you can index */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 32, flexWrap: 'wrap', justifyContent: 'center' }}>
        {[
          { icon: <FileText size={12} />, label: 'Documents & PDFs', color: '#fbbf24' },
          { icon: <Image size={12} />,    label: 'Photos',           color: '#60a5fa' },
          { icon: <Video size={12} />,    label: 'Videos',           color: '#a78bfa' },
          { icon: <Music size={12} />,    label: 'Audio',            color: '#34d399' },
          { icon: <Code2 size={12} />,    label: 'Code',             color: '#f87171' },
        ].map(({ icon, label, color }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 20,
            border: '1px solid #1a1a2e',
            color: '#383850', fontSize: 12,
          }}>
            <span style={{ color }}>{icon}</span>
            {label}
          </div>
        ))}
      </div>

      <motion.button
        whileHover={{ scale: 1.03, boxShadow: '0 0 28px rgba(124,106,247,0.45)' }}
        whileTap={{ scale: 0.97 }}
        onClick={onIngest}
        style={{
          padding: '11px 28px', borderRadius: 14, cursor: 'pointer',
          background: 'linear-gradient(145deg, #8b7cf8, #6b5ce7)',
          border: '1px solid rgba(124,106,247,0.4)',
          color: 'white', fontSize: 14, fontWeight: 500,
          fontFamily: 'inherit', letterSpacing: '0.01em',
          boxShadow: '0 0 20px rgba(124,106,247,0.3)',
        }}
      >
        Open Ingest →
      </motion.button>

      <p style={{ fontSize: 11, color: '#252540', marginTop: 14 }}>
        First ingestion loads models (~30s). Subsequent runs are fast.
      </p>
    </div>
  )
}

/* ── Ready State — shown when index has data but no queries yet ─────────────── */
function ReadyState({ stats, onQuery }: { stats: IndexStats | null, onQuery: (q: string) => void }) {
  const total = stats?.total ?? 0

  const suggestions = [
    'What did I work on recently?',
    'Show me photos from this year',
    'Find documents about contracts',
    'Who appears most in my photos?',
    'What audio files do I have?',
    'Show me code I wrote for authentication',
  ]

  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: '0 40px', minHeight: '60vh', textAlign: 'center',
    }}>
      {/* Brain orb */}
      <motion.div
        animate={{
          boxShadow: [
            '0 0 20px rgba(124,106,247,0.15), 0 0 60px rgba(124,106,247,0.06)',
            '0 0 36px rgba(167,139,250,0.3), 0 0 80px rgba(124,106,247,0.12)',
            '0 0 20px rgba(124,106,247,0.15), 0 0 60px rgba(124,106,247,0.06)',
          ]
        }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          width: 72, height: 72, borderRadius: '50%', marginBottom: 28,
          background: 'radial-gradient(circle at 35% 30%, rgba(192,168,255,0.25) 0%, rgba(124,106,247,0.1) 50%, rgba(5,5,7,0.9) 100%)',
          border: '1px solid rgba(124,106,247,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <Brain size={32} color="#a78bfa" strokeWidth={1.2} />
      </motion.div>

      <h2 style={{ fontSize: 20, fontWeight: 600, color: '#e8e8f0', marginBottom: 8, letterSpacing: '-0.01em' }}>
        {total.toLocaleString()} memories indexed
      </h2>

      {/* Per-type breakdown */}
      {stats && (
        <div style={{ display: 'flex', gap: 14, marginBottom: 32, flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            { count: stats.images,    label: 'photos',    color: '#60a5fa' },
            { count: stats.documents, label: 'docs',      color: '#fbbf24' },
            { count: stats.videos,    label: 'videos',    color: '#a78bfa' },
            { count: stats.audio,     label: 'audio',     color: '#34d399' },
            { count: stats.code,      label: 'code',      color: '#f87171' },
          ].filter(t => t.count > 0).map(({ count, label, color }) => (
            <div key={label} style={{ fontSize: 12, color: '#505068' }}>
              <span style={{ color, fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>
                {count.toLocaleString()}
              </span>{' '}{label}
            </div>
          ))}
        </div>
      )}

      <p style={{ fontSize: 13, color: '#383850', marginBottom: 20 }}>Try asking:</p>

      {/* Suggestion chips */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', maxWidth: 560 }}>
        {suggestions.map((s) => (
          <motion.button
            key={s}
            whileHover={{ borderColor: 'rgba(124,106,247,0.4)', color: '#a78bfa' }}
            whileTap={{ scale: 0.96 }}
            onClick={() => onQuery(s)}
            style={{
              padding: '7px 14px', borderRadius: 20, cursor: 'pointer',
              border: '1px solid #1a1a2e',
              background: 'transparent',
              color: '#505068', fontSize: 12,
              fontFamily: 'inherit',
              transition: 'all 0.15s',
            }}
          >
            {s}
          </motion.button>
        ))}
      </div>
    </div>
  )
}

/* ── Input bar ─────────────────────────────────────────────────────────────── */
function InputBar({
  value, onChange, onSubmit, onKey, onFocus, onBlur,
  focused, loading, listening, voiceSupport, onVoice, inputRef,
  ttsSupport, ttsEnabled, onTtsToggle,
  isSpeaking, micLevel, alwaysListen, onAlwaysListen,
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
  isSpeaking?: boolean
  micLevel?: number
  alwaysListen?: boolean
  onAlwaysListen?: () => void
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
          placeholder={listening ? 'Listening…' : alwaysListen ? 'Always listening — say something…' : 'Ask anything about your data…'}
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
            <MicOrb
              listening={listening}
              micLevel={micLevel ?? 0}
              alwaysListen={alwaysListen ?? false}
              onPress={onVoice}
              onLongPress={onAlwaysListen ?? (() => {})}
            />
          )}

          {ttsSupport && (
            <BrainOrb
              enabled={ttsEnabled ?? false}
              speaking={isSpeaking ?? false}
              onClick={onTtsToggle ?? (() => {})}
            />
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
          Enter to send · Shift+Enter for new line{voiceSupport ? ' · Click mic to speak · Hold mic to always-listen' : ''}
        </span>
      </div>
    </motion.div>
  )
}
