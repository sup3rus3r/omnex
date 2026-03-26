'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, CheckCircle2, AlertCircle, Loader2, Zap } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

interface ModelStatus {
  id:       string
  name:     string
  size_mb:  number
  status:   'pending' | 'downloading' | 'ready' | 'error'
  progress: number
  error:    string
}

interface Props {
  onComplete: () => void
}

// Generic display names so we don't expose model internals
const DISPLAY_NAMES: Record<string, string> = {
  minilm:   'Language Model',
  clip:     'Vision Model',
  codebert: 'Code Model',
  whisper:  'Audio Model',
  deepface: 'Identity Model',
}

export default function SetupScreen({ onComplete }: Props) {
  const [models,   setModels]   = useState<ModelStatus[]>([])
  const [started,  setStarted]  = useState(false)
  const [done,     setDone]     = useState(false)
  const [error,    setError]    = useState<string | null>(null)
  const [totalMb,  setTotalMb]  = useState(0)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    // Load initial status
    fetch(`${API}/setup/status`)
      .then(r => r.json())
      .then(data => {
        setModels(data.models || [])
        const missing = (data.models || []).filter((m: ModelStatus) => m.status !== 'ready')
        setTotalMb(missing.reduce((acc: number, m: ModelStatus) => acc + m.size_mb, 0))
        if (data.ready) {
          setDone(true)
          setTimeout(onComplete, 800)
        }
      })
      .catch(() => setError('Cannot reach Omnex API. Make sure the backend is running.'))

    return () => eventSourceRef.current?.close()
  }, [])

  function startDownload() {
    setStarted(true)
    setError(null)

    // Use fetch with streaming for SSE since EventSource doesn't support POST
    fetch(`${API}/setup/download`, { method: 'POST' })
      .then(async res => {
        if (!res.ok) throw new Error('Download failed to start')
        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done: streamDone, value } = await reader.read()
          if (streamDone) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const evt = JSON.parse(line.slice(6))
              if (evt.type === 'complete') {
                setDone(true)
                setTimeout(onComplete, 1200)
                return
              }
              if (evt.type === 'error') {
                setError(evt.message || 'Download failed')
                setStarted(false)
                return
              }
              // Progress update for a model
              setModels(prev => prev.map(m =>
                m.id === evt.id
                  ? { ...m, status: evt.status, progress: evt.progress }
                  : m
              ))
            } catch (_) {}
          }
        }
      })
      .catch(e => {
        setError(e.message || 'Connection failed')
        setStarted(false)
      })
  }

  const allReady = models.length > 0 && models.every(m => m.status === 'ready')
  const downloading = models.find(m => m.status === 'downloading')
  const overallProgress = models.length === 0 ? 0
    : models.reduce((acc, m) => acc + (m.status === 'ready' ? 100 : m.progress), 0) / models.length

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: '#050507',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
    }}>
      {/* Background grid */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(rgba(124,106,247,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(124,106,247,0.025) 1px, transparent 1px)',
        backgroundSize: '60px 60px',
        maskImage: 'radial-gradient(ellipse 70% 60% at 50% 50%, black, transparent)',
      }} />
      {/* Ambient glow */}
      <div style={{
        position: 'absolute', top: '20%', left: '50%', transform: 'translateX(-50%)',
        width: 600, height: 300, pointerEvents: 'none',
        background: 'radial-gradient(ellipse, rgba(124,106,247,0.07) 0%, transparent 70%)',
        filter: 'blur(40px)',
      }} />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        style={{
          width: '100%', maxWidth: 520, padding: '0 24px',
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          position: 'relative', zIndex: 1,
        }}
      >
        {/* Logo */}
        <motion.div
          animate={{ boxShadow: done
            ? ['0 0 40px rgba(52,211,153,0.4)', '0 0 70px rgba(52,211,153,0.6)', '0 0 40px rgba(52,211,153,0.4)']
            : ['0 0 30px rgba(124,106,247,0.2)', '0 0 55px rgba(124,106,247,0.45)', '0 0 30px rgba(124,106,247,0.2)']
          }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            width: 72, height: 72, borderRadius: '50%', marginBottom: 28,
            background: done
              ? 'radial-gradient(circle at 35% 30%, rgba(52,211,153,0.25) 0%, rgba(16,185,129,0.08) 60%, transparent 100%)'
              : 'radial-gradient(circle at 35% 30%, rgba(192,168,255,0.25) 0%, rgba(124,106,247,0.08) 60%, transparent 100%)',
            border: done ? '1px solid rgba(52,211,153,0.35)' : '1px solid rgba(124,106,247,0.35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.6s',
          }}
        >
          {done
            ? <CheckCircle2 size={28} color="#34d399" strokeWidth={1.5} />
            : <Brain size={28} color="#c4b5fd" strokeWidth={1.5} />
          }
        </motion.div>

        {/* Title */}
        <div style={{ fontSize: 10, letterSpacing: '0.4em', textTransform: 'uppercase', color: 'rgba(124,106,247,0.4)', marginBottom: 16 }}>
          Omnex · First Run Setup
        </div>
        <h1 style={{ fontSize: '1.8rem', fontWeight: 800, letterSpacing: '-0.05em', color: '#e8e8f4', textAlign: 'center', marginBottom: 10, lineHeight: 1.1 }}>
          {done ? 'Ready.' : 'Loading intelligence.'}
        </h1>
        <p style={{ fontSize: 13, color: '#38384e', textAlign: 'center', lineHeight: 1.7, maxWidth: 380, marginBottom: 36 }}>
          {done
            ? 'All models are loaded. Omnex is ready to index your data.'
            : `Omnex needs to download ${models.length} AI models (~${totalMb}MB) before first use. This happens once.`
          }
        </p>

        {/* Model list */}
        {models.length > 0 && (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 28 }}>
            {models.map((m, i) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.06, duration: 0.4 }}
                style={{
                  padding: '12px 16px', borderRadius: 12,
                  background: 'rgba(10,10,15,0.8)',
                  border: `1px solid ${
                    m.status === 'ready'       ? 'rgba(52,211,153,0.2)' :
                    m.status === 'downloading' ? 'rgba(124,106,247,0.3)' :
                    m.status === 'error'       ? 'rgba(248,113,113,0.2)' :
                    '#1a1a2e'
                  }`,
                  transition: 'border-color 0.3s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: m.status === 'downloading' ? 8 : 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <StatusIcon status={m.status} />
                    <span style={{ fontSize: 13, color: m.status === 'ready' ? '#a0f0d0' : '#a0a0b8', fontWeight: 500 }}>
                      {DISPLAY_NAMES[m.id] || m.name}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 11, color: '#383850', fontFamily: 'JetBrains Mono, monospace' }}>
                      {m.size_mb}MB
                    </span>
                    <span style={{
                      fontSize: 10, padding: '2px 7px', borderRadius: 4,
                      letterSpacing: '0.08em', textTransform: 'uppercase',
                      background: m.status === 'ready'       ? 'rgba(52,211,153,0.1)'  :
                                  m.status === 'downloading' ? 'rgba(124,106,247,0.1)' :
                                  m.status === 'error'       ? 'rgba(248,113,113,0.1)' :
                                  'rgba(37,37,64,0.5)',
                      color:      m.status === 'ready'       ? '#34d399' :
                                  m.status === 'downloading' ? '#a78bfa' :
                                  m.status === 'error'       ? '#f87171' :
                                  '#383850',
                      border: `1px solid ${
                                  m.status === 'ready'       ? 'rgba(52,211,153,0.15)'  :
                                  m.status === 'downloading' ? 'rgba(124,106,247,0.2)' :
                                  m.status === 'error'       ? 'rgba(248,113,113,0.15)' :
                                  '#1a1a2e'
                      }`,
                    }}>
                      {m.status === 'pending' ? 'waiting' : m.status === 'downloading' ? `${m.progress.toFixed(0)}%` : m.status}
                    </span>
                  </div>
                </div>
                {m.status === 'downloading' && (
                  <div style={{ height: 2, background: '#1a1a2e', borderRadius: 1, overflow: 'hidden' }}>
                    <motion.div
                      animate={{ width: `${m.progress}%` }}
                      transition={{ duration: 0.3, ease: 'easeOut' }}
                      style={{ height: '100%', background: 'linear-gradient(90deg, #7c6af7, #a78bfa)', borderRadius: 1 }}
                    />
                  </div>
                )}
                {m.status === 'error' && m.error && (
                  <div style={{ fontSize: 11, color: '#f87171', marginTop: 4 }}>{m.error}</div>
                )}
              </motion.div>
            ))}
          </div>
        )}

        {/* Overall progress bar when downloading */}
        {started && !done && (
          <div style={{ width: '100%', marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: '#505068' }}>
                {downloading ? `Loading ${DISPLAY_NAMES[downloading.id] || downloading.name}…` : 'Preparing…'}
              </span>
              <span style={{ fontSize: 11, color: '#7c6af7', fontFamily: 'JetBrains Mono, monospace' }}>
                {overallProgress.toFixed(0)}%
              </span>
            </div>
            <div style={{ height: 3, background: '#1a1a2e', borderRadius: 2, overflow: 'hidden' }}>
              <motion.div
                animate={{ width: `${overallProgress}%` }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
                style={{ height: '100%', background: 'linear-gradient(90deg, #7c6af7, #60a5fa)', borderRadius: 2, position: 'relative' }}
              >
                <div style={{
                  position: 'absolute', inset: 0,
                  background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
                  animation: 'dataStream 1.5s ease-in-out infinite',
                }} />
              </motion.div>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
            color: '#f87171', fontSize: 13, marginBottom: 16,
            padding: '10px 14px', background: 'rgba(248,113,113,0.05)',
            border: '1px solid rgba(248,113,113,0.15)', borderRadius: 10,
          }}>
            <AlertCircle size={14} /><span>{error}</span>
          </div>
        )}

        {/* CTA */}
        {!started && !done && (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={startDownload}
            style={{
              padding: '14px 36px', borderRadius: 14,
              background: 'linear-gradient(135deg, #7c6af7, #6b5ce7)',
              border: 'none', cursor: 'pointer',
              color: 'white', fontSize: 15, fontWeight: 600,
              fontFamily: 'inherit', letterSpacing: '-0.02em',
              display: 'flex', alignItems: 'center', gap: 8,
              boxShadow: '0 0 30px rgba(124,106,247,0.4), inset 0 1px 0 rgba(255,255,255,0.15)',
            }}
          >
            <Zap size={16} />
            Load AI Models
          </motion.button>
        )}

        {done && (
          <motion.button
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={onComplete}
            style={{
              padding: '14px 36px', borderRadius: 14,
              background: 'linear-gradient(135deg, #10b981, #059669)',
              border: 'none', cursor: 'pointer',
              color: 'white', fontSize: 15, fontWeight: 600,
              fontFamily: 'inherit', letterSpacing: '-0.02em',
              display: 'flex', alignItems: 'center', gap: 8,
              boxShadow: '0 0 30px rgba(52,211,153,0.4)',
            }}
          >
            <CheckCircle2 size={16} />
            Enter Omnex
          </motion.button>
        )}

        <p style={{ fontSize: 11, color: '#252540', marginTop: 20, textAlign: 'center' }}>
          All models run locally on your hardware. Nothing leaves your machine.
        </p>
      </motion.div>
    </div>
  )
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'ready')       return <CheckCircle2 size={14} color="#34d399" />
  if (status === 'downloading') return <Loader2 size={14} color="#a78bfa" style={{ animation: 'spin 1s linear infinite' }} />
  if (status === 'error')       return <AlertCircle size={14} color="#f87171" />
  return <div style={{ width: 14, height: 14, borderRadius: '50%', border: '1px solid #383850' }} />
}
