'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUp, Mic, MicOff, Loader2 } from 'lucide-react'

interface Props {
  onQuery: (query: string) => void
  loading: boolean
}

export default function QueryBar({ onQuery, loading }: Props) {
  const [input,        setInput]        = useState('')
  const [listening,    setListening]    = useState(false)
  const [voiceSupport, setVoiceSupport] = useState(false)
  const [focused,      setFocused]      = useState(false)
  const inputRef  = useRef<HTMLTextAreaElement>(null)
  const recognRef = useRef<any>(null)

  useEffect(() => {
    const supported =
      typeof window !== 'undefined' &&
      ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)
    setVoiceSupport(supported)
  }, [])

  function submit() {
    const q = input.trim()
    if (!q || loading) return
    onQuery(q)
    setInput('')
    if (inputRef.current) inputRef.current.style.height = 'auto'
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
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
      if (q) onQuery(q)
    }
    recog.onerror = () => setListening(false)
    recognRef.current = recog
    recog.start()
    setListening(true)
  }

  const ringColor = listening
    ? 'rgba(239,68,68,0.6)'
    : focused
    ? 'rgba(124,106,247,0.5)'
    : 'rgba(38,38,38,1)'

  return (
    <div className="relative w-full">
      <motion.div
        animate={{ boxShadow: `0 0 0 1.5px ${ringColor}, 0 8px 32px rgba(0,0,0,0.4)` }}
        transition={{ duration: 0.15 }}
        style={{ borderRadius: 16, background: '#111111' }}
        className="flex items-end gap-2 px-4 py-3"
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={listening ? 'Listening…' : 'Ask anything about your data…'}
          rows={1}
          style={{
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'none',
            color: '#f0f0f0',
            fontSize: '0.9375rem',
            lineHeight: '1.6',
            width: '100%',
            maxHeight: 160,
            overflowY: 'auto',
            fontFamily: 'inherit',
          }}
          onInput={(e) => {
            const el = e.currentTarget
            el.style.height = 'auto'
            el.style.height = `${el.scrollHeight}px`
          }}
        />

        <div className="flex items-center gap-1.5 flex-shrink-0 pb-0.5">
          {voiceSupport && (
            <motion.button
              whileTap={{ scale: 0.88 }}
              onClick={toggleVoice}
              style={{
                width: 32, height: 32, borderRadius: 10,
                background: listening ? 'rgba(239,68,68,0.15)' : 'transparent',
                border: 'none', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: listening ? '#f87171' : '#555',
              }}
            >
              {listening ? <MicOff size={14} /> : <Mic size={14} />}
            </motion.button>
          )}

          <motion.button
            whileTap={{ scale: 0.88 }}
            onClick={submit}
            disabled={!input.trim() || loading}
            style={{
              width: 32, height: 32, borderRadius: 10,
              background: (!input.trim() || loading) ? 'rgba(124,106,247,0.2)' : '#7c6af7',
              border: 'none', cursor: (!input.trim() || loading) ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'white', flexShrink: 0,
              transition: 'background 0.15s',
            }}
          >
            {loading
              ? <Loader2 size={13} className="animate-spin" />
              : <ArrowUp size={13} />
            }
          </motion.button>
        </div>
      </motion.div>

      <AnimatePresence>
        {listening && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center gap-2 mt-3"
          >
            <div className="flex gap-1 items-end h-5">
              {[0.4, 0.7, 1, 0.7, 0.4].map((h, i) => (
                <motion.div
                  key={i}
                  style={{ width: 2, borderRadius: 2, background: '#f87171' }}
                  animate={{ height: [`${h * 8}px`, `${h * 20}px`, `${h * 8}px`] }}
                  transition={{ duration: 0.5 + i * 0.05, repeat: Infinity, delay: i * 0.08 }}
                />
              ))}
            </div>
            <span style={{ fontSize: 11, color: '#f87171' }}>Listening</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
