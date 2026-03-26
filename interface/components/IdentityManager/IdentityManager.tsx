'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Check, Users, UserPlus } from 'lucide-react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

interface Identity {
  _id:        string
  cluster_id: string
  face_count: number
  label?:     string
}

interface Props {
  onClose: () => void
}

export default function IdentityManager({ onClose }: Props) {
  const [pending,  setPending]  = useState<Identity[]>([])
  const [labelled, setLabelled] = useState<Record<string, string>>({})
  const [saving,   setSaving]   = useState<Record<string, boolean>>({})
  const [loading,  setLoading]  = useState(true)

  useEffect(() => { fetchPending() }, [])

  async function fetchPending() {
    setLoading(true)
    try {
      const res  = await fetch(`${API}/identity/pending?limit=50`)
      const data = await res.json()
      setPending(data.pending || [])
    } catch { /* API unavailable */ }
    finally { setLoading(false) }
  }

  async function saveLabel(clusterId: string) {
    const label = labelled[clusterId]?.trim()
    if (!label) return
    setSaving((s) => ({ ...s, [clusterId]: true }))
    try {
      await fetch(`${API}/identity/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cluster_id: clusterId, label }),
      })
      setPending((p) => p.filter((i) => i.cluster_id !== clusterId))
    } finally {
      setSaving((s) => ({ ...s, [clusterId]: false }))
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.75)',
        backdropFilter: 'blur(12px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 50, padding: 16,
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 16 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        style={{
          background: '#0a0a0f',
          border: '1px solid #1a1a2e',
          borderRadius: 16,
          width: '100%', maxWidth: 480,
          maxHeight: '85vh',
          display: 'flex', flexDirection: 'column',
          boxShadow: '0 24px 64px rgba(0,0,0,0.8)',
        }}
      >
        {/* Header */}
        <div style={{
          flexShrink: 0, padding: '14px 16px',
          borderBottom: '1px solid #1a1a2e',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 26, height: 26, borderRadius: 8,
              background: 'rgba(96,165,250,0.1)',
              border: '1px solid rgba(96,165,250,0.2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Users size={12} color="#60a5fa" />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, color: '#e8e8f0' }}>People</div>
              <div style={{ fontSize: 10, color: '#505068' }}>
                {loading ? 'Loading…' : `${pending.length} cluster${pending.length !== 1 ? 's' : ''} unlabelled`}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              width: 26, height: 26, border: 'none', background: 'transparent',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#383850', borderRadius: 6, transition: 'color 0.12s',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
          >
            <X size={13} />
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 14px' }}>
          {loading && (
            <div style={{ textAlign: 'center', padding: '40px 0', color: '#383850', fontSize: 13 }}>
              Scanning face clusters…
            </div>
          )}

          {!loading && pending.length === 0 && (
            <div style={{ textAlign: 'center', padding: '48px 0' }}>
              <UserPlus size={32} color="#1a1a2e" strokeWidth={1} style={{ marginBottom: 12 }} />
              <p style={{ fontSize: 13, color: '#383850', marginBottom: 4 }}>All faces have been named.</p>
              <p style={{ fontSize: 11, color: '#252540' }}>New people are auto-classified as photos are indexed.</p>
            </div>
          )}

          <AnimatePresence>
            {pending.map((identity) => (
              <motion.div
                key={identity.cluster_id}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -12, transition: { duration: 0.18 } }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 12px', borderRadius: 10, marginBottom: 6,
                  background: 'rgba(5,5,7,0.8)',
                  border: '1px solid #1a1a2e',
                }}
              >
                {/* Face crops */}
                <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
                  {[0, 1, 2].map((i) => (
                    <div key={i} style={{
                      width: 36, height: 36, borderRadius: '50%',
                      overflow: 'hidden', border: '1px solid #1a1a2e',
                      background: '#0f0f18', flexShrink: 0,
                    }}>
                      <img
                        src={`${API}/chunk/${identity._id}_face${i}/thumbnail`}
                        alt=""
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                      />
                    </div>
                  ))}
                </div>

                {/* Label input */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 10, color: '#383850', marginBottom: 5 }}>
                    {identity.face_count} appearance{identity.face_count !== 1 ? 's' : ''}
                  </div>
                  <div style={{ display: 'flex', gap: 5 }}>
                    <input
                      type="text"
                      placeholder="Name this person…"
                      value={labelled[identity.cluster_id] || ''}
                      onChange={(e) => setLabelled((l) => ({ ...l, [identity.cluster_id]: e.target.value }))}
                      onKeyDown={(e) => e.key === 'Enter' && saveLabel(identity.cluster_id)}
                      style={{
                        flex: 1, padding: '6px 10px',
                        background: 'rgba(10,10,15,0.8)', border: '1px solid #1a1a2e',
                        borderRadius: 8, color: '#e8e8f0', fontSize: 12,
                        fontFamily: 'inherit', outline: 'none',
                        transition: 'border-color 0.12s',
                      }}
                      onFocus={(e) => { e.currentTarget.style.borderColor = 'rgba(124,106,247,0.4)' }}
                      onBlur={(e) => { e.currentTarget.style.borderColor = '#1a1a2e' }}
                    />
                    <motion.button
                      whileTap={{ scale: 0.92 }}
                      onClick={() => saveLabel(identity.cluster_id)}
                      disabled={!labelled[identity.cluster_id]?.trim() || saving[identity.cluster_id]}
                      style={{
                        width: 30, height: 30, borderRadius: 8, border: 'none',
                        background: (!labelled[identity.cluster_id]?.trim() || saving[identity.cluster_id])
                          ? 'rgba(124,106,247,0.1)'
                          : 'rgba(124,106,247,0.8)',
                        cursor: (!labelled[identity.cluster_id]?.trim() || saving[identity.cluster_id]) ? 'not-allowed' : 'pointer',
                        color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0, opacity: (!labelled[identity.cluster_id]?.trim() || saving[identity.cluster_id]) ? 0.4 : 1,
                        transition: 'all 0.15s',
                      }}
                    >
                      <Check size={12} />
                    </motion.button>
                    <button
                      onClick={() => setPending((p) => p.filter((i) => i.cluster_id !== identity.cluster_id))}
                      style={{
                        width: 30, height: 30, borderRadius: 8, border: 'none', background: 'transparent',
                        cursor: 'pointer', color: '#383850', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0, transition: 'color 0.12s',
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#a0a0b8' }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = '#383850' }}
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {pending.length > 0 && (
          <div style={{
            flexShrink: 0, padding: '10px 14px',
            borderTop: '1px solid #1a1a2e',
            textAlign: 'center', fontSize: 10, color: '#383850',
          }}>
            Named people become searchable immediately — "show me photos with Sarah"
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}
