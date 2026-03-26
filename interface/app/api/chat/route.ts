/**
 * Omnex — Streaming LLM route
 * POST /api/chat
 *
 * Receives { query, context, messages } from the UI.
 * `messages` is the full prior conversation history (role/content pairs).
 * Streams the LLM response back using Vercel AI SDK.
 *
 * Provider is selected via LLM_PROVIDER env var:
 *   anthropic — claude-sonnet-4-6 (default)
 *   openai    — gpt-4o-mini
 *   local     — Ollama via openai-compatible endpoint
 */

import { streamText } from 'ai'
import type { ModelMessage } from 'ai'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createOpenAI } from '@ai-sdk/openai'

const SYSTEM_PROMPT =
  'You are Omnex, an AI memory assistant. ' +
  'You help users recall and understand their personal data — documents, photos, code, audio, and video. ' +
  'When relevant search results are provided, reference them by number. ' +
  'Be concise and specific. Use conversation history to answer follow-up questions in context.'

export async function POST(req: Request) {
  try {
    const { query, context, messages: history = [] } = await req.json()

    const provider = process.env.LLM_PROVIDER || 'anthropic'

    // Build messages array: prior history + current user turn with context
    const userContent = context
      ? `User query: "${query}"\n\nRelevant items from personal data:\n\n${context}\n\nProvide a concise, helpful response. Reference items by number where relevant.`
      : query

    const messages: ModelMessage[] = [
      ...(history as ModelMessage[]),
      { role: 'user', content: userContent },
    ]

    if (provider === 'anthropic') {
      const anthropic = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
      const result = streamText({
        model:           anthropic(process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6'),
        system:          SYSTEM_PROMPT,
        messages,
        })
      return result.toTextStreamResponse()
    }

    if (provider === 'openai') {
      const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY })
      const result = streamText({
        model:           openai(process.env.OPENAI_MODEL || 'gpt-4o-mini'),
        system:          SYSTEM_PROMPT,
        messages,
        })
      return result.toTextStreamResponse()
    }

    // Local — Ollama via OpenAI-compatible API
    const openai = createOpenAI({
      baseURL: process.env.LOCAL_LLM_HOST
        ? `${process.env.LOCAL_LLM_HOST}/v1`
        : 'http://localhost:11434/v1',
      apiKey: 'ollama',
    })
    const result = streamText({
      model:           openai(process.env.LOCAL_LLM_MODEL || 'phi3:mini'),
      system:          SYSTEM_PROMPT,
      messages,
    })
    return result.toTextStreamResponse()
  } catch (err: any) {
    console.error('[chat route error]', err?.message || err)
    return new Response(
      JSON.stringify({ error: err?.message || 'LLM error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
