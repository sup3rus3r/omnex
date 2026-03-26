/**
 * Omnex — Streaming LLM route
 * POST /api/chat
 *
 * Receives { query, context } from the UI after the vector search completes.
 * Streams the LLM response back using Vercel AI SDK.
 *
 * Provider is selected via NEXT_PUBLIC_LLM_PROVIDER env var:
 *   anthropic — claude-sonnet-4-6 (default)
 *   openai    — gpt-4o-mini
 *   local     — Ollama via openai-compatible endpoint
 */

import { streamText } from 'ai'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createOpenAI } from '@ai-sdk/openai'

export async function POST(req: Request) {
  const { query, context } = await req.json()

  const provider  = process.env.LLM_PROVIDER || 'anthropic'
  const prompt    = buildPrompt(query, context)

  if (provider === 'anthropic') {
    const anthropic = createAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    })
    const result = streamText({
      model:  anthropic(process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6'),
      prompt,
      maxOutputTokens: 1024,
    })
    return result.toTextStreamResponse()
  }

  if (provider === 'openai') {
    const openai = createOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
    const result = streamText({
      model:  openai(process.env.OPENAI_MODEL || 'gpt-4o-mini'),
      prompt,
      maxOutputTokens: 1024,
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
    model:  openai(process.env.LOCAL_LLM_MODEL || 'phi3:mini'),
    prompt,
    maxOutputTokens: 1024,
  })
  return result.toTextStreamResponse()
}

function buildPrompt(query: string, context: string): string {
  return `You are Omnex, an AI memory assistant. The user asked: "${query}"

Here are the most relevant items from their personal data:

${context}

Provide a concise, helpful response. Reference specific items by number. If you suggest the user look at something specific, say so clearly.`
}
