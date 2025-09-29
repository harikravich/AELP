import { NextRequest, NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { serializeBigQueryRows } from '../../../lib/bigquery-serializer'
import { DATASET_COOKIE, PROD_DATASET, SANDBOX_DATASET } from '../../../lib/dataset'
import { cookies } from 'next/headers'

export const dynamic = 'force-dynamic'

async function buildContext() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const mode = cookies().get(DATASET_COOKIE)?.value === 'prod' ? 'prod' : 'sandbox'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  const bq = new BigQuery({ projectId })

  const ctx: any = { projectId, dataset, mode }
  try {
    const [kpi] = await bq.query({ query: `
      SELECT SUM(cost) AS cost, SUM(conversions) AS conv, SUM(revenue) AS revenue,
             MAX(date) AS last_date
      FROM \`${projectId}.${dataset}.ads_kpi_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
    `})
    ctx.kpi = kpi?.[0] || null
  } catch {}
  try {
    const [rows] = await bq.query({ query: `
      SELECT date, cost, conversions, SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac
      FROM \`${projectId}.${dataset}.ads_kpi_daily\`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
      ORDER BY date DESC
    `})
    ctx.kpi_recent = rows || []
  } catch {}
  try {
    const [mmm] = await bq.query({ query: `
      SELECT timestamp, channel, proposed_daily_budget, expected_conversions, expected_cac
      FROM \`${projectId}.${dataset}.mmm_allocations\`
      ORDER BY timestamp DESC LIMIT 1
    `})
    ctx.mmm = mmm?.[0] || null
  } catch {}
  try {
    const [ga4] = await bq.query({ query: `
      SELECT default_channel_group, SUM(conversions) AS conv
      FROM \`${projectId}.${dataset}.ga4_daily\`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      GROUP BY default_channel_group
      ORDER BY conv DESC LIMIT 6
    `})
    ctx.ga4 = ga4 || []
  } catch {}
  return ctx
}

function contextToPrompt(ctx: any) {
  const lines: string[] = []
  lines.push(`Project: ${ctx.projectId}, Dataset: ${ctx.dataset} (mode=${ctx.mode})`)
  if (ctx.kpi) {
    const c = Number(ctx.kpi.cost||0).toFixed(2)
    const v = Number(ctx.kpi.conv||0).toFixed(0)
    const r = Number(ctx.kpi.revenue||0).toFixed(2)
    const cac = (Number(ctx.kpi.cost||0)/Math.max(1, Number(ctx.kpi.conv||0))).toFixed(2)
    lines.push(`KPI (last 28d): cost=$${c}, conv=${v}, revenue=$${r}, CAC=$${cac}, last_date=${ctx.kpi.last_date}`)
  }
  if (ctx.mmm) {
    lines.push(`MMM latest: channel=${ctx.mmm.channel}, proposed_daily_budget=$${Number(ctx.mmm.proposed_daily_budget||0).toFixed(0)}, expected_conversions=${Number(ctx.mmm.expected_conversions||0).toFixed(0)}, expected_cac=$${Number(ctx.mmm.expected_cac||0).toFixed(2)}`)
  }
  if (Array.isArray(ctx.ga4) && ctx.ga4.length) {
    const top = ctx.ga4.map((r:any)=> `${r.default_channel_group}:${Number(r.conv||0)}`).join(', ')
    lines.push(`Top GA4 channels (28d): ${top}`)
  }
  lines.push('Answer with precise numbers when present, clarify date ranges, and say “not enough data” if unknown.')
  return lines.join('\n')
}

async function callOpenAICompat({ messages }:{ messages: any[] }) {
  const apiKey = process.env.OPENAI_API_KEY
  const base = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1'
  const model = process.env.OPENAI_MODEL || 'gpt-4o-mini'
  const r = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
    body: JSON.stringify({ model, messages, temperature: 0.2 }),
  })
  if (!r.ok) {
    const txt = await r.text()
    throw new Error(`Chat backend error: ${r.status} ${txt}`)
  }
  const j = await r.json()
  const reply = j.choices?.[0]?.message?.content || ''
  return reply
}

async function planSqlFromLlm(question: string, ctx: any) {
  const schema = `
Tables (use <PROJECT>.<DATASET>.table):
- ads_kpi_daily(date DATE, cost FLOAT64, conversions INT64, revenue FLOAT64)
- ga4_daily(date DATE, device_category STRING, default_channel_group STRING, conversions INT64)
- mmm_allocations(timestamp TIMESTAMP, channel STRING, proposed_daily_budget FLOAT64, expected_conversions FLOAT64, expected_cac FLOAT64)
- mmm_curves(timestamp TIMESTAMP, channel STRING, spend_grid JSON, conv_grid JSON)
- ads_campaign_performance(date DATE, campaign_id STRING, impressions INT64, clicks INT64, cost_micros INT64, conversions INT64, conversion_value FLOAT64, impression_share FLOAT64)
Rules: only SELECT; add LIMIT 100 if missing; always qualify tables as \n\`<PROJECT>.<DATASET>.table\`.
`
  const prompt = [
    { role: 'system', content: `You translate a question into a safe BigQuery SQL that answers it exactly. ${schema}` },
    { role: 'user', content: question },
  ]
  const jsonHint = 'Return only JSON: {"sql":"..."}'
  const r = await callOpenAICompat({ messages: [...prompt, { role: 'user', content: jsonHint }] })
  try {
    const j = JSON.parse(r)
    if (typeof j?.sql === 'string') return j.sql
  } catch {}
  return null
}

function parseCommand(text: string): { cmd: string, args: string } | null {
  const t = (text || '').trim()
  if (!t.startsWith('/')) return null
  const sp = t.indexOf(' ')
  if (sp === -1) return { cmd: t.toLowerCase(), args: '' }
  return { cmd: t.slice(0, sp).toLowerCase(), args: t.slice(sp+1).trim() }
}

function helpText() {
  return [
    'Commands:',
    '  /help – show this',
    '  /mmm <channel> <daily_budget> – predict conversions & CAC from MMM curve',
    '  /creative ad_id=<id> [campaign_id=..] [customer_id=..] – fetch RSA assets',
    '  /run ga4_ingest | ads_ingest only=<CID> – kick off pipelines (if allowed)',
    '  /sql <SELECT ...> – safe, read‑only query (limited tables, LIMIT 100 max)'
  ].join('\n')
}

const ALLOWED_TABLES = new Set([
  'ads_kpi_daily','ga4_daily','mmm_curves','mmm_allocations','bandit_change_proposals','canary_changes','ads_campaign_performance','ltv_priors_daily','ops_alerts','ops_flow_runs','platform_skeletons'
])

function validateSql(sql: string): string | null {
  const s = sql.trim().replace(/;+$/,'')
  const upper = s.toUpperCase()
  if (!upper.startsWith('SELECT')) return 'Only SELECT queries are allowed.'
  if (/[;]|\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE)\b/i.test(upper)) return 'Disallowed keyword found.'
  // Must reference at least one allowed table
  const lower = s.toLowerCase()
  const ok = Array.from(ALLOWED_TABLES).some(t => lower.includes(`.${t}`))
  if (!ok) return 'Query must reference an allowed table.'
  return null
}

async function runSql(projectId: string, dataset: string, sql: string) {
  const bq = new BigQuery({ projectId })
  const limited = /\blimit\b/i.test(sql) ? sql : `${sql}\nLIMIT 100`
  const q = limited.replace(/<DATASET>/g, dataset).replace(/<PROJECT>/g, projectId)
  const [rows] = await bq.query({ query: q })
  return serializeBigQueryRows(rows || [])
}

async function mmmWhatIf(projectId: string, dataset: string, channel: string, budget: number) {
  const bq = new BigQuery({ projectId })
  const [rows] = await bq.query({ query: `
    SELECT spend_grid, conv_grid FROM \`${projectId}.${dataset}.mmm_curves\`
    WHERE channel=@ch ORDER BY timestamp DESC LIMIT 1`, params: { ch: channel } })
  if (!rows?.[0]) return { error: `No MMM curve for ${channel}` }
  const spend: number[] = JSON.parse(String(rows[0].spend_grid))
  const conv: number[] = JSON.parse(String(rows[0].conv_grid))
  let idx = 0, best = Number.POSITIVE_INFINITY
  for (let i=0;i<spend.length;i++){ const d = Math.abs(spend[i]-budget); if(d<best){best=d;idx=i} }
  const predConv = Number(conv[idx]||0)
  const cac = budget/Math.max(1,predConv)
  return { budget, channel, predicted_conversions: predConv, cac }
}

async function fetchCreative(params: URLSearchParams) {
  const qs = params.toString()
  const { absoluteUrl } = await import('../../lib/url')
  const r = await fetch(absoluteUrl(`/api/ads/creative?${qs}`), { cache: 'no-store' })
  if (!r.ok) return { error: `Creative lookup failed: ${r.status}` }
  return await r.json()
}

async function runControl(action: 'ga4_ingest'|'ads_ingest', arg?: string) {
  if (action === 'ga4_ingest') {
    const { absoluteUrl } = await import('../../lib/url')
    const r = await fetch(absoluteUrl('/api/control/ga4-ingest'), { method:'POST' })
    const j = await r.json().catch(()=>({}))
    return j
  }
  if (action === 'ads_ingest') {
    const { absoluteUrl } = await import('../../lib/url')
    const r = await fetch(absoluteUrl(`/api/control/ads-ingest?only=${encodeURIComponent(arg||'')}`), { method:'POST' })
    const j = await r.json().catch(()=>({}))
    return j
  }
  return { error: 'Unknown control action' }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(()=>({}))
    const userMessages = Array.isArray(body?.messages) ? body.messages : []
    const ctx = await buildContext()
    const system = contextToPrompt(ctx)
    const last = userMessages[userMessages.length-1]?.content || ''
    const cmd = parseCommand(last)
    if (cmd) {
      // Handle commands deterministically
      if (cmd.cmd === '/help') {
        return NextResponse.json({ reply: helpText() })
      }
      if (cmd.cmd === '/mmm') {
        const [ch, bud] = cmd.args.split(/\s+/)
        if (!ch || !bud) return NextResponse.json({ reply: 'Usage: /mmm <channel> <daily_budget>' })
        const res = await mmmWhatIf(ctx.projectId, ctx.dataset, ch, Number(bud))
        if ((res as any).error) return NextResponse.json({ reply: (res as any).error })
        const summary = `MMM what‑if for ${ch} at $${Number(bud).toFixed(0)}/day → conversions ≈ ${(res as any).predicted_conversions.toFixed(0)}, CAC ≈ $${((res as any).cac).toFixed(2)}`
        return NextResponse.json({ reply: summary, data: res })
      }
      if (cmd.cmd === '/creative') {
        const params = new URLSearchParams()
        cmd.args.split(/\s+/).forEach(pair=>{ const [k,v] = pair.split('='); if(k&&v) params.set(k,v) })
        if (!params.get('ad_id')) return NextResponse.json({ reply: 'Usage: /creative ad_id=<id> [campaign_id=..] [customer_id=..]' })
        const res = await fetchCreative(params)
        return NextResponse.json({ reply: JSON.stringify(res, null, 2), data: res })
      }
      if (cmd.cmd === '/run') {
        const arg = cmd.args.trim()
        if (arg.startsWith('ga4_ingest')) {
          const res = await runControl('ga4_ingest')
          return NextResponse.json({ reply: 'Triggered GA4 ingest (see Control).', data: res })
        }
        if (arg.startsWith('ads_ingest')) {
          const m = arg.match(/only=(\d{10})/)
          const only = m?.[1]
          const res = await runControl('ads_ingest', only)
          return NextResponse.json({ reply: `Triggered Ads ingest${only?` for ${only}`:''}.`, data: res })
        }
        return NextResponse.json({ reply: 'Usage: /run ga4_ingest | ads_ingest only=<CID>' })
      }
      if (cmd.cmd === '/sql') {
        const err = validateSql(cmd.args)
        if (err) return NextResponse.json({ reply: `SQL rejected: ${err}` })
        const rows:any[] = await runSql(ctx.projectId, ctx.dataset, cmd.args)
        if (Array.isArray(rows) && rows.length) {
          const first = rows[0]
          const keys = Object.keys(first)
          const dateKey = keys.find(k=>/date|day|time/i.test(k))
          const strKey = keys.find(k=> typeof first[k] === 'string')
          const numKey = keys.find(k=> typeof first[k] === 'number')
          const xKey = dateKey || strKey
          const yKey = numKey
          if (xKey && yKey) {
            const data = rows.map(r=> ({ ...r, [xKey]: String(r[xKey]) }))
            const viz = { type: dateKey ? 'line' : 'bar', title: `${yKey} by ${xKey}`, xKey, yKey, data }
            return NextResponse.json({ reply: `Here is ${yKey} by ${xKey} (${rows.length} rows).`, rows, viz })
          }
        }
        return NextResponse.json({ reply: JSON.stringify(rows, null, 2), rows })
      }
      return NextResponse.json({ reply: 'Unknown command. Type /help.' })
    }

    // LLM path with system context
    // Heuristic intents for simple numeric Q&A without calling LLM
    const text = String(last || '').toLowerCase()
    // "what was my cac yesterday" style
    if (/\bcac\b/.test(text) && /yesterday/.test(text)) {
      const row = await (async()=>{
        const { absoluteUrl } = await import('../../lib/url')
        const r = await fetch(absoluteUrl('/api/bq/kpi/yesterday'), { cache: 'no-store' })
        return await r.json()
      })()
      if (row?.row && row.row.date) {
        const d = row.row.date
        const cac = Number(row.row.cac || 0).toFixed(2)
        const conv = Number(row.row.conversions || 0).toFixed(0)
        const cost = Number(row.row.cost || 0).toFixed(2)
        return NextResponse.json({ reply: `CAC on ${d} was $${cac} (spend $${cost}, conversions ${conv}).` })
      }
      return NextResponse.json({ reply: 'No KPI rows for yesterday.' })
    }
    // "recommend" + "volume" → summarize from MMM & GA4
    if (/recommend/.test(text) && /volume/.test(text)) {
      const mmm = ctx.mmm
      const ga4Top = Array.isArray(ctx.ga4) ? ctx.ga4.slice(0,5).map((r:any)=> `${r.default_channel_group}:${Number(r.conv||0)}`) : []
      const lines = [] as string[]
      if (mmm) lines.push(`Scale ${mmm.channel} toward $${Number(mmm.proposed_daily_budget||0).toFixed(0)}/day (expected CAC $${Number(mmm.expected_cac||0).toFixed(2)}).`)
      if (ga4Top.length) lines.push(`Lean into top GA4 channels (28d): ${ga4Top.join(', ')}.`)
      lines.push('Run canary budget ups on strongest campaigns; expand RSAs with proven headlines; broaden audiences with guardrails; ensure dayparting and impression share headroom are utilized.')
      return NextResponse.json({ reply: lines.join(' ') })
    }

    // Try dynamic SQL planning for KPI/GA4 style questions
    if (/\b(cac|roas|spend|revenue|conversions|ga4|impression share|impressions|clicks)\b/i.test(last || '')) {
      const sqlCandidate = await planSqlFromLlm(String(last||''), ctx)
      if (sqlCandidate) {
        const err = validateSql(sqlCandidate)
        if (!err) {
          const rows:any[] = await runSql(ctx.projectId, ctx.dataset, sqlCandidate)
          if (Array.isArray(rows) && rows.length) {
            // Try to build a simple viz
            const first = rows[0]
            const keys = Object.keys(first)
            // Prefer date/time or string for x, numeric for y
            const dateKey = keys.find(k=>/date|day|time/i.test(k))
            const strKey = keys.find(k=> typeof first[k] === 'string')
            const numKey = keys.find(k=> typeof first[k] === 'number')
            const xKey = dateKey || strKey
            const yKey = numKey
            if (xKey && yKey) {
              const data = rows.map(r=> ({ ...r, [xKey]: String(r[xKey]) }))
              // Ensure plain JSON for client charts
              const dataPlain = JSON.parse(JSON.stringify(data))
              const isTime = !!dateKey
              const viz = { type: isTime ? 'line' : 'bar', title: `${yKey} by ${xKey}`, xKey, yKey, data: dataPlain }
              return NextResponse.json({ reply: `Here is ${yKey} by ${xKey} (${rows.length} rows).`, rows, viz })
            }
            return NextResponse.json({ reply: JSON.stringify(rows, null, 2), rows })
          }
        }
      }
    }

    const messages = [{ role: 'system', content: system }, ...userMessages]
    const reply = await callOpenAICompat({ messages })
    return NextResponse.json({ reply, context: { dataset: ctx.dataset, mode: ctx.mode } })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
