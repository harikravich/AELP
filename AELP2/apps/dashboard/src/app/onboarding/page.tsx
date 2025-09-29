import React from 'react'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../lib/dataset'
import fs from 'fs'
import path from 'path'

async function credsPresence() {
  const cfgDir = path.join(process.cwd(), 'AELP2', 'config')
  const files = await fs.promises.readdir(cfgDir).catch(()=>[] as string[])
  const present = files.filter(f => f.includes('credentials') || f.includes('google_ads_credentials')).map(f => ({ file: f }))
  return present
}

async function lastIngest() {
  const projectId = process.env.GOOGLE_CLOUD_PROJECT as string
  const { dataset } = getDatasetFromCookie()
  const bq = new BigQuery({ projectId })
  const out: any[] = []
  const queries = [
    { platform: 'google_ads', sql: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.ads_campaign_performance\``},
    { platform: 'ga4', sql: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.ga4_aggregates\``},
    { platform: 'meta', sql: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.meta_campaign_performance\``},
    { platform: 'linkedin', sql: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.linkedin_campaign_performance\``},
    { platform: 'tiktok', sql: `SELECT MAX(DATE(date)) AS d FROM \`${projectId}.${dataset}.tiktok_campaign_performance\``},
  ]
  for (const q of queries) {
    try { const [rows] = await bq.query({ query: q.sql }); out.push({ platform: q.platform, date: rows[0]?.d || null }) } catch {}
  }
  return out
}

export default async function OnboardingPage() {
  const creds = await credsPresence()
  const last = await lastIngest()
  return (
    <div className="space-y-6">
      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Credentials</h2>
        <p className="text-sm text-gray-600">Presence of local credentials files in AELP2/config</p>
        <ul className="list-disc ml-6 mt-2 text-sm">
          {creds.map((c:any, i:number) => (<li key={i}>{c.file}</li>))}
          {creds.length === 0 && (<li className="text-gray-500">No credential files detected.</li>)}
        </ul>
      </div>
      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Last Ingest Dates</h2>
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left border-b">
                <th className="py-2 pr-4">Platform</th>
                <th className="py-2 pr-4">Last Date</th>
              </tr>
            </thead>
            <tbody>
              {last.map((r:any, i:number) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2 pr-4">{r.platform}</td>
                  <td className="py-2 pr-4">{r.date || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="bg-white shadow-sm rounded p-4">
        <h2 className="text-lg font-medium">Actions</h2>
        <div className="mt-3 space-x-2">
          {['meta','linkedin','tiktok'].map((p) => (
            <form key={p} action={`/api/control/onboarding/create?platform=${p}`} method="post" className="inline-block">
              <button className="px-3 py-1 text-sm rounded bg-emerald-600 text-white">Create Skeleton ({p})</button>
            </form>
          ))}
        </div>
        <div className="mt-3 space-x-2">
          {['meta','linkedin','tiktok'].map((p) => (
            <form key={p} action={`/api/control/onboarding/backfill?platform=${p}`} method="post" className="inline-block">
              <button className="px-3 py-1 text-sm rounded bg-blue-600 text-white">Backfill 30d ({p})</button>
            </form>
          ))}
        </div>
      </div>
    </div>
  )
}

export const dynamic = 'force-dynamic'
