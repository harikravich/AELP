import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../../lib/dataset'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    const bq = new BigQuery({ projectId })
    const [rows] = await bq.query({ query: `
      SELECT path, count FROM \`${projectId}.${dataset}.journey_paths_daily\`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
        AND count IS NOT NULL
      ORDER BY count DESC LIMIT 200
    `})
    // Build Sankey nodes/links from path strings like "Organic Search>Direct>Paid Social"
    const nodeIndex = new Map<string, number>()
    const nodes: { name: string }[] = []
    const linkKey = (a:string,b:string)=> `${a}:::${b}`
    const linksMap = new Map<string, number>()
    const addNode = (name: string)=>{
      if (!nodeIndex.has(name)) { nodeIndex.set(name, nodes.length); nodes.push({ name }) }
      return nodeIndex.get(name)!
    }
    for (const r of rows as any[]) {
      const path = String(r.path || '')
      const c = Number(r.count || 0)
      if (!path || !c) continue
      const parts = path.split('>')
      for (let i=0; i<parts.length-1; i++) {
        const a = parts[i].trim(); const b = parts[i+1].trim()
        if (!a || !b) continue
        addNode(a); addNode(b)
        const key = linkKey(a,b)
        linksMap.set(key, (linksMap.get(key) || 0) + c)
      }
    }
    const links = Array.from(linksMap.entries()).map(([k,v])=>{
      const [a,b] = k.split(':::')
      return { source: nodeIndex.get(a)!, target: nodeIndex.get(b)!, value: v }
    })
    return NextResponse.json({ nodes, links })
  } catch (e:any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
