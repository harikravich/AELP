// Node 18+ required
const BASE = process.env.API_BASE || process.env.VITE_API_BASE_URL || ''
if (!BASE) {
  console.error('Set API_BASE to Next.js origin, e.g., http://localhost:3000')
  process.exit(1)
}

async function check(path, opts) {
  const url = `${BASE}${path}`
  const res = await fetch(url, { ...opts, headers: { 'content-type':'application/json', ...(opts?.headers||{}) } })
  const ok = res.ok
  const json = await res.json().catch(()=>null)
  console.log(`${ok?'OK':'ERR'} ${res.status} ${path}`, json && json.rows ? `rows=${json.rows.length}` : '')
  return ok
}

const main = async () => {
  const ok = await Promise.all([
    check('/api/dataset'),
    check('/api/bq/kpi/summary'),
    check('/api/bq/kpi/daily?days=7'),
    check('/api/bq/ga4/channels'),
    check('/api/bq/creatives'),
    check('/api/bq/approvals/queue'),
    check('/api/bq/auctions/minutely'),
    check('/api/bq/mmm/allocations')
  ])
  process.exit(ok.every(Boolean) ? 0 : 2)
}

main()
