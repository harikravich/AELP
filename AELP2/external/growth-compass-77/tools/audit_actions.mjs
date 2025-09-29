// Node 18+ required
const BASE = process.env.API_BASE || 'http://localhost:3000'

async function datasetSandbox() {
  const res = await fetch(`${BASE}/api/dataset?mode=sandbox`, { method:'POST', redirect:'manual' })
  const cookie = res.headers.get('set-cookie') || ''
  return cookie.split(';')[0] || ''
}

async function j(path, opts={}, cookie=''){
  const headers = { 'content-type':'application/json', ...(opts.headers||{}) }
  if (cookie) headers['cookie'] = cookie
  const res = await fetch(`${BASE}${path}`, { ...opts, headers })
  const text = await res.text()
  let json=null; try{ json = JSON.parse(text) }catch{}
  return { ok: res.ok, status: res.status, json, text }
}

function log(ok, label, extra=''){
  console.log(`${ok?'PASS':'FAIL'} ${label}${extra?` ${extra}`:''}`)
}

async function run(){
  const cookie = await datasetSandbox()
  log(!!cookie, 'dataset sandbox cookie set')

  // Creatives + Ads preview
  const creatives = await j('/api/bq/creatives', { method:'GET' })
  log(creatives.ok, 'GET /api/bq/creatives', `rows=${creatives.json?.rows?.length||0}`)
  const first = creatives.json?.rows?.[0] || {}
  if (first.ad_id) {
    const prev = await j(`/api/ads/creative?ad_id=${first.ad_id}&campaign_id=${first.campaign_id||''}&customer_id=${first.customer_id||''}`, { method:'GET' })
    log(prev.ok, 'GET /api/ads/creative', prev.ok?'':'(configure Ads creds)')
  }

  // Enqueue creative
  const enqueueA = await j('/api/control/creative/enqueue', { method:'POST', body: JSON.stringify({ platform:'google_ads', type:'rsa', campaign_id:first.campaign_id||null, ad_group_id:first.ad_group_id||null, payload:{ draft:true, source_ad_id:String(first.ad_id||'') }, requested_by:'audit' }) , headers:{}, }, cookie)
  log(enqueueA.json?.ok===true, 'POST /api/control/creative/enqueue', enqueueA.json?.run_id||'')
  const enqueueB = await j('/api/control/creative/enqueue', { method:'POST', body: JSON.stringify({ platform:'google_ads', type:'rsa', campaign_id:first.campaign_id||null, ad_group_id:first.ad_group_id||null, payload:{ draft:true, source_ad_id:String(first.ad_id||'') }, requested_by:'audit' }) , headers:{}, }, cookie)
  log(enqueueB.json?.ok===true, 'POST /api/control/creative/enqueue', enqueueB.json?.run_id||'')

  // Approvals queue & publish/reject if available
  const queue = await j('/api/bq/approvals/queue', {}, cookie)
  log(queue.ok, 'GET /api/bq/approvals/queue', `rows=${queue.json?.rows?.length||0}`)
  // Publish the first new run, then reject the second
  if (enqueueA.json?.run_id){
    const fd = new FormData(); fd.append('run_id', enqueueA.json.run_id)
    const pub = await fetch(`${BASE}/api/control/creative/publish`, { method:'POST', headers:{ cookie }, body: fd })
    log(pub.ok, 'POST /api/control/creative/publish', enqueueA.json.run_id)
  }
  if (enqueueB.json?.run_id){
    const rej = await j('/api/bq/approvals/reject', { method:'POST', body: JSON.stringify({ run_id: enqueueB.json.run_id }) }, cookie)
    log(rej.ok, 'POST /api/bq/approvals/reject', JSON.stringify(rej.json||{}))
  }

  // Spend actions
  const bandit = await j('/api/control/bandit-apply', { method:'POST', body: JSON.stringify({ lookback: 30 }) }, cookie)
  log(bandit.json?.ok===true, 'POST /api/control/bandit-apply')
  const opp = await j('/api/control/opportunity-approve', { method:'POST', body: JSON.stringify({ action:'approve', objective:'audit test', notes:'audit' }) }, cookie)
  log(opp.json?.ok===true, 'POST /api/control/opportunity-approve')

  // MMM what-if
  const whatif = await j('/api/bq/mmm/whatif?channel=google_ads&budget=5000')
  log(whatif.ok, 'GET /api/bq/mmm/whatif', JSON.stringify({ cac: Math.round(Number(whatif.json?.cac||0)) }))

  // GA4/Ads ingest (queued)
  const ga4 = await j('/api/control/ga4-ingest', { method:'POST' }, cookie)
  log(ga4.json?.ok===true, 'POST /api/control/ga4-ingest')
  const adsIngest = await j('/api/control/ads-ingest?only=8448503866', { method:'POST' }, cookie)
  log(adsIngest.json?.ok===true, 'POST /api/control/ads-ingest?only=8448503866')

  // Canvas pin/unpin
  const viz = { type:'chart', data:{ sample:true } }
  const fd = new FormData(); fd.append('title','Audit Pin'); fd.append('viz', JSON.stringify(viz))
  const pin = await fetch(`${BASE}/api/canvas/pin`, { method:'POST', headers:{ cookie }, body: fd, redirect:'manual' })
  log(pin.status===307 || pin.ok, 'POST /api/canvas/pin')
  // list and unpin latest if available
  const list = await j('/api/canvas/list', {}, cookie)
  const id = list.json?.items?.[0]?.id
  if (id){
    const fd2 = new FormData(); fd2.append('id', id)
    const unpin = await fetch(`${BASE}/api/canvas/unpin`, { method:'POST', headers:{ cookie }, body: fd2 })
    log(unpin.ok, 'POST /api/canvas/unpin', id)
  }

  // Auctions + RL + LTV
  const auct = await j('/api/bq/auctions/minutely')
  log(auct.ok, 'GET /api/bq/auctions/minutely', `rows=${auct.json?.rows?.length||0}`)
  const pol = await j('/api/bq/policy-enforcement')
  log(pol.ok, 'GET /api/bq/policy-enforcement', pol.json?.rows?`rows=${pol.json.rows.length}`:pol.json?.error||'')
  const off = await j('/api/bq/offpolicy')
  log(off.ok, 'GET /api/bq/offpolicy', `rows=${off.json?.rows?.length||0}`)
  const inter = await j('/api/bq/interference')
  log(inter.ok, 'GET /api/bq/interference', `rows=${inter.json?.rows?.length||0}`)
  const ltv = await j('/api/bq/ltv/summary')
  log(ltv.ok, 'GET /api/bq/ltv/summary', JSON.stringify(ltv.json||{}))
}

run().catch(e=>{ console.error(e); process.exit(2) })
