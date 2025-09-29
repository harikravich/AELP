import { NextResponse } from 'next/server'
import { BigQuery } from '@google-cloud/bigquery'
import { getDatasetFromCookie } from '../../../../lib/dataset'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

async function lookupCustomerId(projectId: string, dataset: string, campaignId: string): Promise<string | null> {
  try {
    const bq = new BigQuery({ projectId })
    const sql = `
      SELECT ANY_VALUE(customer_id) AS cid
      FROM \`${projectId}.${dataset}.ads_campaign_performance\`
      WHERE CAST(campaign_id AS STRING)=@cid`
    const [rows] = await bq.query({ query: sql, params: { cid: campaignId } })
    if (rows && rows[0] && rows[0].cid) return String(rows[0].cid)
  } catch {}
  return null
}

async function getAccessToken(client_id: string, client_secret: string, refresh_token: string): Promise<string> {
  const params = new URLSearchParams({
    client_id,
    client_secret,
    refresh_token,
    grant_type: 'refresh_token',
  })
  const resp = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params.toString(),
  })
  const json = await resp.json()
  if (!resp.ok) throw new Error(json.error_description || 'Failed to obtain access token')
  return json.access_token
}

async function googleAdsSearch({ version, customerId, developerToken, loginCustomerId, accessToken, query }:{ version:string, customerId:string, developerToken:string, loginCustomerId?:string, accessToken:string, query:string }) {
  // Prefer searchStream (more permissive), fallback to search
  const streamUrl = `https://googleads.googleapis.com/${version}/customers/${customerId}/googleAds:searchStream`
  const searchUrl = `https://googleads.googleapis.com/${version}/customers/${customerId}/googleAds:search`
  const headers: Record<string,string> = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`,
    'developer-token': developerToken,
  }
  if (loginCustomerId) headers['login-customer-id'] = loginCustomerId
  // Try searchStream first
  let resp = await fetch(streamUrl, { method: 'POST', headers, body: JSON.stringify({ query }) })
  try {
    let text = await resp.text()
    // searchStream may return JSON array; if not, try to parse each line
    let json: any
    try { json = JSON.parse(text) } catch { json = [{ results: [] }] }
    if (resp.ok) {
      const results = (Array.isArray(json) ? json : [json]).flatMap((chunk:any)=> chunk.results || [])
      return results
    }
    // If stream failed, fall back to search
  } catch {}
  resp = await fetch(searchUrl, { method: 'POST', headers, body: JSON.stringify({ query }) })
  const json = await resp.json().catch(()=>({}))
  if (!resp.ok) throw new Error(json?.error?.message || `Google Ads search failed (${resp.status})`)
  return (json.results || [])
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url)
    const adId = url.searchParams.get('ad_id')
    const campaignId = url.searchParams.get('campaign_id')
    let customerId = url.searchParams.get('customer_id')
    if (!adId) return NextResponse.json({ error: 'Missing ad_id' }, { status: 400 })

    // Resolve customer id if missing from BQ
    const projectId = process.env.GOOGLE_CLOUD_PROJECT!
    const { dataset } = getDatasetFromCookie()
    if (!customerId && campaignId) customerId = await lookupCustomerId(projectId, dataset, campaignId)
    if (!customerId) return NextResponse.json({ error: 'customer_id required or could not be inferred from BigQuery' }, { status: 400 })

    const dev = process.env.GOOGLE_ADS_DEVELOPER_TOKEN
    const cid = process.env.GOOGLE_ADS_CLIENT_ID
    const cs = process.env.GOOGLE_ADS_CLIENT_SECRET
    const rt = process.env.GOOGLE_ADS_REFRESH_TOKEN
    const login = (process.env.GOOGLE_ADS_LOGIN_CUSTOMER_ID || '').replace(/-/g,'') || undefined
    if (!dev || !cid || !cs || !rt) return NextResponse.json({ error: 'Missing Google Ads API env (client id/secret, refresh token, developer token).' }, { status: 500 })

    const accessToken = await getAccessToken(cid, cs, rt)
    const version = 'v19'
    const cust = String(customerId).replace(/-/g,'')

    // Fetch ad details via REST search (GAQL)
    let adRows = await googleAdsSearch({
      version,
      customerId: cust,
      developerToken: dev,
      loginCustomerId: login,
      accessToken,
      query: `SELECT ad_group.id, ad_group.name, ad_group_ad.ad.id, ad_group_ad.ad.type, ad_group_ad.ad.name, ad_group_ad.ad.responsive_search_ad.headlines, ad_group_ad.ad.responsive_search_ad.descriptions, ad_group_ad.ad.responsive_search_ad.path1, ad_group_ad.ad.responsive_search_ad.path2, ad_group_ad.ad.responsive_display_ad.marketing_images, ad_group_ad.ad.responsive_display_ad.square_marketing_images, ad_group_ad.ad.final_urls FROM ad_group_ad WHERE ad_group_ad.ad.id = ${adId} AND ad_group_ad.status != 'REMOVED' LIMIT 1`,
    })
    // Fallback to querying the ad resource directly if GAQL above is rejected
    if (!adRows.length) {
      try {
        adRows = await googleAdsSearch({
          version,
          customerId: cust,
          developerToken: dev,
          loginCustomerId: login,
          accessToken,
          query: `SELECT ad.id, ad.type, ad.name, ad.responsive_search_ad.headlines, ad.responsive_search_ad.descriptions, ad.responsive_search_ad.path1, ad.responsive_search_ad.path2, ad.final_urls FROM ad WHERE ad.id = ${adId} LIMIT 1`,
        })
      } catch {}
    }
    if (!adRows.length) return NextResponse.json({ error: 'Ad not found' }, { status: 404 })
    const row: any = adRows[0]
    const ad = row.adGroupAd?.ad || row.ad_group_ad?.ad || {}
    const rsa = ad.responsiveSearchAd || ad.responsive_search_ad || {}
    const rda = ad.responsiveDisplayAd || ad.responsive_display_ad || {}

    // Collect image asset resource names if present
    const imageAssetNames: string[] = []
    try { (rda?.marketingImages||rda?.marketing_images||[]).forEach((it:any)=> it?.asset && imageAssetNames.push(it.asset)) } catch {}
    try { (rda?.squareMarketingImages||rda?.square_marketing_images||[]).forEach((it:any)=> it?.asset && imageAssetNames.push(it.asset)) } catch {}

    let imageUrls: string[] = []
    if (imageAssetNames.length) {
      const inList = imageAssetNames.map((a)=>`\"${a}\"`).join(',')
      const assetRows = await googleAdsSearch({
        version,
        customerId: cust,
        developerToken: dev,
        loginCustomerId: login,
        accessToken,
        query: `SELECT asset.resource_name, asset.image_asset.full_size.url FROM asset WHERE asset.resource_name IN (${inList}) LIMIT 25`,
      })
      imageUrls = (assetRows||[]).map((r:any)=> r.asset?.imageAsset?.fullSize?.url || r.asset?.image_asset?.full_size?.url).filter(Boolean)
    }

    const headlines = (rsa?.headlines || rsa?.headlines || []).map((h:any)=> h?.text || h?.assetText || h?.textAsset?.text).filter(Boolean)
    const descriptions = (rsa?.descriptions || []).map((d:any)=> d?.text || d?.assetText || d?.textAsset?.text).filter(Boolean)
    const path1 = rsa?.path1 || (rsa?.path || [])[0] || null
    const path2 = rsa?.path2 || (rsa?.path || [])[1] || null
    const finalUrls = ad.finalUrls || ad.final_urls || []

    return NextResponse.json({
      ad_id: String(ad.id || adId),
      ad_group: row.adGroup?.name || row.ad_group?.name || null,
      type: ad.type || 'RESPONSIVE_SEARCH_AD',
      name: ad.name || '',
      headlines,
      descriptions,
      path1,
      path2,
      final_urls: finalUrls,
      images: imageUrls,
    })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || String(e) }, { status: 500 })
  }
}
