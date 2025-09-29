import { NextRequest, NextResponse } from 'next/server'
import { GoogleAdsApi } from 'google-ads-api'

// Initialize Google Ads client
const client = new GoogleAdsApi({
  client_id: process.env.GOOGLE_ADS_CLIENT_ID!,
  client_secret: process.env.GOOGLE_ADS_CLIENT_SECRET!,
  developer_token: process.env.GOOGLE_ADS_DEVELOPER_TOKEN!,
})

const customer = client.Customer({
  customer_id: process.env.GOOGLE_ADS_CUSTOMER_ID!,
  refresh_token: process.env.GOOGLE_ADS_REFRESH_TOKEN!,
})

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const {
      campaignId,
      adGroupId,
      headlines,
      descriptions,
      finalUrls,
      path1,
      path2,
      action = 'create', // create, update, pause, enable
    } = body

    if (!campaignId || !adGroupId) {
      return NextResponse.json(
        { error: 'Campaign ID and Ad Group ID are required' },
        { status: 400 }
      )
    }

    let result

    switch (action) {
      case 'create':
        result = await createResponsiveSearchAd(
          customer,
          campaignId,
          adGroupId,
          headlines,
          descriptions,
          finalUrls,
          path1,
          path2
        )
        break

      case 'update':
        result = await updateAd(customer, body.adId, headlines, descriptions)
        break

      case 'pause':
        result = await pauseAd(customer, body.adId)
        break

      case 'enable':
        result = await enableAd(customer, body.adId)
        break

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        )
    }

    return NextResponse.json({
      success: true,
      action,
      result,
      timestamp: new Date().toISOString(),
    })
  } catch (error: any) {
    console.error('Google Ads publish error:', error)
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to publish to Google Ads',
        details: error.errors || [],
      },
      { status: 500 }
    )
  }
}

async function createResponsiveSearchAd(
  customer: any,
  campaignId: string,
  adGroupId: string,
  headlines: string[],
  descriptions: string[],
  finalUrls: string[],
  path1?: string,
  path2?: string
) {
  // Build the ad
  const ad = {
    ad_group: `customers/${customer.credentials.customer_id}/adGroups/${adGroupId}`,
    status: 'ENABLED',
    responsive_search_ad: {
      headlines: headlines.slice(0, 15).map((text) => ({
        text,
        pinned_field: null, // Can pin specific positions if needed
      })),
      descriptions: descriptions.slice(0, 4).map((text) => ({
        text,
        pinned_field: null,
      })),
      path1,
      path2,
    },
    final_urls: finalUrls,
  }

  // Create the ad
  const response = await customer.adGroupAds.create([
    {
      ad_group_ad: ad,
    },
  ])

  // Log to BigQuery for tracking
  await logAdCreation({
    customerId: customer.credentials.customer_id,
    campaignId,
    adGroupId,
    adId: response.results[0].resource_name,
    headlines,
    descriptions,
    finalUrls,
  })

  return response
}

async function updateAd(
  customer: any,
  adId: string,
  headlines?: string[],
  descriptions?: string[]
) {
  const updateOperations: any[] = []

  if (headlines) {
    updateOperations.push({
      entity: 'ad',
      resource_name: adId,
      update: {
        responsive_search_ad: {
          headlines: headlines.slice(0, 15).map((text) => ({ text })),
        },
      },
      update_mask: 'responsive_search_ad.headlines',
    })
  }

  if (descriptions) {
    updateOperations.push({
      entity: 'ad',
      resource_name: adId,
      update: {
        responsive_search_ad: {
          descriptions: descriptions.slice(0, 4).map((text) => ({ text })),
        },
      },
      update_mask: 'responsive_search_ad.descriptions',
    })
  }

  const response = await customer.ads.update(updateOperations)
  return response
}

async function pauseAd(customer: any, adId: string) {
  const response = await customer.adGroupAds.update([
    {
      entity: 'ad_group_ad',
      resource_name: adId,
      update: { status: 'PAUSED' },
      update_mask: 'status',
    },
  ])
  return response
}

async function enableAd(customer: any, adId: string) {
  const response = await customer.adGroupAds.update([
    {
      entity: 'ad_group_ad',
      resource_name: adId,
      update: { status: 'ENABLED' },
      update_mask: 'status',
    },
  ])
  return response
}

async function logAdCreation(data: any) {
  // Log to BigQuery for tracking
  try {
    const { BigQuery } = require('@google-cloud/bigquery')
    const bigquery = new BigQuery()
    const dataset = bigquery.dataset(process.env.BIGQUERY_TRAINING_DATASET!)
    const table = dataset.table('creative_publish_log')

    await table.insert({
      ...data,
      created_at: new Date().toISOString(),
      source: 'dashboard_api',
    })
  } catch (error) {
    console.error('Failed to log ad creation:', error)
  }
}