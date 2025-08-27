import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const agentIds = searchParams.get('agentIds')?.split(',').filter(Boolean);
    const campaignIds = searchParams.get('campaignIds')?.split(',').filter(Boolean);
    const platforms = searchParams.get('platforms')?.split(',').filter(Boolean);
    const startTime = new Date(searchParams.get('startTime') || Date.now() - 24 * 60 * 60 * 1000);
    const endTime = new Date(searchParams.get('endTime') || Date.now());

    const filters = {
      startTime,
      endTime,
      ...(agentIds && { agentIds }),
      ...(campaignIds && { campaignIds }),
      ...(platforms && { platforms }),
    };

    const metrics = await bigQueryClient.getCampaignMetrics(filters);

    return NextResponse.json(metrics);
  } catch (error) {
    console.error('Error fetching campaign metrics:', error);
    return NextResponse.json(
      { error: 'Failed to fetch campaign metrics' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { campaignId, agentId, metrics } = body;

    // In a real implementation, this would insert metrics into BigQuery
    console.log('Received campaign metrics:', { campaignId, agentId, metrics });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error posting campaign metrics:', error);
    return NextResponse.json(
      { error: 'Failed to post campaign metrics' },
      { status: 500 }
    );
  }
}