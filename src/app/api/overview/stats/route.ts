import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const startTime = new Date(searchParams.get('startTime') || Date.now() - 24 * 60 * 60 * 1000);
    const endTime = new Date(searchParams.get('endTime') || Date.now());

    // Get agents data
    const agents = await bigQueryClient.getAgents();
    const activeAgents = agents.filter(agent => agent.status === 'training' || agent.status === 'deployed');

    // Get campaign metrics for the time range
    const campaignMetrics = await bigQueryClient.getCampaignMetrics({
      startTime,
      endTime,
    });

    // Get safety events
    const safetyEvents = await bigQueryClient.getSafetyEvents({
      resolved: false,
    });

    // Calculate aggregated stats
    const totalSpend = campaignMetrics.reduce((sum, metric) => sum + metric.spend, 0);
    const totalRevenue = campaignMetrics.reduce((sum, metric) => sum + metric.revenue, 0);
    const avgROAS = totalSpend > 0 ? totalRevenue / totalSpend : 0;

    // Get unique campaigns
    const uniqueCampaigns = new Set(campaignMetrics.map(metric => metric.campaignId));
    const activeCampaigns = uniqueCampaigns.size;

    const stats = {
      totalAgents: agents.length,
      activeAgents: activeAgents.length,
      totalCampaigns: activeCampaigns,
      activeCampaigns,
      totalSpend,
      totalRevenue,
      avgROAS,
      safetyEvents: safetyEvents.length,
    };

    return NextResponse.json(stats);
  } catch (error) {
    console.error('Error fetching overview stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch overview stats' },
      { status: 500 }
    );
  }
}