import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const agentIds = searchParams.get('agentIds')?.split(',').filter(Boolean) || [];
    const startTime = new Date(searchParams.get('startTime') || Date.now() - 24 * 60 * 60 * 1000);
    const endTime = new Date(searchParams.get('endTime') || Date.now());

    if (agentIds.length === 0) {
      return NextResponse.json([]);
    }

    // Fetch training metrics for all specified agents
    const allMetrics = await Promise.all(
      agentIds.map(agentId => 
        bigQueryClient.getTrainingMetrics(agentId, startTime, endTime)
      )
    );

    // Combine and sort metrics
    const combinedMetrics = allMetrics
      .flat()
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    return NextResponse.json(combinedMetrics);
  } catch (error) {
    console.error('Error fetching training metrics:', error);
    return NextResponse.json(
      { error: 'Failed to fetch training metrics' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { agentId, metrics } = body;

    // In a real implementation, this would insert metrics into BigQuery
    // For now, we'll return a success response
    console.log('Received training metrics for agent:', agentId, metrics);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error posting training metrics:', error);
    return NextResponse.json(
      { error: 'Failed to post training metrics' },
      { status: 500 }
    );
  }
}