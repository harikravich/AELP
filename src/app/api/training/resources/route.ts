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

    // Fetch resource utilization for all specified agents
    const allResources = await Promise.all(
      agentIds.map(agentId => 
        bigQueryClient.getResourceUtilization(agentId, startTime, endTime)
      )
    );

    // Combine and sort resource data
    const combinedResources = allResources
      .flat()
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    return NextResponse.json(combinedResources);
  } catch (error) {
    console.error('Error fetching resource utilization:', error);
    return NextResponse.json(
      { error: 'Failed to fetch resource utilization' },
      { status: 500 }
    );
  }
}