import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get('status')?.split(',').filter(Boolean);
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');

    const filters = {
      ...(status && { status }),
      limit,
      offset,
    };

    const agents = await bigQueryClient.getAgents(filters);

    return NextResponse.json(agents);
  } catch (error) {
    console.error('Error fetching agents:', error);
    return NextResponse.json(
      { error: 'Failed to fetch agents' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { name, version, config } = body;

    // In a real implementation, this would create a new agent in BigQuery
    console.log('Creating new agent:', { name, version, config });

    return NextResponse.json({ 
      success: true, 
      agentId: `agent-${Date.now()}` 
    });
  } catch (error) {
    console.error('Error creating agent:', error);
    return NextResponse.json(
      { error: 'Failed to create agent' },
      { status: 500 }
    );
  }
}