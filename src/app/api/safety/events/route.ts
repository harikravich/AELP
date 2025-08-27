import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const agentIds = searchParams.get('agentIds')?.split(',').filter(Boolean);
    const severity = searchParams.get('severity')?.split(',').filter(Boolean);
    const resolved = searchParams.get('resolved');
    const limit = parseInt(searchParams.get('limit') || '50');

    const filters = {
      ...(agentIds && { agentIds }),
      ...(severity && { severity }),
      ...(resolved !== null && { resolved: resolved === 'true' }),
      limit,
    };

    const events = await bigQueryClient.getSafetyEvents(filters);

    return NextResponse.json(events);
  } catch (error) {
    console.error('Error fetching safety events:', error);
    return NextResponse.json(
      { error: 'Failed to fetch safety events' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { agentId, eventType, severity, description } = body;

    // In a real implementation, this would insert the event into BigQuery
    console.log('Received safety event:', { agentId, eventType, severity, description });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error posting safety event:', error);
    return NextResponse.json(
      { error: 'Failed to post safety event' },
      { status: 500 }
    );
  }
}