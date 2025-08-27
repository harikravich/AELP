import { NextRequest, NextResponse } from 'next/server';
import { bigQueryClient } from '@/lib/bigquery';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const metric = searchParams.get('metric') || 'roas';
    const startTime = new Date(searchParams.get('startTime') || Date.now() - 7 * 24 * 60 * 60 * 1000);
    const endTime = new Date(searchParams.get('endTime') || Date.now());
    const limit = parseInt(searchParams.get('limit') || '50');

    const timeRange = { start: startTime, end: endTime };
    const leaderboardData = await bigQueryClient.getLeaderboardData(metric, timeRange, limit);

    return NextResponse.json(leaderboardData);
  } catch (error) {
    console.error('Error fetching leaderboard:', error);
    return NextResponse.json(
      { error: 'Failed to fetch leaderboard data' },
      { status: 500 }
    );
  }
}