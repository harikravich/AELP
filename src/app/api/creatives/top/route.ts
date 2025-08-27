import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const startTime = new Date(searchParams.get('startTime') || Date.now() - 7 * 24 * 60 * 60 * 1000);
    const endTime = new Date(searchParams.get('endTime') || Date.now());
    const limit = parseInt(searchParams.get('limit') || '10');

    // Mock data for top creatives - in production this would query BigQuery
    const mockCreatives = [
      {
        creativeId: 'creative-001',
        name: 'Summer Sale Banner',
        type: 'image',
        avgCTR: 0.045,
        avgConversionRate: 0.032,
        avgROAS: 4.2,
        impressions: 125000,
        platforms: ['Facebook', 'Google'],
        lastUsed: new Date(),
      },
      {
        creativeId: 'creative-002',
        name: 'Product Demo Video',
        type: 'video',
        avgCTR: 0.038,
        avgConversionRate: 0.028,
        avgROAS: 3.8,
        impressions: 98000,
        platforms: ['YouTube', 'TikTok'],
        lastUsed: new Date(),
      },
      {
        creativeId: 'creative-003',
        name: 'Limited Time Offer',
        type: 'text',
        avgCTR: 0.041,
        avgConversionRate: 0.025,
        avgROAS: 3.5,
        impressions: 87000,
        platforms: ['Google', 'Bing'],
        lastUsed: new Date(),
      },
      {
        creativeId: 'creative-004',
        name: 'Holiday Collection',
        type: 'image',
        avgCTR: 0.036,
        avgConversionRate: 0.022,
        avgROAS: 3.2,
        impressions: 76000,
        platforms: ['Instagram', 'Facebook'],
        lastUsed: new Date(),
      },
      {
        creativeId: 'creative-005',
        name: 'Customer Testimonial',
        type: 'video',
        avgCTR: 0.033,
        avgConversionRate: 0.019,
        avgROAS: 2.9,
        impressions: 65000,
        platforms: ['LinkedIn', 'Facebook'],
        lastUsed: new Date(),
      },
    ];

    const topCreatives = mockCreatives
      .sort((a, b) => b.avgROAS - a.avgROAS)
      .slice(0, limit);

    return NextResponse.json(topCreatives);
  } catch (error) {
    console.error('Error fetching top creatives:', error);
    return NextResponse.json(
      { error: 'Failed to fetch top creatives' },
      { status: 500 }
    );
  }
}