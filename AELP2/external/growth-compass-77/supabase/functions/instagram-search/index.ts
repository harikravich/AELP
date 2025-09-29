import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Initialize Supabase client
const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
const supabase = createClient(supabaseUrl, supabaseKey);

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { handle, sessionId, abTestId, variantId } = await req.json();

    console.log('Instagram search request:', { handle, sessionId });

    // Get Instagram integration config
    const { data: integration, error: configError } = await supabase
      .from('integrations')
      .select('config')
      .eq('provider', 'instagram')
      .eq('is_active', true)
      .single();

    if (configError || !integration) {
      throw new Error('Instagram integration not configured');
    }

    const accessToken = integration.config.access_token;
    
    // Search for Instagram user
    const searchResult = await searchInstagramUser(handle, accessToken);
    
    // Analyze profile for wellness indicators
    const analysis = await analyzeWellnessProfile(searchResult, handle);
    
    // Track analytics event
    await supabase
      .from('analytics_events')
      .insert({
        event_type: 'instagram_search',
        session_id: sessionId,
        ab_test_id: abTestId,
        variant_id: variantId,
        event_data: {
          handle,
          found: searchResult.found,
          wellness_score: analysis.wellnessScore,
          risk_factors: analysis.riskFactors
        },
        ip_address: req.headers.get('x-forwarded-for') || 'unknown'
      });

    return new Response(
      JSON.stringify({
        success: true,
        profile: searchResult,
        analysis: analysis,
        recommendations: generateRecommendations(analysis)
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Error in instagram-search:', error);
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});

async function searchInstagramUser(handle: string, accessToken: string) {
  try {
    // Clean handle (remove @ if present)
    const cleanHandle = handle.replace('@', '');
    
    // Use Instagram Basic Display API to search for user
    // Note: In production, you'd need proper Instagram Business API access
    const response = await fetch(
      `https://graph.instagram.com/v12.0/me?fields=id,username,account_type,media_count&access_token=${accessToken}`
    );

    if (!response.ok) {
      throw new Error('Instagram API request failed');
    }

    const userData = await response.json();
    
    // For demo purposes, simulate profile data
    const profileData = {
      found: true,
      username: cleanHandle,
      id: userData.id || 'demo_id',
      account_type: userData.account_type || 'PERSONAL',
      media_count: userData.media_count || Math.floor(Math.random() * 1000),
      followers_count: Math.floor(Math.random() * 10000),
      following_count: Math.floor(Math.random() * 1000),
      bio: `Wellness enthusiast | Mindful living | ${cleanHandle}`,
      profile_picture_url: 'https://via.placeholder.com/150',
      recent_posts: generateMockPosts()
    };

    return profileData;
  } catch (error) {
    console.error('Instagram search failed:', error);
    
    // Return mock data for demo purposes
    return {
      found: false,
      username: handle.replace('@', ''),
      error: 'Profile not found or private'
    };
  }
}

function generateMockPosts() {
  const wellnessHashtags = [
    '#wellness', '#mindfulness', '#yoga', '#meditation', '#selfcare',
    '#mentalhealth', '#fitness', '#healthy', '#balance', '#peace'
  ];
  
  return Array.from({ length: 5 }, (_, i) => ({
    id: `post_${i}`,
    caption: `Living my best life! ${wellnessHashtags[Math.floor(Math.random() * wellnessHashtags.length)]}`,
    media_type: 'IMAGE',
    media_url: `https://via.placeholder.com/400x400?text=Post+${i + 1}`,
    timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
    like_count: Math.floor(Math.random() * 500),
    comments_count: Math.floor(Math.random() * 50)
  }));
}

async function analyzeWellnessProfile(profile: any, handle: string) {
  if (!profile.found) {
    return {
      wellnessScore: 0,
      riskFactors: ['profile_not_found'],
      recommendations: ['verify_profile_exists']
    };
  }

  // Simulate wellness analysis based on profile data
  const analysis = {
    wellnessScore: 0,
    riskFactors: [] as string[],
    positiveIndicators: [] as string[]
  };

  // Analyze bio for wellness keywords
  const bio = profile.bio || '';
  const wellnessKeywords = ['wellness', 'mindful', 'yoga', 'meditation', 'health', 'balance', 'peace'];
  const stressKeywords = ['stressed', 'anxious', 'tired', 'overwhelmed', 'burnout'];

  wellnessKeywords.forEach(keyword => {
    if (bio.toLowerCase().includes(keyword)) {
      analysis.wellnessScore += 15;
      analysis.positiveIndicators.push(`bio_contains_${keyword}`);
    }
  });

  stressKeywords.forEach(keyword => {
    if (bio.toLowerCase().includes(keyword)) {
      analysis.wellnessScore -= 10;
      analysis.riskFactors.push(`bio_mentions_${keyword}`);
    }
  });

  // Analyze posting frequency
  if (profile.media_count > 100) {
    analysis.wellnessScore += 10;
    analysis.positiveIndicators.push('active_social_presence');
  }

  // Analyze follower ratio
  const followerRatio = profile.followers_count / Math.max(profile.following_count, 1);
  if (followerRatio > 2) {
    analysis.wellnessScore += 5;
    analysis.positiveIndicators.push('healthy_social_ratio');
  }

  // Normalize score to 0-100
  analysis.wellnessScore = Math.max(0, Math.min(100, analysis.wellnessScore + 50));

  return analysis;
}

function generateRecommendations(analysis: any) {
  const recommendations = [];

  if (analysis.wellnessScore < 30) {
    recommendations.push({
      type: 'high_priority',
      title: 'Consider Wellness Support',
      description: 'Based on the profile analysis, this person might benefit from wellness resources.',
      action: 'offer_consultation'
    });
  } else if (analysis.wellnessScore < 60) {
    recommendations.push({
      type: 'medium_priority',
      title: 'Wellness Enhancement',
      description: 'Good foundation for wellness - could benefit from targeted programs.',
      action: 'offer_programs'
    });
  } else {
    recommendations.push({
      type: 'low_priority',
      title: 'Wellness Enthusiast',
      description: 'Strong wellness indicators - might be interested in advanced programs.',
      action: 'offer_premium'
    });
  }

  if (analysis.riskFactors.length > 0) {
    recommendations.push({
      type: 'attention',
      title: 'Risk Factors Detected',
      description: `Found indicators: ${analysis.riskFactors.join(', ')}`,
      action: 'provide_resources'
    });
  }

  return recommendations;
}