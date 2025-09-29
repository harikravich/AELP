import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
const openaiApiKey = Deno.env.get('OPENAI_API_KEY')!;

const supabase = createClient(supabaseUrl, supabaseKey);

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { campaignId, briefing } = await req.json();

    console.log('AI Brainstorming request:', { campaignId, briefing });

    // Get campaign details
    const { data: campaign, error: campaignError } = await supabase
      .from('ai_campaigns')
      .select('*')
      .eq('id', campaignId)
      .single();

    if (campaignError || !campaign) {
      throw new Error('Campaign not found');
    }

    // Create specialized AI agents for this campaign
    const agents = await createCampaignAgents(campaignId, campaign);

    // Generate concepts using different AI agents
    const concepts = await generateConceptsWithAgents(campaign, briefing, agents);

    // Store concepts in database
    const storedConcepts = await storeConcepts(campaignId, concepts);

    return new Response(
      JSON.stringify({
        success: true,
        campaign: campaign,
        agents: agents,
        concepts: storedConcepts,
        summary: {
          totalConcepts: concepts.length,
          avgCreativityScore: concepts.reduce((acc, c) => acc + c.creativity_score, 0) / concepts.length,
          topConcept: concepts.sort((a, b) => b.creativity_score - a.creativity_score)[0]
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Error in ai-brainstorm:', error);
    return new Response(
      JSON.stringify({ success: false, error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

async function createCampaignAgents(campaignId: string, campaign: any) {
  const agentTemplates = [
    {
      agent_type: 'concept_brainstormer',
      name: 'Creative Concept Generator',
      personality: 'innovative, bold, thinks outside the box',
      expertise: 'viral marketing, psychological triggers, creative campaigns',
      prompt_template: `You are a world-class creative strategist specializing in ${campaign.industry} marketing. 
        Your mission is to brainstorm innovative landing page concepts that convert.
        
        Focus on:
        - Psychological triggers that drive action
        - Unique value propositions that stand out
        - Interactive elements that engage users
        - Converting skeptics into believers
        
        Think like the best agencies: Ogilvy, Wieden+Kennedy, Droga5.`
    },
    {
      agent_type: 'component_generator',
      name: 'Technical Component Architect',
      personality: 'systematic, detail-oriented, solution-focused',
      expertise: 'React components, UX optimization, conversion engineering',
      prompt_template: `You are a senior frontend architect and conversion optimization expert.
        Your role is to design functional components that maximize conversions.
        
        Specialize in:
        - Interactive components that guide user behavior
        - Data capture mechanisms that feel natural
        - Trust-building elements and social proof
        - Mobile-first responsive designs
        
        Think like: Unbounce, Leadpages, but with custom AI-powered functionality.`
    },
    {
      agent_type: 'copy_writer',
      name: 'Conversion Copywriter',
      personality: 'persuasive, empathetic, results-driven',
      expertise: 'direct response copy, emotional triggers, conversion optimization',
      prompt_template: `You are a direct response copywriter with expertise in ${campaign.industry}.
        Your words directly impact conversion rates.
        
        Master of:
        - Headlines that stop the scroll
        - Benefits over features
        - Urgency and scarcity tactics
        - Emotional resonance with ${campaign.target_audience}
        
        Write like: David Ogilvy, Gary Halbert, Eugene Schwartz.`
    },
    {
      agent_type: 'optimizer',
      name: 'Growth Hacker',
      personality: 'analytical, experimental, data-obsessed',
      expertise: 'A/B testing, analytics, growth optimization',
      prompt_template: `You are a growth hacker focused on optimization and experimentation.
        Your job is to identify optimization opportunities and test hypotheses.
        
        Expert in:
        - A/B testing strategies
        - Conversion funnel analysis
        - User behavior prediction
        - Performance metrics that matter
        
        Think like: Sean Ellis, Brian Balfour, Neil Patel.`
    }
  ];

  const createdAgents = [];
  
  for (const template of agentTemplates) {
    const { data: agent, error } = await supabase
      .from('ai_agents')
      .insert([{
        campaign_id: campaignId,
        ...template,
        config: {
          max_completion_tokens: 2000,
          model: 'gpt-5-2025-08-07'
        }
      }])
      .select()
      .single();

    if (!error && agent) {
      createdAgents.push(agent);
    }
  }

  return createdAgents;
}

async function generateConceptsWithAgents(campaign: any, briefing: any, agents: any[]) {
  const concepts = [];
  
  const brainstormerAgent = agents.find(a => a.agent_type === 'concept_brainstormer');
  
  if (!brainstormerAgent) {
    throw new Error('No brainstormer agent found');
  }

  // Generate 5 different concepts using AI
  for (let i = 0; i < 5; i++) {
    const concept = await generateSingleConcept(campaign, briefing, brainstormerAgent, i + 1);
    concepts.push(concept);
  }

  return concepts;
}

async function generateSingleConcept(campaign: any, briefing: any, agent: any, conceptNumber: number) {
  const prompt = `${agent.prompt_template}

CAMPAIGN BRIEF:
Industry: ${campaign.industry}
Target Audience: ${campaign.target_audience}
Business Goals: ${campaign.business_goals}
Tone: ${campaign.tone_of_voice}

ADDITIONAL BRIEFING:
${JSON.stringify(briefing, null, 2)}

TASK: Generate a unique landing page concept #${conceptNumber} that will convert ${campaign.target_audience} in the ${campaign.industry} industry.

Respond with a JSON object containing:
{
  "concept_name": "Catchy name for this concept",
  "concept_description": "Detailed description of the landing page concept",
  "target_problem": "Specific problem this addresses",
  "solution_approach": "How this concept solves the problem",
  "key_benefits": ["benefit1", "benefit2", "benefit3"],
  "psychological_triggers": ["trigger1", "trigger2", "trigger3"],
  "components_needed": ["component1", "component2", "component3"],
  "estimated_conversion_rate": 8.5,
  "creativity_score": 85,
  "feasibility_score": 90,
  "unique_selling_point": "What makes this different from typical landing pages",
  "target_emotion": "Primary emotion to evoke",
  "call_to_action": "Specific CTA for this concept"
}

Make this concept UNIQUE and INNOVATIVE. Think beyond typical landing pages.`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiApiKey}`,
      'Content-Type': 'application/json',
    },
      body: JSON.stringify({
        model: 'gpt-5-2025-08-07',
        messages: [
          { role: 'system', content: 'You are a world-class creative strategist. Respond only with valid JSON.' },
          { role: 'user', content: prompt }
        ],
        max_completion_tokens: 2000
      }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status}`);
  }

  const data = await response.json();
  const content = data.choices[0].message.content;

  try {
    const concept = JSON.parse(content);
    return {
      agent_id: agent.id,
      ...concept,
      metadata: {
        generated_at: new Date().toISOString(),
        model_used: 'gpt-5-2025-08-07',
        agent_type: agent.agent_type,
        concept_number: conceptNumber
      }
    };
  } catch (parseError) {
    console.error('Failed to parse AI response:', content);
    throw new Error('Invalid JSON response from AI');
  }
}

async function storeConcepts(campaignId: string, concepts: any[]) {
  const storedConcepts = [];
  
  for (const concept of concepts) {
    const { data, error } = await supabase
      .from('ai_concepts')
      .insert([{
        campaign_id: campaignId,
        ...concept
      }])
      .select()
      .single();

    if (!error && data) {
      storedConcepts.push(data);
    }
  }

  return storedConcepts;
}