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
    const { conceptId } = await req.json();

    console.log('AI Component Generation request:', { conceptId });

    // Get concept details with campaign and agent info
    const { data: concept, error: conceptError } = await supabase
      .from('ai_concepts')
      .select(`
        *,
        ai_campaigns (*),
        ai_agents (*)
      `)
      .eq('id', conceptId)
      .single();

    if (conceptError || !concept) {
      throw new Error('Concept not found');
    }

    // Get the component generator agent
    const { data: componentAgent, error: agentError } = await supabase
      .from('ai_agents')
      .select('*')
      .eq('campaign_id', concept.campaign_id)
      .eq('agent_type', 'component_generator')
      .single();

    if (agentError || !componentAgent) {
      throw new Error('Component generator agent not found');
    }

    // Generate functional components for this concept
    const components = await generateFunctionalComponents(concept, componentAgent);

    // Store generated components
    const storedComponents = await storeComponents(conceptId, componentAgent.id, components);

    return new Response(
      JSON.stringify({
        success: true,
        concept: concept,
        generated_components: storedComponents,
        summary: {
          totalComponents: components.length,
          complexityAvg: components.reduce((acc, c) => acc + c.implementation_complexity, 0) / components.length,
          readyForCoding: components.filter(c => c.status === 'generated').length
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Error in ai-generate-components:', error);
    return new Response(
      JSON.stringify({ success: false, error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

async function generateFunctionalComponents(concept: any, agent: any) {
  const components = [];
  
  // Generate components based on the concept's needs
  for (const componentType of concept.components_needed) {
    const component = await generateSingleComponent(concept, componentType, agent);
    components.push(component);
  }

  // Add some standard smart components
  const smartComponents = [
    'AI-Powered Social Proof Widget',
    'Dynamic Pricing Calculator',
    'Intelligent Lead Qualifier',
    'Behavioral Analytics Tracker'
  ];

  for (const smartType of smartComponents) {
    const component = await generateSingleComponent(concept, smartType, agent);
    components.push(component);
  }

  return components;
}

async function generateSingleComponent(concept: any, componentType: string, agent: any) {
  const prompt = `${agent.prompt_template}

CONCEPT CONTEXT:
Name: ${concept.concept_name}
Description: ${concept.concept_description}
Target Problem: ${concept.target_problem}
Solution Approach: ${concept.solution_approach}
Key Benefits: ${concept.key_benefits?.join(', ')}
Psychological Triggers: ${concept.psychological_triggers?.join(', ')}
Target Audience: ${concept.ai_campaigns?.target_audience}
Industry: ${concept.ai_campaigns?.industry}

COMPONENT TO GENERATE: ${componentType}

TASK: Design a functional React component that serves the concept's goals and maximizes conversions.

Respond with a JSON object containing:
{
  "component_type": "${componentType}",
  "component_name": "Specific component name",
  "description": "What this component does and why it's effective",
  "functionality": "Detailed functionality description",
  "code_structure": {
    "props": ["prop1: string", "prop2: number"],
    "state": ["state1", "state2"],
    "hooks": ["useState", "useEffect", "useCallback"],
    "external_apis": ["api1", "api2"],
    "events": ["onClick", "onSubmit", "onScroll"]
  },
  "styling_requirements": "Specific styling needs for maximum conversion",
  "api_endpoints": ["/api/endpoint1", "/api/endpoint2"],
  "dependencies": ["react", "lucide-react", "@supabase/supabase-js"],
  "implementation_complexity": 3,
  "conversion_impact": "How this component specifically increases conversions",
  "user_psychology": "How this leverages user psychology",
  "mobile_considerations": "Mobile-specific optimizations",
  "a_b_test_variants": ["variant1", "variant2", "variant3"]
}

Make this component SMART and CONVERSION-FOCUSED. Think beyond basic UI elements.`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-5-2025-08-07',
      messages: [
        { role: 'system', content: 'You are a senior frontend architect specializing in conversion optimization. Respond only with valid JSON.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_completion_tokens: 2000
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status}`);
  }

  const data = await response.json();
  const content = data.choices[0].message.content;

  try {
    const component = JSON.parse(content);
    
    // Generate actual React code for the component
    const generatedCode = await generateComponentCode(component, concept);
    
    return {
      ...component,
      generated_code: generatedCode,
      metadata: {
        generated_at: new Date().toISOString(),
        model_used: 'gpt-5-2025-08-07',
        concept_id: concept.id,
        agent_id: agent.id
      }
    };
  } catch (parseError) {
    console.error('Failed to parse AI component response:', content);
    throw new Error('Invalid JSON response from AI');
  }
}

async function generateComponentCode(component: any, concept: any) {
  const codePrompt = `Generate production-ready React TypeScript code for this component:

Component: ${component.component_name}
Description: ${component.description}
Functionality: ${component.functionality}

Requirements:
- Use TypeScript
- Include proper prop types
- Use Tailwind CSS for styling
- Include conversion optimization features
- Add analytics tracking
- Make it mobile-responsive
- Include accessibility features
- Use shadcn/ui components where appropriate

Code Structure: ${JSON.stringify(component.code_structure, null, 2)}

Generate complete, working code that can be copy-pasted into a React project.`;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-5-2025-08-07',
      messages: [
        { role: 'system', content: 'You are a senior React developer. Generate only clean, production-ready TypeScript React code.' },
        { role: 'user', content: codePrompt }
      ],
      temperature: 0.3,
      max_completion_tokens: 3000
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

async function storeComponents(conceptId: string, agentId: string, components: any[]) {
  const storedComponents = [];
  
  for (const component of components) {
    const { data, error } = await supabase
      .from('ai_generated_components')
      .insert([{
        concept_id: conceptId,
        agent_id: agentId,
        ...component
      }])
      .select()
      .single();

    if (!error && data) {
      storedComponents.push(data);
    }
  }

  return storedComponents;
}