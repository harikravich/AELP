-- Create tables for AI campaign management and AI agents

-- AI Campaigns table
CREATE TABLE public.ai_campaigns (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name text NOT NULL,
  description text,
  target_audience text,
  business_goals text,
  industry text,
  tone_of_voice text DEFAULT 'professional',
  brand_guidelines jsonb DEFAULT '{}',
  status text NOT NULL DEFAULT 'brainstorming' CHECK (status IN ('brainstorming', 'generating', 'review', 'active', 'paused', 'completed')),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now()
);

-- AI Agents table
CREATE TABLE public.ai_agents (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  campaign_id uuid REFERENCES public.ai_campaigns(id) ON DELETE CASCADE,
  agent_type text NOT NULL CHECK (agent_type IN ('concept_brainstormer', 'component_generator', 'copy_writer', 'optimizer')),
  name text NOT NULL,
  personality text,
  expertise text,
  prompt_template text,
  config jsonb DEFAULT '{}',
  is_active boolean DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now()
);

-- AI Generated Concepts table
CREATE TABLE public.ai_concepts (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  campaign_id uuid REFERENCES public.ai_campaigns(id) ON DELETE CASCADE,
  agent_id uuid REFERENCES public.ai_agents(id) ON DELETE SET NULL,
  concept_name text NOT NULL,
  concept_description text,
  target_problem text,
  solution_approach text,
  key_benefits text[],
  psychological_triggers text[],
  components_needed text[],
  estimated_conversion_rate numeric,
  creativity_score integer DEFAULT 50 CHECK (creativity_score BETWEEN 0 AND 100),
  feasibility_score integer DEFAULT 50 CHECK (feasibility_score BETWEEN 0 AND 100),
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'reviewed', 'approved', 'rejected', 'implemented')),
  metadata jsonb DEFAULT '{}',
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now()
);

-- AI Generated Components table
CREATE TABLE public.ai_generated_components (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  concept_id uuid REFERENCES public.ai_concepts(id) ON DELETE CASCADE,
  agent_id uuid REFERENCES public.ai_agents(id) ON DELETE SET NULL,
  component_type text NOT NULL,
  component_name text NOT NULL,
  description text,
  functionality text,
  code_structure jsonb,
  styling_requirements text,
  api_endpoints text[],
  dependencies text[],
  implementation_complexity integer DEFAULT 3 CHECK (implementation_complexity BETWEEN 1 AND 5),
  status text NOT NULL DEFAULT 'generated' CHECK (status IN ('generated', 'refined', 'coded', 'tested', 'deployed')),
  generated_code text,
  metadata jsonb DEFAULT '{}',
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now()
);

-- Campaign Analytics table
CREATE TABLE public.campaign_analytics (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  campaign_id uuid REFERENCES public.ai_campaigns(id) ON DELETE CASCADE,
  concept_id uuid REFERENCES public.ai_concepts(id) ON DELETE SET NULL,
  metric_name text NOT NULL,
  metric_value numeric,
  metric_type text NOT NULL CHECK (metric_type IN ('conversion_rate', 'engagement', 'creativity', 'performance')),
  recorded_at timestamp with time zone NOT NULL DEFAULT now(),
  metadata jsonb DEFAULT '{}'
);

-- Enable RLS on all tables
ALTER TABLE public.ai_campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_concepts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_generated_components ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.campaign_analytics ENABLE ROW LEVEL SECURITY;

-- Create public access policies
CREATE POLICY "Public access to ai_campaigns" ON public.ai_campaigns FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access to ai_agents" ON public.ai_agents FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access to ai_concepts" ON public.ai_concepts FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access to ai_generated_components" ON public.ai_generated_components FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Public access to campaign_analytics" ON public.campaign_analytics FOR ALL USING (true) WITH CHECK (true);

-- Add update triggers
CREATE TRIGGER update_ai_campaigns_updated_at
BEFORE UPDATE ON public.ai_campaigns
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_ai_agents_updated_at
BEFORE UPDATE ON public.ai_agents
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_ai_concepts_updated_at
BEFORE UPDATE ON public.ai_concepts
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_ai_generated_components_updated_at
BEFORE UPDATE ON public.ai_generated_components
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();