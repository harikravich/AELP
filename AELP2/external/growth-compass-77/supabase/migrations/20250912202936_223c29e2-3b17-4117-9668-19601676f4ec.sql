-- Fix RLS policies for public access without authentication
-- First, let's check current policies and fix them

-- Drop all existing policies to start fresh
DROP POLICY IF EXISTS "Anyone can view published landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Anyone can create landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Anyone can update landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Anyone can delete landing pages" ON public.landing_pages;

-- Create simple public access policies
CREATE POLICY "Public can view all landing pages" 
ON public.landing_pages 
FOR SELECT 
USING (true);

CREATE POLICY "Public can insert landing pages" 
ON public.landing_pages 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Public can update landing pages" 
ON public.landing_pages 
FOR UPDATE 
USING (true);

CREATE POLICY "Public can delete landing pages" 
ON public.landing_pages 
FOR DELETE 
USING (true);

-- Fix A/B tests policies
DROP POLICY IF EXISTS "Anyone can manage A/B tests" ON public.ab_tests;

CREATE POLICY "Public access to ab_tests" 
ON public.ab_tests 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Fix test variants policies  
DROP POLICY IF EXISTS "Anyone can manage test variants" ON public.test_variants;

CREATE POLICY "Public access to test_variants" 
ON public.test_variants 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Fix integrations policies
DROP POLICY IF EXISTS "Anyone can manage integrations" ON public.integrations;

CREATE POLICY "Public access to integrations" 
ON public.integrations 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Make sure analytics events can be inserted publicly
CREATE POLICY "Public can insert analytics events" 
ON public.analytics_events 
FOR INSERT 
WITH CHECK (true);

-- Update analytics events select policy to be more permissive
DROP POLICY IF EXISTS "Users can view analytics for their pages" ON public.analytics_events;

CREATE POLICY "Public can view analytics events" 
ON public.analytics_events 
FOR SELECT 
USING (true);