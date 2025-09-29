-- Update RLS policies to work without authentication for public landing pages
-- Make landing pages publicly accessible
DROP POLICY IF EXISTS "Users can view their own landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Users can create their own landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Users can update their own landing pages" ON public.landing_pages;
DROP POLICY IF EXISTS "Users can delete their own landing pages" ON public.landing_pages;

-- Create new policies for public access
CREATE POLICY "Anyone can view published landing pages" 
ON public.landing_pages 
FOR SELECT 
USING (status = 'published');

CREATE POLICY "Anyone can create landing pages" 
ON public.landing_pages 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Anyone can update landing pages" 
ON public.landing_pages 
FOR UPDATE 
USING (true);

CREATE POLICY "Anyone can delete landing pages" 
ON public.landing_pages 
FOR DELETE 
USING (true);

-- Update A/B tests policies for public access
DROP POLICY IF EXISTS "Users can view their own A/B tests" ON public.ab_tests;
DROP POLICY IF EXISTS "Users can create their own A/B tests" ON public.ab_tests;
DROP POLICY IF EXISTS "Users can update their own A/B tests" ON public.ab_tests;
DROP POLICY IF EXISTS "Users can delete their own A/B tests" ON public.ab_tests;

CREATE POLICY "Anyone can manage A/B tests" 
ON public.ab_tests 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Update test variants policies
DROP POLICY IF EXISTS "Users can view variants of their tests" ON public.test_variants;
DROP POLICY IF EXISTS "Users can create variants for their tests" ON public.test_variants;
DROP POLICY IF EXISTS "Users can update variants of their tests" ON public.test_variants;
DROP POLICY IF EXISTS "Users can delete variants of their tests" ON public.test_variants;

CREATE POLICY "Anyone can manage test variants" 
ON public.test_variants 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Update integrations policies
DROP POLICY IF EXISTS "Users can view their own integrations" ON public.integrations;
DROP POLICY IF EXISTS "Users can create their own integrations" ON public.integrations;
DROP POLICY IF EXISTS "Users can update their own integrations" ON public.integrations;
DROP POLICY IF EXISTS "Users can delete their own integrations" ON public.integrations;

CREATE POLICY "Anyone can manage integrations" 
ON public.integrations 
FOR ALL 
USING (true) 
WITH CHECK (true);

-- Make user_id nullable since we're not using auth
ALTER TABLE public.landing_pages ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE public.ab_tests ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE public.integrations ALTER COLUMN user_id DROP NOT NULL;