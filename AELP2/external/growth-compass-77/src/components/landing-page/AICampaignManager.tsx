import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, 
  Lightbulb, 
  Cpu, 
  Zap,
  Play,
  Pause,
  Settings,
  TrendingUp,
  Users,
  Sparkles,
  Code,
  Eye,
  CheckCircle,
  Clock,
  Target,
  Rocket
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface AICampaign {
  id: string;
  name: string;
  description: string;
  target_audience: string;
  business_goals: string;
  industry: string;
  tone_of_voice: string;
  status: 'brainstorming' | 'generating' | 'review' | 'active' | 'paused' | 'completed';
  created_at: string;
}

interface AIConcept {
  id: string;
  campaign_id: string;
  concept_name: string;
  concept_description: string;
  target_problem: string;
  solution_approach: string;
  key_benefits: string[];
  psychological_triggers: string[];
  components_needed: string[];
  estimated_conversion_rate: number;
  creativity_score: number;
  feasibility_score: number;
  status: 'draft' | 'reviewed' | 'approved' | 'rejected' | 'implemented';
  created_at: string;
}

interface AIAgent {
  id: string;
  name: string;
  agent_type: string;
  personality: string;
  expertise: string;
  is_active: boolean;
}

export function AICampaignManager() {
  const [campaigns, setCampaigns] = useState<AICampaign[]>([]);
  const [selectedCampaign, setSelectedCampaign] = useState<AICampaign | null>(null);
  const [concepts, setConcepts] = useState<AIConcept[]>([]);
  const [agents, setAgents] = useState<AIAgent[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [isBrainstorming, setIsBrainstorming] = useState(false);
  const [isGeneratingComponents, setIsGeneratingComponents] = useState(false);
  const [newCampaign, setNewCampaign] = useState({
    name: '',
    description: '',
    target_audience: '',
    business_goals: '',
    industry: '',
    tone_of_voice: 'professional'
  });
  const [briefing, setBriefing] = useState({
    campaign_type: '',
    budget_range: '',
    timeline: '',
    special_requirements: '',
    inspiration: ''
  });
  const { toast } = useToast();

  useEffect(() => {
    loadCampaigns();
  }, []);

  useEffect(() => {
    if (selectedCampaign) {
      loadCampaignData(selectedCampaign.id);
    }
  }, [selectedCampaign]);

  const loadCampaigns = async () => {
    try {
      const { data, error } = await supabase
        .from('ai_campaigns')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setCampaigns((data || []).map(campaign => ({
        ...campaign,
        status: campaign.status as 'brainstorming' | 'generating' | 'review' | 'active' | 'paused' | 'completed'
      })));
    } catch (error) {
      console.error('Error loading campaigns:', error);
    }
  };

  const loadCampaignData = async (campaignId: string) => {
    try {
      // Load concepts
      const { data: conceptsData, error: conceptsError } = await supabase
        .from('ai_concepts')
        .select('*')
        .eq('campaign_id', campaignId)
        .order('creativity_score', { ascending: false });

      if (conceptsError) throw conceptsError;
      setConcepts((conceptsData || []).map(concept => ({
        ...concept,
        status: concept.status as 'draft' | 'reviewed' | 'approved' | 'rejected' | 'implemented'
      })));

      // Load agents
      const { data: agentsData, error: agentsError } = await supabase
        .from('ai_agents')
        .select('*')
        .eq('campaign_id', campaignId)
        .order('created_at', { ascending: false });

      if (agentsError) throw agentsError;
      setAgents(agentsData || []);
    } catch (error) {
      console.error('Error loading campaign data:', error);
    }
  };

  const createCampaign = async () => {
    try {
      const { data, error } = await supabase
        .from('ai_campaigns')
        .insert([newCampaign])
        .select()
        .single();

      if (error) throw error;

      toast({
        title: "Campaign created!",
        description: `AI Campaign "${newCampaign.name}" has been created. Ready for AI brainstorming.`
      });

      setIsCreating(false);
      setNewCampaign({
        name: '',
        description: '',
        target_audience: '',
        business_goals: '',
        industry: '',
        tone_of_voice: 'professional'
      });
      loadCampaigns();
      setSelectedCampaign({
        ...data,
        status: data.status as 'brainstorming' | 'generating' | 'review' | 'active' | 'paused' | 'completed'
      });
    } catch (error) {
      console.error('Error creating campaign:', error);
      toast({
        title: "Error creating campaign",
        description: "Please try again.",
        variant: "destructive"
      });
    }
  };

  const startAIBrainstorming = async () => {
    if (!selectedCampaign) return;

    setIsBrainstorming(true);
    
    try {
      const { data, error } = await supabase.functions.invoke('ai-brainstorm', {
        body: {
          campaignId: selectedCampaign.id,
          briefing: briefing
        }
      });

      if (error) throw error;

      toast({
        title: "AI Brainstorming Complete!",
        description: `Generated ${data.concepts?.length} innovative landing page concepts.`
      });

      // Update campaign status
      await supabase
        .from('ai_campaigns')
        .update({ status: 'review' })
        .eq('id', selectedCampaign.id);

      loadCampaignData(selectedCampaign.id);
    } catch (error) {
      console.error('Error in AI brainstorming:', error);
      toast({
        title: "AI Brainstorming failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsBrainstorming(false);
    }
  };

  const generateComponents = async (conceptId: string) => {
    setIsGeneratingComponents(true);
    
    try {
      const { data, error } = await supabase.functions.invoke('ai-generate-components', {
        body: { conceptId }
      });

      if (error) throw error;

      toast({
        title: "Components Generated!",
        description: `AI generated ${data.generated_components?.length} functional components.`
      });

      loadCampaignData(selectedCampaign!.id);
    } catch (error) {
      console.error('Error generating components:', error);
      toast({
        title: "Component generation failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsGeneratingComponents(false);
    }
  };

  const approveConcept = async (conceptId: string) => {
    try {
      await supabase
        .from('ai_concepts')
        .update({ status: 'approved' })
        .eq('id', conceptId);

      toast({
        title: "Concept approved!",
        description: "Ready for component generation."
      });

      loadCampaignData(selectedCampaign!.id);
    } catch (error) {
      console.error('Error approving concept:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'brainstorming': return 'bg-blue-500';
      case 'generating': return 'bg-purple-500';
      case 'review': return 'bg-yellow-500';
      case 'active': return 'bg-green-500';
      case 'approved': return 'bg-green-500';
      case 'implemented': return 'bg-emerald-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            AI Campaign Manager
          </h1>
          <p className="text-muted-foreground">Let AI brainstorm, design, and code your landing pages</p>
        </div>
        <Button onClick={() => setIsCreating(true)}>
          <Sparkles className="h-4 w-4 mr-2" />
          New AI Campaign
        </Button>
      </div>

      {isCreating && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Create AI-Powered Campaign
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="campaign-name">Campaign Name</Label>
                <Input
                  id="campaign-name"
                  value={newCampaign.name}
                  onChange={(e) => setNewCampaign(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Q1 Growth Campaign"
                />
              </div>
              <div>
                <Label htmlFor="industry">Industry</Label>
                <Select
                  value={newCampaign.industry}
                  onValueChange={(value) => setNewCampaign(prev => ({ ...prev, industry: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select industry" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="saas">SaaS</SelectItem>
                    <SelectItem value="ecommerce">E-commerce</SelectItem>
                    <SelectItem value="wellness">Health & Wellness</SelectItem>
                    <SelectItem value="fintech">FinTech</SelectItem>
                    <SelectItem value="education">Education</SelectItem>
                    <SelectItem value="consulting">Consulting</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="col-span-2">
                <Label htmlFor="description">Campaign Description</Label>
                <Textarea
                  id="description"
                  value={newCampaign.description}
                  onChange={(e) => setNewCampaign(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe your campaign goals, challenges, and what success looks like..."
                />
              </div>
              <div>
                <Label htmlFor="target-audience">Target Audience</Label>
                <Input
                  id="target-audience"
                  value={newCampaign.target_audience}
                  onChange={(e) => setNewCampaign(prev => ({ ...prev, target_audience: e.target.value }))}
                  placeholder="B2B decision makers, age 30-50"
                />
              </div>
              <div>
                <Label htmlFor="business-goals">Business Goals</Label>
                <Input
                  id="business-goals"
                  value={newCampaign.business_goals}
                  onChange={(e) => setNewCampaign(prev => ({ ...prev, business_goals: e.target.value }))}
                  placeholder="Increase trial signups by 40%"
                />
              </div>
              <div className="col-span-2 flex gap-2 justify-end">
                <Button variant="outline" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
                <Button onClick={createCampaign}>
                  <Brain className="h-4 w-4 mr-2" />
                  Create Campaign
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Campaign List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Active Campaigns</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {campaigns.map((campaign) => (
                <div
                  key={campaign.id}
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedCampaign?.id === campaign.id 
                      ? 'bg-primary/10 border-primary' 
                      : 'hover:bg-muted/50'
                  }`}
                  onClick={() => setSelectedCampaign(campaign)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-sm">{campaign.name}</h4>
                    <Badge 
                      variant="outline" 
                      className={`text-xs ${getStatusColor(campaign.status)} text-white border-0`}
                    >
                      {campaign.status}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{campaign.industry}</p>
                  <p className="text-xs text-muted-foreground">{campaign.target_audience}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Campaign Details */}
        <div className="lg:col-span-2">
          {selectedCampaign ? (
            <Tabs defaultValue="brainstorm" className="space-y-4">
              <TabsList className="grid grid-cols-4 w-full">
                <TabsTrigger value="brainstorm">🧠 Brainstorm</TabsTrigger>
                <TabsTrigger value="concepts">💡 Concepts</TabsTrigger>
                <TabsTrigger value="agents">🤖 AI Agents</TabsTrigger>
                <TabsTrigger value="deploy">🚀 Deploy</TabsTrigger>
              </TabsList>

              <TabsContent value="brainstorm" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Lightbulb className="h-5 w-5" />
                      AI Brainstorming Session
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Campaign Type</Label>
                        <Select
                          value={briefing.campaign_type}
                          onValueChange={(value) => setBriefing(prev => ({ ...prev, campaign_type: value }))}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="lead_generation">Lead Generation</SelectItem>
                            <SelectItem value="product_launch">Product Launch</SelectItem>
                            <SelectItem value="trial_signup">Trial Signup</SelectItem>
                            <SelectItem value="webinar_registration">Webinar Registration</SelectItem>
                            <SelectItem value="consultation_booking">Consultation Booking</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>Budget Range</Label>
                        <Select
                          value={briefing.budget_range}
                          onValueChange={(value) => setBriefing(prev => ({ ...prev, budget_range: value }))}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select budget" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="5k-10k">$5k - $10k</SelectItem>
                            <SelectItem value="10k-25k">$10k - $25k</SelectItem>
                            <SelectItem value="25k-50k">$25k - $50k</SelectItem>
                            <SelectItem value="50k+">$50k+</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div>
                      <Label>Special Requirements</Label>
                      <Textarea
                        value={briefing.special_requirements}
                        onChange={(e) => setBriefing(prev => ({ ...prev, special_requirements: e.target.value }))}
                        placeholder="Any specific features, integrations, or constraints..."
                      />
                    </div>
                    <Button 
                      onClick={startAIBrainstorming} 
                      disabled={isBrainstorming}
                      className="w-full"
                    >
                      {isBrainstorming ? (
                        <>
                          <Cpu className="h-4 w-4 mr-2 animate-spin" />
                          AI is brainstorming...
                        </>
                      ) : (
                        <>
                          <Brain className="h-4 w-4 mr-2" />
                          Start AI Brainstorming
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="concepts" className="space-y-4">
                {concepts.map((concept) => (
                  <Card key={concept.id}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{concept.concept_name}</CardTitle>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{concept.status}</Badge>
                          <div className="flex items-center gap-1 text-sm">
                            <TrendingUp className="h-4 w-4 text-green-600" />
                            {concept.estimated_conversion_rate}%
                          </div>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-4">{concept.concept_description}</p>
                      
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <span className="text-xs font-medium">Creativity Score</span>
                          <Progress value={concept.creativity_score} className="h-2 mt-1" />
                        </div>
                        <div>
                          <span className="text-xs font-medium">Feasibility Score</span>
                          <Progress value={concept.feasibility_score} className="h-2 mt-1" />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <span className="text-xs font-medium text-muted-foreground">Key Benefits</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {concept.key_benefits?.map((benefit, i) => (
                              <Badge key={i} variant="secondary" className="text-xs">{benefit}</Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <span className="text-xs font-medium text-muted-foreground">Psychological Triggers</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {concept.psychological_triggers?.map((trigger, i) => (
                              <Badge key={i} variant="outline" className="text-xs">{trigger}</Badge>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        {concept.status === 'draft' && (
                          <Button 
                            size="sm" 
                            onClick={() => approveConcept(concept.id)}
                          >
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Approve
                          </Button>
                        )}
                        {concept.status === 'approved' && (
                          <Button 
                            size="sm" 
                            onClick={() => generateComponents(concept.id)}
                            disabled={isGeneratingComponents}
                          >
                            {isGeneratingComponents ? (
                              <>
                                <Cpu className="h-4 w-4 mr-1 animate-spin" />
                                Generating...
                              </>
                            ) : (
                              <>
                                <Code className="h-4 w-4 mr-1" />
                                Generate Components
                              </>
                            )}
                          </Button>
                        )}
                        <Button size="sm" variant="outline">
                          <Eye className="h-4 w-4 mr-1" />
                          Preview
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              <TabsContent value="agents" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {agents.map((agent) => (
                    <Card key={agent.id}>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm">{agent.name}</CardTitle>
                          <Badge variant={agent.is_active ? 'default' : 'secondary'}>
                            {agent.is_active ? 'Active' : 'Inactive'}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p className="text-xs text-muted-foreground mb-2">{agent.personality}</p>
                        <p className="text-xs font-medium">Expertise: {agent.expertise}</p>
                        <Badge variant="outline" className="mt-2 text-xs">
                          {agent.agent_type.replace('_', ' ')}
                        </Badge>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>

              <TabsContent value="deploy" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Rocket className="h-5 w-5" />
                      Deploy to Your System
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Connect approved concepts to your external system for implementation
                    </p>
                    <Button disabled>
                      <Settings className="h-4 w-4 mr-2" />
                      Configure External System Integration
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="text-center py-12">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium">Select a Campaign</h3>
                <p className="text-muted-foreground">
                  Choose a campaign to start AI brainstorming and component generation
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}