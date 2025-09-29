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
  Play, 
  Pause, 
  Trophy, 
  TrendingUp, 
  Users, 
  Target,
  BarChart3,
  Split
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface ABTest {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'running' | 'paused' | 'completed';
  landing_page_id: string;
  variants: TestVariant[];
  conversion_goal: string;
  confidence_level: number;
  statistical_power: number;
  started_at?: string;
  ended_at?: string;
  winner_variant_id?: string;
}

interface TestVariant {
  id: string;
  name: string;
  description: string;
  traffic_percentage: number;
  visitors: number;
  conversions: number;
  conversion_rate: number;
  is_control: boolean;
  config: any;
}

export function ABTestManager() {
  const [tests, setTests] = useState<ABTest[]>([]);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newTest, setNewTest] = useState({
    name: '',
    description: '',
    landing_page_id: '',
    conversion_goal: 'form_submission',
    confidence_level: 0.95,
    statistical_power: 0.80
  });
  const { toast } = useToast();

  useEffect(() => {
    loadTests();
  }, []);

  const loadTests = async () => {
    try {
      const { data, error } = await supabase
        .from('ab_tests')
        .select(`
          *,
          test_variants (*)
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setTests((data || []).map(test => ({
        ...test,
        status: test.status as 'draft' | 'running' | 'paused' | 'completed',
        variants: test.test_variants || []
      })));
    } catch (error) {
      console.error('Error loading A/B tests:', error);
    }
  };

  const createTest = async () => {
    try {
      const { data: testData, error: testError } = await supabase
        .from('ab_tests')
        .insert([{
          ...newTest,
          status: 'draft',
          variants: []
        }])
        .select()
        .single();

      if (testError) throw testError;

      // Create default variants
      const variants = [
        {
          ab_test_id: testData.id,
          name: 'Control (A)',
          description: 'Original version',
          traffic_percentage: 50,
          is_control: true,
          config: {}
        },
        {
          ab_test_id: testData.id,
          name: 'Variant B',
          description: 'Test version',
          traffic_percentage: 50,
          is_control: false,
          config: {}
        }
      ];

      const { error: variantsError } = await supabase
        .from('test_variants')
        .insert(variants);

      if (variantsError) throw variantsError;

      toast({
        title: "A/B test created!",
        description: `Test "${newTest.name}" has been created with default variants.`
      });

      setIsCreating(false);
      setNewTest({
        name: '',
        description: '',
        landing_page_id: '',
        conversion_goal: 'form_submission',
        confidence_level: 0.95,
        statistical_power: 0.80
      });
      loadTests();
    } catch (error) {
      console.error('Error creating test:', error);
      toast({
        title: "Error creating test",
        description: "Please try again.",
        variant: "destructive"
      });
    }
  };

  const startTest = async (testId: string) => {
    try {
      const { error } = await supabase
        .from('ab_tests')
        .update({
          status: 'running',
          started_at: new Date().toISOString()
        })
        .eq('id', testId);

      if (error) throw error;

      toast({
        title: "Test started!",
        description: "Your A/B test is now running."
      });

      loadTests();
    } catch (error) {
      console.error('Error starting test:', error);
    }
  };

  const pauseTest = async (testId: string) => {
    try {
      const { error } = await supabase
        .from('ab_tests')
        .update({ status: 'paused' })
        .eq('id', testId);

      if (error) throw error;

      toast({
        title: "Test paused",
        description: "Your A/B test has been paused."
      });

      loadTests();
    } catch (error) {
      console.error('Error pausing test:', error);
    }
  };

  const declareWinner = async (testId: string, winnerVariantId: string) => {
    try {
      const { error } = await supabase
        .from('ab_tests')
        .update({
          status: 'completed',
          ended_at: new Date().toISOString(),
          winner_variant_id: winnerVariantId
        })
        .eq('id', testId);

      if (error) throw error;

      toast({
        title: "Winner declared!",
        description: "Test completed and winner has been selected."
      });

      loadTests();
    } catch (error) {
      console.error('Error declaring winner:', error);
    }
  };

  const calculateLift = (variantRate: number, controlRate: number) => {
    if (controlRate === 0) return 0;
    return ((variantRate - controlRate) / controlRate) * 100;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 95) return 'text-green-600';
    if (confidence >= 90) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">A/B Test Manager</h1>
          <p className="text-muted-foreground">Optimize your landing pages with data-driven testing</p>
        </div>
        <Button onClick={() => setIsCreating(true)}>
          <Split className="h-4 w-4 mr-2" />
          Create Test
        </Button>
      </div>

      {isCreating && (
        <Card>
          <CardHeader>
            <CardTitle>Create New A/B Test</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="test-name">Test Name</Label>
                <Input
                  id="test-name"
                  value={newTest.name}
                  onChange={(e) => setNewTest(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Hero Section Test"
                />
              </div>
              <div>
                <Label htmlFor="conversion-goal">Conversion Goal</Label>
                <Select
                  value={newTest.conversion_goal}
                  onValueChange={(value) => setNewTest(prev => ({ ...prev, conversion_goal: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="form_submission">Form Submission</SelectItem>
                    <SelectItem value="button_click">Button Click</SelectItem>
                    <SelectItem value="page_scroll">Page Scroll (50%)</SelectItem>
                    <SelectItem value="time_on_page">Time on Page (2min+)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="col-span-2">
                <Label htmlFor="test-description">Description</Label>
                <Textarea
                  id="test-description"
                  value={newTest.description}
                  onChange={(e) => setNewTest(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Testing different headlines to improve conversion rate"
                />
              </div>
              <div className="col-span-2 flex gap-2 justify-end">
                <Button variant="outline" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
                <Button onClick={createTest}>
                  Create Test
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6">
        {tests.map((test) => {
          const controlVariant = test.variants?.find(v => v.is_control);
          const testVariants = test.variants?.filter(v => !v.is_control) || [];
          const bestVariant = test.variants?.reduce((best, current) => 
            current.conversion_rate > (best?.conversion_rate || 0) ? current : best
          );

          return (
            <Card key={test.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      {test.name}
                      <Badge variant={
                        test.status === 'running' ? 'default' :
                        test.status === 'completed' ? 'secondary' :
                        test.status === 'paused' ? 'outline' : 'secondary'
                      }>
                        {test.status}
                      </Badge>
                    </CardTitle>
                    <p className="text-sm text-muted-foreground">{test.description}</p>
                  </div>
                  <div className="flex gap-2">
                    {test.status === 'draft' && (
                      <Button size="sm" onClick={() => startTest(test.id)}>
                        <Play className="h-4 w-4 mr-1" />
                        Start
                      </Button>
                    )}
                    {test.status === 'running' && (
                      <Button size="sm" variant="outline" onClick={() => pauseTest(test.id)}>
                        <Pause className="h-4 w-4 mr-1" />
                        Pause
                      </Button>
                    )}
                    {test.status === 'running' && bestVariant && (
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => declareWinner(test.id, bestVariant.id)}
                      >
                        <Trophy className="h-4 w-4 mr-1" />
                        Declare Winner
                      </Button>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="overview" className="w-full">
                  <TabsList>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="variants">Variants</TabsTrigger>
                    <TabsTrigger value="analytics">Analytics</TabsTrigger>
                  </TabsList>

                  <TabsContent value="overview" className="space-y-4">
                    <div className="grid grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold">{test.variants?.reduce((sum, v) => sum + v.visitors, 0) || 0}</div>
                        <div className="text-sm text-muted-foreground">Total Visitors</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold">{test.variants?.reduce((sum, v) => sum + v.conversions, 0) || 0}</div>
                        <div className="text-sm text-muted-foreground">Total Conversions</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold">
                          {controlVariant ? `${(controlVariant.conversion_rate * 100).toFixed(1)}%` : '0%'}
                        </div>
                        <div className="text-sm text-muted-foreground">Control Rate</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold">
                          {bestVariant && controlVariant ? 
                            `+${calculateLift(bestVariant.conversion_rate, controlVariant.conversion_rate).toFixed(1)}%` : 
                            '0%'
                          }
                        </div>
                        <div className="text-sm text-muted-foreground">Best Lift</div>
                      </div>
                    </div>

                    {test.status === 'running' && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Statistical Confidence</span>
                          <span className={getConfidenceColor(85)}>85%</span>
                        </div>
                        <Progress value={85} className="h-2" />
                        <p className="text-xs text-muted-foreground">
                          Need 95% confidence to declare a winner
                        </p>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="variants" className="space-y-4">
                    <div className="grid gap-4">
                      {test.variants?.map((variant) => (
                        <div key={variant.id} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{variant.name}</h4>
                              {variant.is_control && <Badge variant="outline">Control</Badge>}
                              {test.winner_variant_id === variant.id && (
                                <Badge variant="default">Winner</Badge>
                              )}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {variant.traffic_percentage}% traffic
                            </div>
                          </div>
                          <p className="text-sm text-muted-foreground mb-3">{variant.description}</p>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <div className="font-medium">{variant.visitors}</div>
                              <div className="text-muted-foreground">Visitors</div>
                            </div>
                            <div>
                              <div className="font-medium">{variant.conversions}</div>
                              <div className="text-muted-foreground">Conversions</div>
                            </div>
                            <div>
                              <div className="font-medium">{(variant.conversion_rate * 100).toFixed(2)}%</div>
                              <div className="text-muted-foreground">Conv. Rate</div>
                            </div>
                          </div>
                          {!variant.is_control && controlVariant && (
                            <div className="mt-2 text-sm">
                              <span className={`font-medium ${
                                variant.conversion_rate > controlVariant.conversion_rate ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {variant.conversion_rate > controlVariant.conversion_rate ? '+' : ''}
                                {calculateLift(variant.conversion_rate, controlVariant.conversion_rate).toFixed(1)}% lift
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="analytics" className="space-y-4">
                    <div className="text-center py-8 text-muted-foreground">
                      <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Detailed analytics coming soon</p>
                      <p className="text-sm">Real-time conversion tracking and statistical analysis</p>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          );
        })}

        {tests.length === 0 && (
          <Card>
            <CardContent className="text-center py-12">
              <Split className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">No A/B tests yet</h3>
              <p className="text-muted-foreground mb-4">
                Create your first A/B test to start optimizing your landing pages
              </p>
              <Button onClick={() => setIsCreating(true)}>
                Create Your First Test
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}