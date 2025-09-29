import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  TestTube, 
  Plus, 
  Play,
  Pause,
  CheckCircle,
  Target,
  TrendingUp,
  BarChart3,
  ArrowRight
} from "lucide-react";

export default function Experiments() {
  const activeExperiments = [
    {
      id: "exp_001",
      name: "Q4 Headline Urgency Test",
      description: "Testing urgency-driven headlines vs. benefit-focused messaging for B2B SaaS trial conversion",
      status: "running",
      type: "Creative",
      visitors: "12,847",
      conversionLift: "+23.4%",
      confidence: "94%",
      timeRemaining: "3 days",
      totalSpend: "$34,290",
      cacImprovement: "-$47"
    },
    {
      id: "exp_002", 
      name: "Social Proof Video Testing",
      description: "Customer testimonial video vs. product demo for display campaign engagement",
      status: "running",
      type: "Video",
      visitors: "8,934",
      conversionLift: "+15.7%",
      confidence: "87%",
      timeRemaining: "1 week",
      totalSpend: "$19,430",
      cacImprovement: "-$23"
    },
    {
      id: "exp_003",
      name: "Mobile CTA Button Optimization",
      description: "Testing button size, color, and copy for mobile conversion optimization",
      status: "completed",
      type: "Design",
      visitors: "15,621",
      conversionLift: "+31.2%",
      confidence: "96%",
      timeRemaining: "Complete",
      totalSpend: "$41,230",
      cacImprovement: "-$62"
    }
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Experiments Hub</h1>
            <p className="text-muted-foreground mt-1">
              A/B testing dashboard • Results auto-feed MMM & RL training systems
            </p>
          </div>
          <Button className="bg-primary">
            <TestTube className="w-4 h-4 mr-2" />
            New Experiment
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-6">
          {activeExperiments.map((exp) => (
            <Card key={exp.id} className="border shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <Badge variant={exp.status === 'running' ? 'default' : 'secondary'}>
                        {exp.status}
                      </Badge>
                      <Badge variant="outline">{exp.type}</Badge>
                      <h3 className="font-semibold text-lg">{exp.name}</h3>
                      <span className="text-sm text-muted-foreground">#{exp.id}</span>
                    </div>
                    
                    <p className="text-muted-foreground mb-4">{exp.description}</p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                      <div>
                        <span className="text-xs text-muted-foreground">Visitors</span>
                        <div className="font-semibold">{exp.visitors}</div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">Conversion Lift</span>
                        <div className="font-semibold text-primary">{exp.conversionLift}</div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">Confidence</span>
                        <div className="font-semibold text-accent">{exp.confidence}</div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">Time Remaining</span>
                        <div className="font-semibold">{exp.timeRemaining}</div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">Total Spend</span>
                        <div className="font-semibold">{exp.totalSpend}</div>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">CAC Improvement</span>
                        <div className="font-semibold text-primary">{exp.cacImprovement}</div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-sm">
                      <ArrowRight className="w-4 h-4 text-accent" />
                      <span className="text-muted-foreground">Data flowing to:</span>
                      <Badge variant="outline" className="text-xs">MMM Model</Badge>
                      <Badge variant="outline" className="text-xs">RL Training</Badge>
                    </div>
                  </div>
                  
                  <div className="flex flex-col gap-2 ml-6">
                    {exp.status === 'running' ? (
                      <>
                        <Button size="sm" variant="outline">
                          <Pause className="w-4 h-4 mr-2" />
                          Pause Test
                        </Button>
                        <Button size="sm" variant="outline">
                          <BarChart3 className="w-4 h-4 mr-2" />
                          Live Results
                        </Button>
                      </>
                    ) : (
                      <Button size="sm" className="bg-primary">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Deploy Winner
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}