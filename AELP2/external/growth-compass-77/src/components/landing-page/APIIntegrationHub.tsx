import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Zap, 
  Instagram, 
  Mail, 
  MessageSquare, 
  Webhook, 
  Database,
  Settings,
  Plus,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface Integration {
  id: string;
  provider: string;
  config: any;
  is_active: boolean;
  created_at: string;
}

const availableIntegrations = [
  {
    id: 'instagram',
    name: 'Instagram API',
    icon: Instagram,
    description: 'Search Instagram profiles and analyze wellness indicators',
    fields: [
      { key: 'access_token', label: 'Access Token', type: 'password', required: true },
      { key: 'client_id', label: 'Client ID', type: 'text', required: true }
    ]
  },
  {
    id: 'email',
    name: 'Email Service',
    icon: Mail,
    description: 'Send automated emails and lead nurturing sequences',
    fields: [
      { key: 'api_key', label: 'API Key', type: 'password', required: true },
      { key: 'from_email', label: 'From Email', type: 'email', required: true },
      { key: 'provider', label: 'Provider', type: 'select', options: ['sendgrid', 'mailgun', 'resend'], required: true }
    ]
  },
  {
    id: 'sms',
    name: 'SMS Service',
    icon: MessageSquare,
    description: 'Send SMS notifications and reminders',
    fields: [
      { key: 'account_sid', label: 'Account SID', type: 'text', required: true },
      { key: 'auth_token', label: 'Auth Token', type: 'password', required: true },
      { key: 'from_number', label: 'From Number', type: 'text', required: true }
    ]
  },
  {
    id: 'webhook',
    name: 'Custom Webhooks',
    icon: Webhook,
    description: 'Connect to any external service via webhooks',
    fields: [
      { key: 'endpoint_url', label: 'Endpoint URL', type: 'url', required: true },
      { key: 'secret_key', label: 'Secret Key', type: 'password', required: false },
      { key: 'headers', label: 'Custom Headers (JSON)', type: 'textarea', required: false }
    ]
  },
  {
    id: 'analytics',
    name: 'Analytics & Tracking',
    icon: Database,
    description: 'Advanced analytics and user behavior tracking',
    fields: [
      { key: 'google_analytics_id', label: 'Google Analytics ID', type: 'text', required: false },
      { key: 'facebook_pixel_id', label: 'Facebook Pixel ID', type: 'text', required: false },
      { key: 'hotjar_id', label: 'Hotjar Site ID', type: 'text', required: false }
    ]
  }
];

export function APIIntegrationHub() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [selectedIntegration, setSelectedIntegration] = useState<any>(null);
  const [configData, setConfigData] = useState<any>({});
  const [isConfiguring, setIsConfiguring] = useState(false);
  const [testResults, setTestResults] = useState<{ [key: string]: boolean }>({});
  const { toast } = useToast();

  useEffect(() => {
    loadIntegrations();
  }, []);

  const loadIntegrations = async () => {
    try {
      const { data, error } = await supabase
        .from('integrations')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setIntegrations(data || []);
    } catch (error) {
      console.error('Error loading integrations:', error);
    }
  };

  const saveIntegration = async () => {
    try {
      const existingIntegration = integrations.find(i => i.provider === selectedIntegration.id);
      
      if (existingIntegration) {
        const { error } = await supabase
          .from('integrations')
          .update({
            config: configData as any,
            is_active: true
          })
          .eq('id', existingIntegration.id);

        if (error) throw error;
      } else {
        const { error } = await supabase
          .from('integrations')
          .insert([{
            provider: selectedIntegration.id,
            config: configData as any,
            is_active: true
          }]);

        if (error) throw error;
      }

      toast({
        title: "Integration saved!",
        description: `${selectedIntegration.name} has been configured successfully.`
      });

      setIsConfiguring(false);
      setSelectedIntegration(null);
      setConfigData({});
      loadIntegrations();
    } catch (error) {
      console.error('Error saving integration:', error);
      toast({
        title: "Error saving integration",
        description: "Please check your configuration and try again.",
        variant: "destructive"
      });
    }
  };

  const toggleIntegration = async (integrationId: string, isActive: boolean) => {
    try {
      const { error } = await supabase
        .from('integrations')
        .update({ is_active: isActive })
        .eq('id', integrationId);

      if (error) throw error;

      toast({
        title: isActive ? "Integration enabled" : "Integration disabled",
        description: `The integration has been ${isActive ? 'enabled' : 'disabled'}.`
      });

      loadIntegrations();
    } catch (error) {
      console.error('Error toggling integration:', error);
    }
  };

  const testIntegration = async (integration: any) => {
    try {
      // Call the test endpoint for the specific integration
      const { data, error } = await supabase.functions.invoke('test-integration', {
        body: { 
          provider: integration.provider,
          config: integration.config
        }
      });

      if (error) throw error;

      setTestResults(prev => ({ ...prev, [integration.id]: true }));
      
      toast({
        title: "Test successful!",
        description: `${integration.provider} integration is working correctly.`
      });
    } catch (error) {
      setTestResults(prev => ({ ...prev, [integration.id]: false }));
      
      toast({
        title: "Test failed",
        description: "Please check your configuration.",
        variant: "destructive"
      });
    }
  };

  const getIntegrationStatus = (providerId: string) => {
    const integration = integrations.find(i => i.provider === providerId);
    if (!integration) return 'not_configured';
    return integration.is_active ? 'active' : 'inactive';
  };

  const handleConfigChange = (field: any, value: string) => {
    setConfigData(prev => ({
      ...prev,
      [field.key]: value
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Integration Hub</h1>
          <p className="text-muted-foreground">Connect external services to power your landing pages</p>
        </div>
        <Badge variant="outline" className="px-3 py-1">
          <Zap className="h-4 w-4 mr-1" />
          {integrations.filter(i => i.is_active).length} Active
        </Badge>
      </div>

      {isConfiguring && selectedIntegration && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <selectedIntegration.icon className="h-5 w-5" />
              Configure {selectedIntegration.name}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {selectedIntegration.description}
              </p>
              
              {selectedIntegration.fields.map((field: any) => (
                <div key={field.key}>
                  <Label htmlFor={field.key}>
                    {field.label} {field.required && <span className="text-red-500">*</span>}
                  </Label>
                  {field.type === 'textarea' ? (
                    <Textarea
                      id={field.key}
                      value={configData[field.key] || ''}
                      onChange={(e) => handleConfigChange(field, e.target.value)}
                      placeholder={`Enter ${field.label.toLowerCase()}`}
                    />
                  ) : field.type === 'select' ? (
                    <select
                      id={field.key}
                      value={configData[field.key] || ''}
                      onChange={(e) => handleConfigChange(field, e.target.value)}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="">Select {field.label}</option>
                      {field.options?.map((option: string) => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  ) : (
                    <Input
                      id={field.key}
                      type={field.type}
                      value={configData[field.key] || ''}
                      onChange={(e) => handleConfigChange(field, e.target.value)}
                      placeholder={`Enter ${field.label.toLowerCase()}`}
                    />
                  )}
                </div>
              ))}
              
              <div className="flex gap-2 justify-end pt-4">
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setIsConfiguring(false);
                    setSelectedIntegration(null);
                    setConfigData({});
                  }}
                >
                  Cancel
                </Button>
                <Button onClick={saveIntegration}>
                  Save Configuration
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {availableIntegrations.map((integration) => {
          const status = getIntegrationStatus(integration.id);
          const existingIntegration = integrations.find(i => i.provider === integration.id);
          const Icon = integration.icon;

          return (
            <Card key={integration.id} className="relative">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="h-6 w-6" />
                    <CardTitle className="text-lg">{integration.name}</CardTitle>
                  </div>
                  <div className="flex items-center gap-2">
                    {status === 'active' && (
                      <Badge variant="default" className="text-xs">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Active
                      </Badge>
                    )}
                    {status === 'inactive' && (
                      <Badge variant="secondary" className="text-xs">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        Inactive
                      </Badge>
                    )}
                    {status === 'not_configured' && (
                      <Badge variant="outline" className="text-xs">
                        Not Setup
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  {integration.description}
                </p>
                
                <div className="space-y-3">
                  {existingIntegration && (
                    <div className="flex items-center justify-between">
                      <Label htmlFor={`toggle-${integration.id}`} className="text-sm">
                        Enable Integration
                      </Label>
                      <Switch
                        id={`toggle-${integration.id}`}
                        checked={existingIntegration.is_active}
                        onCheckedChange={(checked) => 
                          toggleIntegration(existingIntegration.id, checked)
                        }
                      />
                    </div>
                  )}
                  
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant={status === 'not_configured' ? 'default' : 'outline'}
                      onClick={() => {
                        setSelectedIntegration(integration);
                        setConfigData(existingIntegration?.config || {});
                        setIsConfiguring(true);
                      }}
                      className="flex-1"
                    >
                      <Settings className="h-4 w-4 mr-1" />
                      {status === 'not_configured' ? 'Setup' : 'Configure'}
                    </Button>
                    
                    {existingIntegration && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testIntegration(existingIntegration)}
                      >
                        Test
                      </Button>
                    )}
                  </div>
                  
                  {testResults[existingIntegration?.id] !== undefined && (
                    <div className={`text-xs p-2 rounded ${
                      testResults[existingIntegration?.id] 
                        ? 'bg-green-50 text-green-700' 
                        : 'bg-red-50 text-red-700'
                    }`}>
                      {testResults[existingIntegration?.id] 
                        ? 'Test successful!' 
                        : 'Test failed - check configuration'
                      }
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {integrations.length === 0 && (
        <Card>
          <CardContent className="text-center py-12">
            <Zap className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">No integrations configured</h3>
            <p className="text-muted-foreground mb-4">
              Connect external services to unlock powerful automation features
            </p>
            <Button onClick={() => {
              setSelectedIntegration(availableIntegrations[0]);
              setIsConfiguring(true);
            }}>
              <Plus className="h-4 w-4 mr-2" />
              Setup First Integration
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}