import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Plus, Trash2, Move, Settings, Eye, Save, Rocket } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface ComponentConfig {
  id: string;
  type: 'hero' | 'form' | 'text' | 'image' | 'cta' | 'social-proof' | 'instagram-search';
  config: any;
  order: number;
}

interface PageConfig {
  id?: string;
  title: string;
  slug: string;
  template_type: string;
  status: 'draft' | 'published';
  components: ComponentConfig[];
  seo_config: any;
}

const componentTypes = [
  { value: 'hero', label: 'Hero Section', icon: '🎯' },
  { value: 'form', label: 'Lead Form', icon: '📝' },
  { value: 'text', label: 'Text Block', icon: '📄' },
  { value: 'image', label: 'Image', icon: '🖼️' },
  { value: 'cta', label: 'Call to Action', icon: '🎬' },
  { value: 'social-proof', label: 'Social Proof', icon: '⭐' },
  { value: 'instagram-search', label: 'Instagram Search', icon: '📸' }
];

export function DynamicPageBuilder() {
  const [pageConfig, setPageConfig] = useState<PageConfig>({
    title: '',
    slug: '',
    template_type: 'custom',
    status: 'draft',
    components: [],
    seo_config: {}
  });
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [isPreview, setIsPreview] = useState(false);
  const { toast } = useToast();

  const addComponent = useCallback((type: ComponentConfig['type']) => {
    const newComponent: ComponentConfig = {
      id: `${type}-${Date.now()}`,
      type,
      config: getDefaultConfig(type),
      order: pageConfig.components.length
    };
    
    setPageConfig(prev => ({
      ...prev,
      components: [...prev.components, newComponent]
    }));
  }, [pageConfig.components.length]);

  const updateComponent = useCallback((id: string, config: any) => {
    setPageConfig(prev => ({
      ...prev,
      components: prev.components.map(comp => 
        comp.id === id ? { ...comp, config } : comp
      )
    }));
  }, []);

  const removeComponent = useCallback((id: string) => {
    setPageConfig(prev => ({
      ...prev,
      components: prev.components.filter(comp => comp.id !== id)
    }));
  }, []);

  const savePage = async () => {
    try {
      const pageData = {
        title: pageConfig.title,
        slug: pageConfig.slug,
        template_type: pageConfig.template_type,
        status: pageConfig.status,
        components: pageConfig.components as any,
        config: { layout: 'dynamic' } as any,
        seo_config: pageConfig.seo_config as any
      };

      const { data, error } = await supabase
        .from('landing_pages')
        .insert([pageData])
        .select();

      if (error) throw error;

      toast({
        title: "Page saved successfully!",
        description: `Landing page "${pageConfig.title}" has been saved.`
      });

      setPageConfig(prev => ({ ...prev, id: data[0].id }));
    } catch (error) {
      console.error('Error saving page:', error);
      toast({
        title: "Error saving page",
        description: "Please try again.",
        variant: "destructive"
      });
    }
  };

  const publishPage = async () => {
    try {
      const updatedPage = { ...pageConfig, status: 'published' as const };
      
      if (pageConfig.id) {
        const { error } = await supabase
          .from('landing_pages')
          .update({
            status: 'published',
            components: updatedPage.components as any,
            config: { layout: 'dynamic' } as any
          })
          .eq('id', pageConfig.id);

        if (error) throw error;
      } else {
        await savePage();
      }

      setPageConfig(updatedPage);
      
      toast({
        title: "Page published successfully!",
        description: `Your landing page is now live at /${pageConfig.slug}`
      });
    } catch (error) {
      console.error('Error publishing page:', error);
      toast({
        title: "Error publishing page",
        description: "Please try again.",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="h-screen flex">
      {/* Left Sidebar - Components & Settings */}
      <div className="w-80 border-r bg-muted/10 p-4 overflow-y-auto">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Page Builder</h2>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => setIsPreview(!isPreview)}>
                <Eye className="h-4 w-4" />
              </Button>
              <Button size="sm" onClick={savePage}>
                <Save className="h-4 w-4" />
              </Button>
              <Button size="sm" onClick={publishPage}>
                <Rocket className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <Tabs defaultValue="settings" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="components">Components</TabsTrigger>
            </TabsList>
            
            <TabsContent value="settings" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Page Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <Label htmlFor="title">Page Title</Label>
                    <Input
                      id="title"
                      value={pageConfig.title}
                      onChange={(e) => setPageConfig(prev => ({ ...prev, title: e.target.value }))}
                      placeholder="Enter page title"
                    />
                  </div>
                  <div>
                    <Label htmlFor="slug">URL Slug</Label>
                    <Input
                      id="slug"
                      value={pageConfig.slug}
                      onChange={(e) => setPageConfig(prev => ({ ...prev, slug: e.target.value }))}
                      placeholder="page-url"
                    />
                  </div>
                  <div>
                    <Label>Status</Label>
                    <Badge variant={pageConfig.status === 'published' ? 'default' : 'secondary'}>
                      {pageConfig.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="components" className="space-y-4">
              <div>
                <h3 className="text-sm font-medium mb-3">Add Components</h3>
                <div className="grid grid-cols-1 gap-2">
                  {componentTypes.map((type) => (
                    <Button
                      key={type.value}
                      variant="outline"
                      size="sm"
                      onClick={() => addComponent(type.value as ComponentConfig['type'])}
                      className="justify-start text-left"
                    >
                      <span className="mr-2">{type.icon}</span>
                      {type.label}
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium mb-3">Page Structure</h3>
                <div className="space-y-2">
                  {pageConfig.components.map((component, index) => (
                    <div
                      key={component.id}
                      className={`p-2 border rounded-md cursor-pointer transition-colors ${
                        selectedComponent === component.id ? 'bg-primary/10 border-primary' : 'hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedComponent(component.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Move className="h-3 w-3" />
                          <span className="text-xs">
                            {componentTypes.find(t => t.value === component.type)?.label}
                          </span>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeComponent(component.id);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Component Editor */}
        {selectedComponent && !isPreview && (
          <div className="border-b bg-muted/5 p-4">
            <ComponentEditor
              component={pageConfig.components.find(c => c.id === selectedComponent)!}
              onUpdate={(config) => updateComponent(selectedComponent, config)}
            />
          </div>
        )}

        {/* Preview Area */}
        <div className="flex-1 overflow-y-auto bg-background">
          {isPreview ? (
            <PagePreview components={pageConfig.components} />
          ) : (
            <PageCanvas 
              components={pageConfig.components}
              selectedComponent={selectedComponent}
              onSelectComponent={setSelectedComponent}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function getDefaultConfig(type: ComponentConfig['type']) {
  switch (type) {
    case 'hero':
      return {
        title: 'Welcome to Our Amazing Product',
        subtitle: 'Transform your life with our innovative solution',
        ctaText: 'Get Started',
        backgroundImage: '',
        textAlign: 'center'
      };
    case 'form':
      return {
        title: 'Get Started Today',
        fields: [
          { type: 'text', label: 'Name', required: true },
          { type: 'email', label: 'Email', required: true }
        ],
        submitText: 'Submit',
        successMessage: 'Thank you for your submission!'
      };
    case 'text':
      return {
        content: 'Add your content here...',
        alignment: 'left',
        fontSize: 'base'
      };
    case 'image':
      return {
        src: '',
        alt: '',
        caption: '',
        width: '100%'
      };
    case 'cta':
      return {
        title: 'Ready to Get Started?',
        description: 'Join thousands of satisfied customers',
        buttonText: 'Start Now',
        buttonUrl: '#',
        style: 'primary'
      };
    case 'social-proof':
      return {
        title: 'Trusted by thousands',
        testimonials: [
          { name: 'John Doe', text: 'This product changed my life!', avatar: '' }
        ],
        showLogos: true,
        logos: []
      };
    case 'instagram-search':
      return {
        title: 'Find Your Instagram Profile',
        placeholder: 'Enter your Instagram handle',
        apiEndpoint: '/api/instagram-search',
        showResults: true
      };
    default:
      return {};
  }
}

function ComponentEditor({ component, onUpdate }: { component: ComponentConfig; onUpdate: (config: any) => void }) {
  const { type, config } = component;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <Settings className="h-4 w-4" />
          Edit {componentTypes.find(t => t.value === type)?.label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {type === 'hero' && (
          <div className="space-y-3">
            <div>
              <Label>Title</Label>
              <Input
                value={config.title || ''}
                onChange={(e) => onUpdate({ ...config, title: e.target.value })}
              />
            </div>
            <div>
              <Label>Subtitle</Label>
              <Textarea
                value={config.subtitle || ''}
                onChange={(e) => onUpdate({ ...config, subtitle: e.target.value })}
              />
            </div>
            <div>
              <Label>CTA Button Text</Label>
              <Input
                value={config.ctaText || ''}
                onChange={(e) => onUpdate({ ...config, ctaText: e.target.value })}
              />
            </div>
          </div>
        )}
        {type === 'form' && (
          <div className="space-y-3">
            <div>
              <Label>Form Title</Label>
              <Input
                value={config.title || ''}
                onChange={(e) => onUpdate({ ...config, title: e.target.value })}
              />
            </div>
            <div>
              <Label>Submit Button Text</Label>
              <Input
                value={config.submitText || ''}
                onChange={(e) => onUpdate({ ...config, submitText: e.target.value })}
              />
            </div>
          </div>
        )}
        {type === 'instagram-search' && (
          <div className="space-y-3">
            <div>
              <Label>Section Title</Label>
              <Input
                value={config.title || ''}
                onChange={(e) => onUpdate({ ...config, title: e.target.value })}
              />
            </div>
            <div>
              <Label>Placeholder Text</Label>
              <Input
                value={config.placeholder || ''}
                onChange={(e) => onUpdate({ ...config, placeholder: e.target.value })}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function PageCanvas({ 
  components, 
  selectedComponent, 
  onSelectComponent 
}: { 
  components: ComponentConfig[]; 
  selectedComponent: string | null;
  onSelectComponent: (id: string) => void;
}) {
  return (
    <div className="p-8 space-y-4">
      {components.map((component) => (
        <div
          key={component.id}
          className={`border-2 border-dashed rounded-lg p-4 cursor-pointer transition-all ${
            selectedComponent === component.id 
              ? 'border-primary bg-primary/5' 
              : 'border-muted hover:border-primary/50'
          }`}
          onClick={() => onSelectComponent(component.id)}
        >
          <ComponentRenderer component={component} isCanvas />
        </div>
      ))}
      {components.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Plus className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Add components to start building your landing page</p>
        </div>
      )}
    </div>
  );
}

function PagePreview({ components }: { components: ComponentConfig[] }) {
  return (
    <div className="space-y-0">
      {components.map((component) => (
        <ComponentRenderer key={component.id} component={component} />
      ))}
    </div>
  );
}

function ComponentRenderer({ component, isCanvas = false }: { component: ComponentConfig; isCanvas?: boolean }) {
  const { type, config } = component;

  if (isCanvas) {
    return (
      <div className="text-center space-y-2">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">
          {componentTypes.find(t => t.value === type)?.label}
        </div>
        <div className="text-sm font-medium">{config.title || 'Component Title'}</div>
      </div>
    );
  }

  switch (type) {
    case 'hero':
      return (
        <section className="py-20 px-8 text-center bg-gradient-to-r from-primary/10 to-primary/5">
          <h1 className="text-4xl font-bold mb-4">{config.title}</h1>
          <p className="text-xl text-muted-foreground mb-8">{config.subtitle}</p>
          <Button size="lg">{config.ctaText}</Button>
        </section>
      );
    case 'form':
      return (
        <section className="py-16 px-8">
          <div className="max-w-md mx-auto">
            <h2 className="text-2xl font-bold mb-6 text-center">{config.title}</h2>
            <form className="space-y-4">
              {config.fields?.map((field: any, index: number) => (
                <div key={index}>
                  <Label>{field.label}</Label>
                  <Input type={field.type} required={field.required} />
                </div>
              ))}
              <Button type="submit" className="w-full">{config.submitText}</Button>
            </form>
          </div>
        </section>
      );
    case 'instagram-search':
      return (
        <section className="py-16 px-8">
          <div className="max-w-md mx-auto text-center">
            <h2 className="text-2xl font-bold mb-6">{config.title}</h2>
            <div className="flex gap-2">
              <Input placeholder={config.placeholder} className="flex-1" />
              <Button>Search</Button>
            </div>
          </div>
        </section>
      );
    default:
      return (
        <div className="py-8 px-8 text-center">
          <p>Component: {type}</p>
        </div>
      );
  }
}