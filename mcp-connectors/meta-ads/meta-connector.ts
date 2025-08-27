/**
 * Meta (Facebook/Instagram) Ads MCP Connector
 */

import { 
  BaseConfig,
  ApiResponse, 
  Campaign, 
  CampaignPerformance, 
  Creative, 
  Audience,
  ContentPolicyValidation,
  ConnectorError
} from '../shared/types.js';
import { BaseAdConnector } from '../shared/base-connector.js';
import { buildQueryString, sanitizeInput } from '../shared/utils.js';

interface MetaConfig extends BaseConfig {
  appId: string;
  appSecret: string;
  businessAccountId: string;
  apiVersion?: string;
}

interface MetaCampaignData {
  name: string;
  objective: string;
  status: string;
  daily_budget?: number;
  lifetime_budget?: number;
  bid_strategy?: string;
  start_time?: string;
  stop_time?: string;
  targeting?: any;
  creative?: any;
}

export class MetaAdsConnector extends BaseAdConnector {
  protected config: MetaConfig;
  private baseUrl: string;
  
  constructor(config: MetaConfig) {
    super(config, 'Meta');
    this.config = config;
    this.baseUrl = `https://graph.facebook.com/${config.apiVersion || 'v18.0'}`;
  }
  
  protected getPlatform(): 'META' {
    return 'META';
  }
  
  async testConnection(): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/me?fields=id,name`
      );
      
      return {
        success: true,
        data: true
      };
    } catch (error) {
      this.logger.error('Connection test failed', error);
      return {
        success: false,
        error: {
          code: 'CONNECTION_FAILED',
          message: 'Failed to connect to Meta API'
        }
      };
    }
  }
  
  // Campaign Management
  async createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>> {
    try {
      // Check spending limits
      const estimatedDailySpend = campaign.budget.type === 'DAILY' ? campaign.budget.amount : 
                                  campaign.budget.amount / 30; // Rough estimate
      
      if (!(await this.checkSpendingLimits(estimatedDailySpend))) {
        return {
          success: false,
          error: {
            code: 'SPENDING_LIMIT_EXCEEDED',
            message: 'Campaign creation would exceed spending limits'
          }
        };
      }
      
      // Transform GAELP campaign to Meta format
      const metaCampaignData: MetaCampaignData = {
        name: sanitizeInput(campaign.name),
        objective: this.mapObjectiveToMeta(campaign.objective.type),
        status: campaign.status === 'ACTIVE' ? 'ACTIVE' : 'PAUSED',
        bid_strategy: campaign.bidStrategy?.type || 'LOWEST_COST_WITHOUT_CAP'
      };
      
      // Set budget
      if (campaign.budget.type === 'DAILY') {
        metaCampaignData.daily_budget = Math.round(campaign.budget.amount * 100); // Meta uses cents
      } else {
        metaCampaignData.lifetime_budget = Math.round(campaign.budget.amount * 100);
      }
      
      // Set schedule
      if (campaign.startDate) {
        metaCampaignData.start_time = new Date(campaign.startDate).toISOString();
      }
      if (campaign.endDate) {
        metaCampaignData.stop_time = new Date(campaign.endDate).toISOString();
      }
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${this.config.accountId}/campaigns`,
        metaCampaignData
      );
      
      if (!response.success) {
        return response;
      }
      
      const createdCampaign = await this.getCampaign(response.data.id);
      return createdCampaign;
      
    } catch (error) {
      this.logger.error('Failed to create campaign', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_CREATION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>> {
    try {
      const updateData: Partial<MetaCampaignData> = {};
      
      if (updates.name) {
        updateData.name = sanitizeInput(updates.name);
      }
      
      if (updates.status) {
        updateData.status = updates.status === 'ACTIVE' ? 'ACTIVE' : 'PAUSED';
      }
      
      if (updates.budget) {
        if (updates.budget.type === 'DAILY') {
          updateData.daily_budget = Math.round(updates.budget.amount * 100);
        } else {
          updateData.lifetime_budget = Math.round(updates.budget.amount * 100);
        }
      }
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${campaignId}`,
        updateData
      );
      
      if (!response.success) {
        return response;
      }
      
      return this.getCampaign(campaignId);
      
    } catch (error) {
      this.logger.error('Failed to update campaign', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_UPDATE_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async getCampaign(campaignId: string): Promise<ApiResponse<Campaign>> {
    try {
      const fields = [
        'id', 'name', 'objective', 'status', 'created_time', 'updated_time',
        'daily_budget', 'lifetime_budget', 'bid_strategy', 'start_time', 'stop_time'
      ].join(',');
      
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${campaignId}?fields=${fields}`
      );
      
      if (!response.success) {
        return response;
      }
      
      const metaCampaign = response.data;
      const campaign: Campaign = {
        id: metaCampaign.id,
        name: metaCampaign.name,
        status: this.mapMetaStatusToGAELP(metaCampaign.status),
        objective: { type: this.mapMetaObjectiveToGAELP(metaCampaign.objective) },
        budget: {
          amount: metaCampaign.daily_budget ? 
                  metaCampaign.daily_budget / 100 : 
                  metaCampaign.lifetime_budget / 100,
          currency: 'USD', // Default, should be retrieved from account settings
          type: metaCampaign.daily_budget ? 'DAILY' : 'LIFETIME'
        },
        targeting: {}, // Would need additional API calls to get full targeting
        creatives: [], // Would need additional API calls to get creatives
        startDate: metaCampaign.start_time,
        endDate: metaCampaign.stop_time,
        bidStrategy: {
          type: this.mapMetaBidStrategyToGAELP(metaCampaign.bid_strategy)
        }
      };
      
      return {
        success: true,
        data: campaign
      };
      
    } catch (error) {
      this.logger.error('Failed to get campaign', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_RETRIEVAL_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async listCampaigns(filters?: any): Promise<ApiResponse<Campaign[]>> {
    try {
      const params: any = {
        fields: 'id,name,objective,status,daily_budget,lifetime_budget',
        limit: filters?.limit || 25
      };
      
      if (filters?.status) {
        params.effective_status = [filters.status];
      }
      
      const queryString = buildQueryString(params);
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${this.config.accountId}/campaigns?${queryString}`
      );
      
      if (!response.success) {
        return response;
      }
      
      const campaigns: Campaign[] = response.data.data.map((metaCampaign: any) => ({
        id: metaCampaign.id,
        name: metaCampaign.name,
        status: this.mapMetaStatusToGAELP(metaCampaign.status),
        objective: { type: this.mapMetaObjectiveToGAELP(metaCampaign.objective) },
        budget: {
          amount: metaCampaign.daily_budget ? 
                  metaCampaign.daily_budget / 100 : 
                  metaCampaign.lifetime_budget / 100,
          currency: 'USD',
          type: metaCampaign.daily_budget ? 'DAILY' : 'LIFETIME'
        },
        targeting: {},
        creatives: []
      }));
      
      return {
        success: true,
        data: campaigns
      };
      
    } catch (error) {
      this.logger.error('Failed to list campaigns', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_LIST_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async deleteCampaign(campaignId: string): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'DELETE',
        `${this.baseUrl}/${campaignId}`
      );
      
      return {
        success: response.success,
        data: response.success
      };
      
    } catch (error) {
      this.logger.error('Failed to delete campaign', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_DELETION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async pauseCampaign(campaignId: string): Promise<ApiResponse<boolean>> {
    return this.updateCampaignStatus(campaignId, 'PAUSED');
  }
  
  async resumeCampaign(campaignId: string): Promise<ApiResponse<boolean>> {
    return this.updateCampaignStatus(campaignId, 'ACTIVE');
  }
  
  private async updateCampaignStatus(campaignId: string, status: string): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${campaignId}`,
        { status }
      );
      
      return {
        success: response.success,
        data: response.success
      };
      
    } catch (error) {
      this.logger.error('Failed to update campaign status', error);
      return {
        success: false,
        error: {
          code: 'CAMPAIGN_STATUS_UPDATE_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  // Creative Management
  async uploadCreative(creative: Creative): Promise<ApiResponse<Creative>> {
    try {
      // For Meta, creatives are uploaded as ad creatives
      const creativeData: any = {
        name: sanitizeInput(creative.name),
        object_story_spec: {}
      };
      
      // Handle different creative types
      if (creative.type === 'IMAGE' && creative.assets.images?.[0]) {
        // First upload the image
        const imageUpload = await this.uploadImage(creative.assets.images[0].url);
        if (!imageUpload.success) {
          return imageUpload;
        }
        
        creativeData.object_story_spec = {
          page_id: this.config.businessAccountId, // This should be the page ID
          link_data: {
            image_hash: imageUpload.data.hash,
            link: creative.destinationUrl,
            message: creative.assets.text?.description,
            name: creative.assets.text?.headline,
            call_to_action: {
              type: creative.assets.text?.callToAction || 'LEARN_MORE'
            }
          }
        };
      }
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${this.config.accountId}/adcreatives`,
        creativeData
      );
      
      if (!response.success) {
        return response;
      }
      
      const uploadedCreative: Creative = {
        id: response.data.id,
        name: creative.name,
        type: creative.type,
        assets: creative.assets,
        destinationUrl: creative.destinationUrl
      };
      
      return {
        success: true,
        data: uploadedCreative
      };
      
    } catch (error) {
      this.logger.error('Failed to upload creative', error);
      return {
        success: false,
        error: {
          code: 'CREATIVE_UPLOAD_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  private async uploadImage(imageUrl: string): Promise<ApiResponse<{ hash: string }>> {
    try {
      // In a real implementation, you would download the image and upload it to Meta
      // This is a simplified version
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${this.config.accountId}/adimages`,
        { url: imageUrl }
      );
      
      return response;
    } catch (error) {
      this.logger.error('Failed to upload image', error);
      return {
        success: false,
        error: {
          code: 'IMAGE_UPLOAD_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async updateCreative(creativeId: string, updates: Partial<Creative>): Promise<ApiResponse<Creative>> {
    // Meta doesn't allow updating creatives directly - you need to create a new one
    throw new Error('Meta Ads API does not support updating existing creatives. Create a new creative instead.');
  }
  
  async getCreative(creativeId: string): Promise<ApiResponse<Creative>> {
    try {
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${creativeId}?fields=id,name,object_story_spec`
      );
      
      if (!response.success) {
        return response;
      }
      
      // Transform Meta creative back to GAELP format
      const metaCreative = response.data;
      const creative: Creative = {
        id: metaCreative.id,
        name: metaCreative.name,
        type: 'IMAGE', // Simplified - would need to determine from object_story_spec
        assets: {
          text: {
            headline: metaCreative.object_story_spec?.link_data?.name || '',
            description: metaCreative.object_story_spec?.link_data?.message || ''
          }
        },
        destinationUrl: metaCreative.object_story_spec?.link_data?.link
      };
      
      return {
        success: true,
        data: creative
      };
      
    } catch (error) {
      this.logger.error('Failed to get creative', error);
      return {
        success: false,
        error: {
          code: 'CREATIVE_RETRIEVAL_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async listCreatives(filters?: any): Promise<ApiResponse<Creative[]>> {
    try {
      const params: any = {
        fields: 'id,name,object_story_spec',
        limit: filters?.limit || 25
      };
      
      const queryString = buildQueryString(params);
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${this.config.accountId}/adcreatives?${queryString}`
      );
      
      if (!response.success) {
        return response;
      }
      
      const creatives: Creative[] = response.data.data.map((metaCreative: any) => ({
        id: metaCreative.id,
        name: metaCreative.name,
        type: 'IMAGE' as const,
        assets: {
          text: {
            headline: metaCreative.object_story_spec?.link_data?.name || '',
            description: metaCreative.object_story_spec?.link_data?.message || ''
          }
        },
        destinationUrl: metaCreative.object_story_spec?.link_data?.link
      }));
      
      return {
        success: true,
        data: creatives
      };
      
    } catch (error) {
      this.logger.error('Failed to list creatives', error);
      return {
        success: false,
        error: {
          code: 'CREATIVE_LIST_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async deleteCreative(creativeId: string): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'DELETE',
        `${this.baseUrl}/${creativeId}`
      );
      
      return {
        success: response.success,
        data: response.success
      };
      
    } catch (error) {
      this.logger.error('Failed to delete creative', error);
      return {
        success: false,
        error: {
          code: 'CREATIVE_DELETION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  // Performance Monitoring
  async getCampaignPerformance(
    campaignId: string,
    dateRange: { start: string; end: string },
    metrics?: string[]
  ): Promise<ApiResponse<CampaignPerformance>> {
    try {
      const defaultMetrics = [
        'impressions', 'clicks', 'spend', 'actions',
        'cost_per_action_type', 'ctr', 'cpm', 'cpp'
      ];
      
      const params = {
        fields: (metrics || defaultMetrics).join(','),
        time_range: JSON.stringify({
          since: dateRange.start,
          until: dateRange.end
        }),
        level: 'campaign'
      };
      
      const queryString = buildQueryString(params);
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${campaignId}/insights?${queryString}`
      );
      
      if (!response.success) {
        return response;
      }
      
      const insights = response.data.data[0] || {};
      
      // Get campaign name
      const campaignResponse = await this.getCampaign(campaignId);
      const campaignName = campaignResponse.success ? 
                          campaignResponse.data!.name : 
                          `Campaign ${campaignId}`;
      
      const performance: CampaignPerformance = {
        campaignId,
        campaignName,
        impressions: parseInt(insights.impressions || '0'),
        clicks: parseInt(insights.clicks || '0'),
        conversions: this.extractConversions(insights.actions),
        spend: parseFloat(insights.spend || '0'),
        ctr: parseFloat(insights.ctr || '0'),
        cpc: parseFloat(insights.cpc || '0'),
        cpm: parseFloat(insights.cpm || '0'),
        dateRange
      };
      
      // Calculate derived metrics
      if (performance.conversions > 0 && performance.spend > 0) {
        performance.cpa = performance.spend / performance.conversions;
      }
      
      return {
        success: true,
        data: performance
      };
      
    } catch (error) {
      this.logger.error('Failed to get campaign performance', error);
      return {
        success: false,
        error: {
          code: 'PERFORMANCE_RETRIEVAL_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async getAccountPerformance(
    dateRange: { start: string; end: string },
    breakdown?: string
  ): Promise<ApiResponse<CampaignPerformance[]>> {
    try {
      const campaigns = await this.listCampaigns();
      if (!campaigns.success) {
        return campaigns as ApiResponse<CampaignPerformance[]>;
      }
      
      const performances: CampaignPerformance[] = [];
      
      for (const campaign of campaigns.data!) {
        if (campaign.id) {
          const performance = await this.getCampaignPerformance(campaign.id, dateRange);
          if (performance.success) {
            performances.push(performance.data!);
          }
        }
      }
      
      return {
        success: true,
        data: performances
      };
      
    } catch (error) {
      this.logger.error('Failed to get account performance', error);
      return {
        success: false,
        error: {
          code: 'ACCOUNT_PERFORMANCE_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  private extractConversions(actions: any[]): number {
    if (!actions || !Array.isArray(actions)) {
      return 0;
    }
    
    const conversionActions = actions.filter(action => 
      action.action_type === 'purchase' || 
      action.action_type === 'lead' ||
      action.action_type === 'complete_registration'
    );
    
    return conversionActions.reduce((sum, action) => sum + parseInt(action.value || '0'), 0);
  }
  
  // Audience Management
  async createAudience(audience: Audience): Promise<ApiResponse<Audience>> {
    try {
      const audienceData = {
        name: sanitizeInput(audience.name),
        description: audience.description,
        subtype: this.mapAudienceTypeToMeta(audience.type),
        retention_days: audience.retentionDays || 180
      };
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/${this.config.accountId}/customaudiences`,
        audienceData
      );
      
      if (!response.success) {
        return response;
      }
      
      const createdAudience: Audience = {
        id: response.data.id,
        name: audience.name,
        type: audience.type,
        description: audience.description,
        retentionDays: audience.retentionDays
      };
      
      return {
        success: true,
        data: createdAudience
      };
      
    } catch (error) {
      this.logger.error('Failed to create audience', error);
      return {
        success: false,
        error: {
          code: 'AUDIENCE_CREATION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async getAudience(audienceId: string): Promise<ApiResponse<Audience>> {
    try {
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${audienceId}?fields=id,name,description,approximate_count,retention_days`
      );
      
      if (!response.success) {
        return response;
      }
      
      const metaAudience = response.data;
      const audience: Audience = {
        id: metaAudience.id,
        name: metaAudience.name,
        type: 'CUSTOM', // Simplified
        description: metaAudience.description,
        size: metaAudience.approximate_count,
        retentionDays: metaAudience.retention_days
      };
      
      return {
        success: true,
        data: audience
      };
      
    } catch (error) {
      this.logger.error('Failed to get audience', error);
      return {
        success: false,
        error: {
          code: 'AUDIENCE_RETRIEVAL_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async listAudiences(filters?: any): Promise<ApiResponse<Audience[]>> {
    try {
      const params = {
        fields: 'id,name,description,approximate_count',
        limit: filters?.limit || 25
      };
      
      const queryString = buildQueryString(params);
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/${this.config.accountId}/customaudiences?${queryString}`
      );
      
      if (!response.success) {
        return response;
      }
      
      const audiences: Audience[] = response.data.data.map((metaAudience: any) => ({
        id: metaAudience.id,
        name: metaAudience.name,
        type: 'CUSTOM' as const,
        description: metaAudience.description,
        size: metaAudience.approximate_count
      }));
      
      return {
        success: true,
        data: audiences
      };
      
    } catch (error) {
      this.logger.error('Failed to list audiences', error);
      return {
        success: false,
        error: {
          code: 'AUDIENCE_LIST_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async deleteAudience(audienceId: string): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'DELETE',
        `${this.baseUrl}/${audienceId}`
      );
      
      return {
        success: response.success,
        data: response.success
      };
      
    } catch (error) {
      this.logger.error('Failed to delete audience', error);
      return {
        success: false,
        error: {
          code: 'AUDIENCE_DELETION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  // Content Policy Validation
  async validateContent(creative: Creative): Promise<ApiResponse<ContentPolicyValidation>> {
    try {
      // Meta doesn't have a direct content validation API, but we can do basic checks
      const validation: ContentPolicyValidation = {
        valid: true,
        errors: [],
        warnings: [],
        violatedPolicies: [],
        suggestedChanges: []
      };
      
      // Basic text validation
      if (creative.assets.text) {
        const { headline, description } = creative.assets.text;
        
        // Check for prohibited content
        const prohibitedTerms = ['guarantee', 'click here', 'free money', 'get rich quick'];
        const text = `${headline} ${description}`.toLowerCase();
        
        for (const term of prohibitedTerms) {
          if (text.includes(term)) {
            validation.warnings.push(`Content contains potentially problematic term: "${term}"`);
            validation.suggestedChanges.push(`Consider removing or replacing "${term}"`);
          }
        }
        
        // Check length limits
        if (headline && headline.length > 25) {
          validation.warnings.push('Headline may be truncated if longer than 25 characters');
        }
        
        if (description && description.length > 125) {
          validation.warnings.push('Description may be truncated if longer than 125 characters');
        }
      }
      
      return {
        success: true,
        data: validation
      };
      
    } catch (error) {
      this.logger.error('Failed to validate content', error);
      return {
        success: false,
        error: {
          code: 'CONTENT_VALIDATION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  // Helper methods for mapping between GAELP and Meta formats
  private mapObjectiveToMeta(objective: string): string {
    const mapping: Record<string, string> = {
      'BRAND_AWARENESS': 'BRAND_AWARENESS',
      'REACH': 'REACH',
      'TRAFFIC': 'LINK_CLICKS',
      'ENGAGEMENT': 'ENGAGEMENT',
      'APP_INSTALLS': 'APP_INSTALLS',
      'VIDEO_VIEWS': 'VIDEO_VIEWS',
      'LEAD_GENERATION': 'LEAD_GENERATION',
      'MESSAGES': 'MESSAGES',
      'CONVERSIONS': 'CONVERSIONS',
      'SALES': 'CONVERSIONS'
    };
    return mapping[objective] || 'LINK_CLICKS';
  }
  
  private mapMetaObjectiveToGAELP(objective: string): any {
    const mapping: Record<string, string> = {
      'BRAND_AWARENESS': 'BRAND_AWARENESS',
      'REACH': 'REACH',
      'LINK_CLICKS': 'TRAFFIC',
      'ENGAGEMENT': 'ENGAGEMENT',
      'APP_INSTALLS': 'APP_INSTALLS',
      'VIDEO_VIEWS': 'VIDEO_VIEWS',
      'LEAD_GENERATION': 'LEAD_GENERATION',
      'MESSAGES': 'MESSAGES',
      'CONVERSIONS': 'CONVERSIONS'
    };
    return mapping[objective] || 'TRAFFIC';
  }
  
  private mapMetaStatusToGAELP(status: string): any {
    const mapping: Record<string, string> = {
      'ACTIVE': 'ACTIVE',
      'PAUSED': 'PAUSED',
      'DELETED': 'DELETED',
      'PENDING_REVIEW': 'PENDING_REVIEW',
      'DISAPPROVED': 'DISAPPROVED'
    };
    return mapping[status] || 'PAUSED';
  }
  
  private mapMetaBidStrategyToGAELP(bidStrategy: string): any {
    const mapping: Record<string, string> = {
      'LOWEST_COST_WITHOUT_CAP': 'AUTOMATIC',
      'LOWEST_COST_WITH_BID_CAP': 'MANUAL_CPC',
      'TARGET_COST': 'TARGET_CPA'
    };
    return mapping[bidStrategy] || 'AUTOMATIC';
  }
  
  private mapAudienceTypeToMeta(type: string): string {
    const mapping: Record<string, string> = {
      'CUSTOM': 'CUSTOM',
      'LOOKALIKE': 'LOOKALIKE',
      'SAVED': 'CUSTOM'
    };
    return mapping[type] || 'CUSTOM';
  }
}