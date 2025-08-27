/**
 * Google Ads MCP Connector
 */

import { 
  BaseConfig,
  ApiResponse, 
  Campaign, 
  CampaignPerformance, 
  Creative, 
  Audience,
  ContentPolicyValidation
} from '../shared/types.js';
import { BaseAdConnector } from '../shared/base-connector.js';
import { buildQueryString, sanitizeInput } from '../shared/utils.js';

interface GoogleAdsConfig extends BaseConfig {
  customerId: string;
  developerToken: string;
  loginCustomerId?: string;
  apiVersion?: string;
}

interface GoogleCampaignData {
  name: string;
  status: string;
  advertising_channel_type: string;
  advertising_channel_sub_type?: string;
  campaign_budget: string;
  target_cpa?: {
    target_cpa_micros: number;
  };
  target_roas?: {
    target_roas: number;
  };
  manual_cpc?: {
    enhanced_cpc_enabled: boolean;
  };
  start_date?: string;
  end_date?: string;
}

export class GoogleAdsConnector extends BaseAdConnector {
  protected config: GoogleAdsConfig;
  private baseUrl: string;
  
  constructor(config: GoogleAdsConfig) {
    super(config, 'Google');
    this.config = config;
    this.baseUrl = `https://googleads.googleapis.com/${config.apiVersion || 'v14'}`;
  }
  
  protected getPlatform(): 'GOOGLE' {
    return 'GOOGLE';
  }
  
  async testConnection(): Promise<ApiResponse<boolean>> {
    try {
      const response = await this.makeApiRequest<any>(
        'GET',
        `${this.baseUrl}/customers/${this.config.customerId}`
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
          message: 'Failed to connect to Google Ads API'
        }
      };
    }
  }
  
  protected async makeApiRequest<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    url: string,
    data?: any,
    headers?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    const googleHeaders = {
      'developer-token': this.config.developerToken,
      'login-customer-id': this.config.loginCustomerId || this.config.customerId,
      ...headers
    };
    
    return super.makeApiRequest(method, url, data, googleHeaders);
  }
  
  // Campaign Management
  async createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>> {
    try {
      // Check spending limits
      const estimatedDailySpend = campaign.budget.type === 'DAILY' ? campaign.budget.amount : 
                                  campaign.budget.amount / 30;
      
      if (!(await this.checkSpendingLimits(estimatedDailySpend))) {
        return {
          success: false,
          error: {
            code: 'SPENDING_LIMIT_EXCEEDED',
            message: 'Campaign creation would exceed spending limits'
          }
        };
      }
      
      // First create campaign budget
      const budgetResponse = await this.createCampaignBudget(campaign.budget);
      if (!budgetResponse.success) {
        return {
          success: false,
          error: budgetResponse.error
        };
      }
      
      // Transform GAELP campaign to Google Ads format
      const googleCampaignData: GoogleCampaignData = {
        name: sanitizeInput(campaign.name),
        status: campaign.status === 'ACTIVE' ? 'ENABLED' : 'PAUSED',
        advertising_channel_type: this.mapObjectiveToGoogleChannel(campaign.objective.type),
        campaign_budget: budgetResponse.data!.resourceName
      };
      
      // Set bid strategy
      if (campaign.bidStrategy) {
        switch (campaign.bidStrategy.type) {
          case 'TARGET_CPA':
            googleCampaignData.target_cpa = {
              target_cpa_micros: (campaign.bidStrategy.amount || 0) * 1000000
            };
            break;
          case 'TARGET_ROAS':
            googleCampaignData.target_roas = {
              target_roas: campaign.bidStrategy.amount || 0
            };
            break;
          case 'MANUAL_CPC':
            googleCampaignData.manual_cpc = {
              enhanced_cpc_enabled: true
            };
            break;
        }
      }
      
      // Set schedule
      if (campaign.startDate) {
        googleCampaignData.start_date = campaign.startDate.replace(/-/g, '');
      }
      if (campaign.endDate) {
        googleCampaignData.end_date = campaign.endDate.replace(/-/g, '');
      }
      
      const operations = [{
        create: googleCampaignData
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/campaigns:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      const createdCampaignResource = response.data.results[0].resourceName;
      const campaignId = createdCampaignResource.split('/').pop();
      
      const createdCampaign = await this.getCampaign(campaignId);
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
  
  private async createCampaignBudget(budget: any): Promise<ApiResponse<{ resourceName: string }>> {
    try {
      const budgetData = {
        name: `Budget_${Date.now()}`,
        amount_micros: budget.amount * 1000000, // Convert to micros
        delivery_method: 'STANDARD',
        explicitly_shared: false
      };
      
      const operations = [{
        create: budgetData
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/campaignBudgets:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      return {
        success: true,
        data: {
          resourceName: response.data.results[0].resourceName
        }
      };
      
    } catch (error) {
      this.logger.error('Failed to create campaign budget', error);
      return {
        success: false,
        error: {
          code: 'BUDGET_CREATION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>> {
    try {
      const updateData: any = {
        resource_name: `customers/${this.config.customerId}/campaigns/${campaignId}`
      };
      
      if (updates.name) {
        updateData.name = sanitizeInput(updates.name);
      }
      
      if (updates.status) {
        updateData.status = updates.status === 'ACTIVE' ? 'ENABLED' : 'PAUSED';
      }
      
      const operations = [{
        update: updateData,
        update_mask: {
          paths: Object.keys(updateData).filter(key => key !== 'resource_name')
        }
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/campaigns:mutate`,
        { operations }
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
      const query = `
        SELECT 
          campaign.id,
          campaign.name,
          campaign.status,
          campaign.advertising_channel_type,
          campaign.campaign_budget,
          campaign.start_date,
          campaign.end_date,
          campaign.target_cpa.target_cpa_micros,
          campaign.target_roas.target_roas,
          campaign.manual_cpc.enhanced_cpc_enabled,
          campaign_budget.amount_micros
        FROM campaign 
        LEFT JOIN campaign_budget ON campaign.campaign_budget = campaign_budget.resource_name
        WHERE campaign.id = ${campaignId}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success || !response.data.results?.length) {
        return {
          success: false,
          error: {
            code: 'CAMPAIGN_NOT_FOUND',
            message: `Campaign ${campaignId} not found`
          }
        };
      }
      
      const googleCampaign = response.data.results[0];
      const campaign: Campaign = {
        id: googleCampaign.campaign.id,
        name: googleCampaign.campaign.name,
        status: this.mapGoogleStatusToGAELP(googleCampaign.campaign.status),
        objective: { type: this.mapGoogleChannelToGAELP(googleCampaign.campaign.advertising_channel_type) },
        budget: {
          amount: googleCampaign.campaignBudget?.amount_micros ? 
                  googleCampaign.campaignBudget.amount_micros / 1000000 : 0,
          currency: 'USD', // Default, should be retrieved from account settings
          type: 'DAILY' // Google Ads typically uses daily budgets
        },
        targeting: { demographics: {}, locations: {} }, // Would need additional queries to get targeting
        creatives: [], // Would need additional queries to get creatives
        startDate: this.formatGoogleDate(googleCampaign.campaign.start_date),
        endDate: this.formatGoogleDate(googleCampaign.campaign.end_date),
        bidStrategy: this.mapGoogleBidStrategyToGAELP(googleCampaign.campaign)
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
      let whereClause = '';
      if (filters?.status) {
        whereClause = `WHERE campaign.status = '${filters.status === 'ACTIVE' ? 'ENABLED' : 'PAUSED'}'`;
      }
      
      const query = `
        SELECT 
          campaign.id,
          campaign.name,
          campaign.status,
          campaign.advertising_channel_type,
          campaign_budget.amount_micros
        FROM campaign 
        LEFT JOIN campaign_budget ON campaign.campaign_budget = campaign_budget.resource_name
        ${whereClause}
        LIMIT ${filters?.limit || 25}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success) {
        return response;
      }
      
      const campaigns: Campaign[] = (response.data.results || []).map((result: any) => ({
        id: result.campaign.id,
        name: result.campaign.name,
        status: this.mapGoogleStatusToGAELP(result.campaign.status),
        objective: { type: this.mapGoogleChannelToGAELP(result.campaign.advertising_channel_type) },
        budget: {
          amount: result.campaignBudget?.amount_micros ? 
                  result.campaignBudget.amount_micros / 1000000 : 0,
          currency: 'USD',
          type: 'DAILY'
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
      const operations = [{
        remove: `customers/${this.config.customerId}/campaigns/${campaignId}`
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/campaigns:mutate`,
        { operations }
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
    return this.updateCampaignStatus(campaignId, 'ENABLED');
  }
  
  private async updateCampaignStatus(campaignId: string, status: string): Promise<ApiResponse<boolean>> {
    try {
      const operations = [{
        update: {
          resource_name: `customers/${this.config.customerId}/campaigns/${campaignId}`,
          status
        },
        update_mask: {
          paths: ['status']
        }
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/campaigns:mutate`,
        { operations }
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
  
  // Creative Management (Ad Groups and Ads)
  async uploadCreative(creative: Creative): Promise<ApiResponse<Creative>> {
    try {
      // In Google Ads, we need to create an ad group first, then an ad
      const adGroupResponse = await this.createAdGroup(creative);
      if (!adGroupResponse.success) {
        return adGroupResponse as any;
      }
      
      // Create the ad
      const adData: any = {
        ad_group: adGroupResponse.data.resourceName,
        status: 'ENABLED',
        ad: {}
      };
      
      // Handle different creative types
      if (creative.type === 'TEXT') {
        adData.ad.text_ad = {
          headline: creative.assets.text?.headline,
          description1: creative.assets.text?.description?.substring(0, 90),
          description2: creative.assets.text?.description?.substring(90, 180),
          display_url: creative.destinationUrl
        };
      } else if (creative.type === 'IMAGE') {
        // For image ads, we'd need to upload the image first
        adData.ad.image_ad = {
          image: 'IMAGE_ASSET_RESOURCE_NAME', // Would be from image upload
          name: creative.name
        };
      }
      
      const operations = [{
        create: adData
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/adGroupAds:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      const createdAdResource = response.data.results[0].resourceName;
      const adId = createdAdResource.split('/').pop();
      
      const uploadedCreative: Creative = {
        id: adId,
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
  
  private async createAdGroup(creative: Creative): Promise<ApiResponse<{ resourceName: string }>> {
    try {
      // This is simplified - in practice, you'd want to specify which campaign
      const adGroupData = {
        name: `AdGroup_${creative.name}_${Date.now()}`,
        campaign: `customers/${this.config.customerId}/campaigns/CAMPAIGN_ID`, // Would need campaign ID
        status: 'ENABLED',
        type: 'SEARCH_STANDARD',
        cpc_bid_micros: 1000000 // $1.00 default bid
      };
      
      const operations = [{
        create: adGroupData
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/adGroups:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      return {
        success: true,
        data: {
          resourceName: response.data.results[0].resourceName
        }
      };
      
    } catch (error) {
      this.logger.error('Failed to create ad group', error);
      return {
        success: false,
        error: {
          code: 'ADGROUP_CREATION_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async updateCreative(creativeId: string, updates: Partial<Creative>): Promise<ApiResponse<Creative>> {
    try {
      const updateData: any = {
        resource_name: `customers/${this.config.customerId}/adGroupAds/${creativeId}`
      };
      
      if (updates.name) {
        updateData.ad = { name: sanitizeInput(updates.name) };
      }
      
      const operations = [{
        update: updateData,
        update_mask: {
          paths: ['ad.name'] // Simplified - would need to handle different ad types
        }
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/adGroupAds:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      return this.getCreative(creativeId);
      
    } catch (error) {
      this.logger.error('Failed to update creative', error);
      return {
        success: false,
        error: {
          code: 'CREATIVE_UPDATE_FAILED',
          message: (error as Error).message
        }
      };
    }
  }
  
  async getCreative(creativeId: string): Promise<ApiResponse<Creative>> {
    try {
      const query = `
        SELECT 
          ad_group_ad.ad.id,
          ad_group_ad.ad.name,
          ad_group_ad.ad.text_ad.headline,
          ad_group_ad.ad.text_ad.description1,
          ad_group_ad.ad.text_ad.display_url
        FROM ad_group_ad 
        WHERE ad_group_ad.ad.id = ${creativeId}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success || !response.data.results?.length) {
        return {
          success: false,
          error: {
            code: 'CREATIVE_NOT_FOUND',
            message: `Creative ${creativeId} not found`
          }
        };
      }
      
      const googleAd = response.data.results[0].adGroupAd.ad;
      const creative: Creative = {
        id: googleAd.id,
        name: googleAd.name || `Ad ${googleAd.id}`,
        type: googleAd.text_ad ? 'TEXT' : 'IMAGE',
        assets: {
          text: googleAd.text_ad ? {
            headline: googleAd.text_ad.headline,
            description: googleAd.text_ad.description1
          } : undefined
        },
        destinationUrl: googleAd.text_ad?.display_url
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
      const query = `
        SELECT 
          ad_group_ad.ad.id,
          ad_group_ad.ad.name,
          ad_group_ad.ad.text_ad.headline,
          ad_group_ad.ad.text_ad.description1,
          ad_group_ad.ad.text_ad.display_url
        FROM ad_group_ad 
        LIMIT ${filters?.limit || 25}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success) {
        return response;
      }
      
      const creatives: Creative[] = (response.data.results || []).map((result: any) => {
        const googleAd = result.adGroupAd.ad;
        return {
          id: googleAd.id,
          name: googleAd.name || `Ad ${googleAd.id}`,
          type: googleAd.text_ad ? 'TEXT' : 'IMAGE',
          assets: {
            text: googleAd.text_ad ? {
              headline: googleAd.text_ad.headline,
              description: googleAd.text_ad.description1
            } : undefined
          },
          destinationUrl: googleAd.text_ad?.display_url
        } as Creative;
      });
      
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
      const operations = [{
        remove: `customers/${this.config.customerId}/adGroupAds/${creativeId}`
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/adGroupAds:mutate`,
        { operations }
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
        'impressions', 'clicks', 'cost_micros', 'conversions',
        'ctr', 'average_cpc', 'average_cpm'
      ];
      
      const selectedMetrics = metrics || defaultMetrics;
      
      const query = `
        SELECT 
          campaign.id,
          campaign.name,
          ${selectedMetrics.join(', ')}
        FROM campaign 
        WHERE campaign.id = ${campaignId}
        AND segments.date BETWEEN '${dateRange.start}' AND '${dateRange.end}'
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success) {
        return response;
      }
      
      // Aggregate metrics across all date segments
      const results = response.data.results || [];
      const aggregated = results.reduce((acc: any, result: any) => {
        acc.impressions = (acc.impressions || 0) + parseInt(result.metrics?.impressions || '0');
        acc.clicks = (acc.clicks || 0) + parseInt(result.metrics?.clicks || '0');
        acc.cost_micros = (acc.cost_micros || 0) + parseInt(result.metrics?.cost_micros || '0');
        acc.conversions = (acc.conversions || 0) + parseFloat(result.metrics?.conversions || '0');
        acc.campaign = result.campaign;
        return acc;
      }, {});
      
      const performance: CampaignPerformance = {
        campaignId,
        campaignName: aggregated.campaign?.name || `Campaign ${campaignId}`,
        impressions: aggregated.impressions || 0,
        clicks: aggregated.clicks || 0,
        conversions: aggregated.conversions || 0,
        spend: (aggregated.cost_micros || 0) / 1000000,
        ctr: aggregated.impressions > 0 ? (aggregated.clicks / aggregated.impressions) * 100 : 0,
        cpc: aggregated.clicks > 0 ? (aggregated.cost_micros / 1000000) / aggregated.clicks : 0,
        cpm: aggregated.impressions > 0 ? (aggregated.cost_micros / 1000000) / (aggregated.impressions / 1000) : 0,
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
        return campaigns as any;
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
  
  // Audience Management (Customer Lists)
  async createAudience(audience: Audience): Promise<ApiResponse<Audience>> {
    try {
      const audienceData = {
        name: sanitizeInput(audience.name),
        description: audience.description,
        membership_life_span: audience.retentionDays || 180
      };
      
      const operations = [{
        create: audienceData
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/userLists:mutate`,
        { operations }
      );
      
      if (!response.success) {
        return response;
      }
      
      const createdAudienceResource = response.data.results[0].resourceName;
      const audienceId = createdAudienceResource.split('/').pop();
      
      const createdAudience: Audience = {
        id: audienceId,
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
      const query = `
        SELECT 
          user_list.id,
          user_list.name,
          user_list.description,
          user_list.size_for_display,
          user_list.membership_life_span
        FROM user_list 
        WHERE user_list.id = ${audienceId}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success || !response.data.results?.length) {
        return {
          success: false,
          error: {
            code: 'AUDIENCE_NOT_FOUND',
            message: `Audience ${audienceId} not found`
          }
        };
      }
      
      const googleAudience = response.data.results[0].userList;
      const audience: Audience = {
        id: googleAudience.id,
        name: googleAudience.name,
        type: 'CUSTOM',
        description: googleAudience.description,
        size: googleAudience.size_for_display,
        retentionDays: googleAudience.membership_life_span
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
      const query = `
        SELECT 
          user_list.id,
          user_list.name,
          user_list.description,
          user_list.size_for_display
        FROM user_list 
        LIMIT ${filters?.limit || 25}
      `;
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/googleAds:search`,
        { query }
      );
      
      if (!response.success) {
        return response;
      }
      
      const audiences: Audience[] = (response.data.results || []).map((result: any) => ({
        id: result.userList.id,
        name: result.userList.name,
        type: 'CUSTOM' as const,
        description: result.userList.description,
        size: result.userList.size_for_display
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
      const operations = [{
        remove: `customers/${this.config.customerId}/userLists/${audienceId}`
      }];
      
      const response = await this.makeApiRequest<any>(
        'POST',
        `${this.baseUrl}/customers/${this.config.customerId}/userLists:mutate`,
        { operations }
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
      // Google Ads doesn't have a direct content validation API, but we can do basic checks
      const validation: ContentPolicyValidation = {
        valid: true,
        errors: [],
        warnings: [],
        violatedPolicies: [],
        suggestedChanges: []
      };
      
      // Basic text validation for Google Ads policies
      if (creative.assets.text) {
        const { headline, description } = creative.assets.text;
        
        // Check for prohibited content
        const prohibitedTerms = ['free', 'guaranteed', 'miracle', 'secret'];
        const text = `${headline} ${description}`.toLowerCase();
        
        for (const term of prohibitedTerms) {
          if (text.includes(term)) {
            validation.warnings.push(`Content contains term that may require substantiation: "${term}"`);
            validation.suggestedChanges.push(`Ensure claims about "${term}" are substantiated`);
          }
        }
        
        // Check character limits for text ads
        if (headline && headline.length > 30) {
          validation.errors.push('Headline must be 30 characters or less for text ads');
        }
        
        if (description && description.length > 90) {
          validation.errors.push('Description must be 90 characters or less for text ads');
        }
        
        // Check for excessive capitalization
        const capsRatio = (headline + ' ' + description).replace(/[^A-Z]/g, '').length / 
                         (headline + ' ' + description).length;
        if (capsRatio > 0.3) {
          validation.warnings.push('Excessive capitalization may violate ad policies');
          validation.suggestedChanges.push('Use normal capitalization');
        }
      }
      
      validation.valid = validation.errors.length === 0;
      
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
  
  // Helper methods for mapping between GAELP and Google Ads formats
  private mapObjectiveToGoogleChannel(objective: string): string {
    const mapping: Record<string, string> = {
      'BRAND_AWARENESS': 'DISPLAY',
      'REACH': 'DISPLAY',
      'TRAFFIC': 'SEARCH',
      'ENGAGEMENT': 'DISPLAY',
      'APP_INSTALLS': 'DISPLAY',
      'VIDEO_VIEWS': 'VIDEO',
      'LEAD_GENERATION': 'SEARCH',
      'MESSAGES': 'DISPLAY',
      'CONVERSIONS': 'SEARCH',
      'SALES': 'SHOPPING'
    };
    return mapping[objective] || 'SEARCH';
  }
  
  private mapGoogleChannelToGAELP(channel: string): any {
    const mapping: Record<string, string> = {
      'SEARCH': 'TRAFFIC',
      'DISPLAY': 'BRAND_AWARENESS',
      'SHOPPING': 'SALES',
      'VIDEO': 'VIDEO_VIEWS'
    };
    return mapping[channel] || 'TRAFFIC';
  }
  
  private mapGoogleStatusToGAELP(status: string): any {
    const mapping: Record<string, string> = {
      'ENABLED': 'ACTIVE',
      'PAUSED': 'PAUSED',
      'REMOVED': 'DELETED'
    };
    return mapping[status] || 'PAUSED';
  }
  
  private mapGoogleBidStrategyToGAELP(campaign: any): any {
    if (campaign.target_cpa) {
      return {
        type: 'TARGET_CPA',
        amount: campaign.target_cpa.target_cpa_micros / 1000000
      };
    } else if (campaign.target_roas) {
      return {
        type: 'TARGET_ROAS',
        amount: campaign.target_roas.target_roas
      };
    } else if (campaign.manual_cpc) {
      return {
        type: 'MANUAL_CPC'
      };
    }
    
    return {
      type: 'AUTOMATIC'
    };
  }
  
  private formatGoogleDate(dateString: string): string | undefined {
    if (!dateString) return undefined;
    
    // Google Ads dates are in YYYYMMDD format
    const year = dateString.substring(0, 4);
    const month = dateString.substring(4, 6);
    const day = dateString.substring(6, 8);
    
    return `${year}-${month}-${day}`;
  }
}