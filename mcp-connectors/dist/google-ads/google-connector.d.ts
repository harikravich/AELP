/**
 * Google Ads MCP Connector
 */
import { BaseConfig, ApiResponse, Campaign, CampaignPerformance, Creative, Audience, ContentPolicyValidation } from '../shared/types.js';
import { BaseAdConnector } from '../shared/base-connector.js';
interface GoogleAdsConfig extends BaseConfig {
    customerId: string;
    developerToken: string;
    loginCustomerId?: string;
    apiVersion?: string;
}
export declare class GoogleAdsConnector extends BaseAdConnector {
    private config;
    private baseUrl;
    constructor(config: GoogleAdsConfig);
    protected getPlatform(): 'GOOGLE';
    testConnection(): Promise<ApiResponse<boolean>>;
    protected makeApiRequest<T>(method: 'GET' | 'POST' | 'PUT' | 'DELETE', url: string, data?: any, headers?: Record<string, string>): Promise<ApiResponse<T>>;
    createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>>;
    private createCampaignBudget;
    updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>>;
    getCampaign(campaignId: string): Promise<ApiResponse<Campaign>>;
    listCampaigns(filters?: any): Promise<ApiResponse<Campaign[]>>;
    deleteCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    pauseCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    resumeCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    private updateCampaignStatus;
    uploadCreative(creative: Creative): Promise<ApiResponse<Creative>>;
    private createAdGroup;
    updateCreative(creativeId: string, updates: Partial<Creative>): Promise<ApiResponse<Creative>>;
    getCreative(creativeId: string): Promise<ApiResponse<Creative>>;
    listCreatives(filters?: any): Promise<ApiResponse<Creative[]>>;
    deleteCreative(creativeId: string): Promise<ApiResponse<boolean>>;
    getCampaignPerformance(campaignId: string, dateRange: {
        start: string;
        end: string;
    }, metrics?: string[]): Promise<ApiResponse<CampaignPerformance>>;
    getAccountPerformance(dateRange: {
        start: string;
        end: string;
    }, breakdown?: string): Promise<ApiResponse<CampaignPerformance[]>>;
    createAudience(audience: Audience): Promise<ApiResponse<Audience>>;
    getAudience(audienceId: string): Promise<ApiResponse<Audience>>;
    listAudiences(filters?: any): Promise<ApiResponse<Audience[]>>;
    deleteAudience(audienceId: string): Promise<ApiResponse<boolean>>;
    validateContent(creative: Creative): Promise<ApiResponse<ContentPolicyValidation>>;
    private mapObjectiveToGoogleChannel;
    private mapGoogleChannelToGAELP;
    private mapGoogleStatusToGAELP;
    private mapGoogleBidStrategyToGAELP;
    private formatGoogleDate;
}
export {};
//# sourceMappingURL=google-connector.d.ts.map