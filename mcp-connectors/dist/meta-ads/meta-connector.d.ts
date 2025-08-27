/**
 * Meta (Facebook/Instagram) Ads MCP Connector
 */
import { BaseConfig, ApiResponse, Campaign, CampaignPerformance, Creative, Audience, ContentPolicyValidation } from '../shared/types.js';
import { BaseAdConnector } from '../shared/base-connector.js';
interface MetaConfig extends BaseConfig {
    appId: string;
    appSecret: string;
    businessAccountId: string;
    apiVersion?: string;
}
export declare class MetaAdsConnector extends BaseAdConnector {
    private config;
    private baseUrl;
    constructor(config: MetaConfig);
    protected getPlatform(): 'META';
    testConnection(): Promise<ApiResponse<boolean>>;
    createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>>;
    updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>>;
    getCampaign(campaignId: string): Promise<ApiResponse<Campaign>>;
    listCampaigns(filters?: any): Promise<ApiResponse<Campaign[]>>;
    deleteCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    pauseCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    resumeCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    private updateCampaignStatus;
    uploadCreative(creative: Creative): Promise<ApiResponse<Creative>>;
    private uploadImage;
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
    private extractConversions;
    createAudience(audience: Audience): Promise<ApiResponse<Audience>>;
    getAudience(audienceId: string): Promise<ApiResponse<Audience>>;
    listAudiences(filters?: any): Promise<ApiResponse<Audience[]>>;
    deleteAudience(audienceId: string): Promise<ApiResponse<boolean>>;
    validateContent(creative: Creative): Promise<ApiResponse<ContentPolicyValidation>>;
    private mapObjectiveToMeta;
    private mapMetaObjectiveToGAELP;
    private mapMetaStatusToGAELP;
    private mapMetaBidStrategyToGAELP;
    private mapAudienceTypeToMeta;
}
export {};
//# sourceMappingURL=meta-connector.d.ts.map