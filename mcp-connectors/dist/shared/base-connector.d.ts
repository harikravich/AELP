/**
 * Base connector class for advertising platform integrations
 */
import { BaseConfig, ApiResponse, Campaign, CampaignPerformance, Creative, Audience, ConnectorError, SpendingLimits, SpendingStatus, ContentPolicyValidation } from './types.js';
import { RateLimiter, Logger } from './utils.js';
export declare abstract class BaseAdConnector {
    protected config: BaseConfig;
    protected rateLimiter: RateLimiter;
    protected logger: Logger;
    protected spendingLimits?: SpendingLimits;
    constructor(config: BaseConfig, platform: string);
    abstract testConnection(): Promise<ApiResponse<boolean>>;
    abstract createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>>;
    abstract updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>>;
    abstract getCampaign(campaignId: string): Promise<ApiResponse<Campaign>>;
    abstract listCampaigns(filters?: any): Promise<ApiResponse<Campaign[]>>;
    abstract deleteCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    abstract pauseCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    abstract resumeCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
    abstract uploadCreative(creative: Creative): Promise<ApiResponse<Creative>>;
    abstract updateCreative(creativeId: string, updates: Partial<Creative>): Promise<ApiResponse<Creative>>;
    abstract getCreative(creativeId: string): Promise<ApiResponse<Creative>>;
    abstract listCreatives(filters?: any): Promise<ApiResponse<Creative[]>>;
    abstract deleteCreative(creativeId: string): Promise<ApiResponse<boolean>>;
    abstract getCampaignPerformance(campaignId: string, dateRange: {
        start: string;
        end: string;
    }, metrics?: string[]): Promise<ApiResponse<CampaignPerformance>>;
    abstract getAccountPerformance(dateRange: {
        start: string;
        end: string;
    }, breakdown?: string): Promise<ApiResponse<CampaignPerformance[]>>;
    abstract createAudience(audience: Audience): Promise<ApiResponse<Audience>>;
    abstract getAudience(audienceId: string): Promise<ApiResponse<Audience>>;
    abstract listAudiences(filters?: any): Promise<ApiResponse<Audience[]>>;
    abstract deleteAudience(audienceId: string): Promise<ApiResponse<boolean>>;
    abstract validateContent(creative: Creative): Promise<ApiResponse<ContentPolicyValidation>>;
    protected makeApiRequest<T>(method: 'GET' | 'POST' | 'PUT' | 'DELETE', url: string, data?: any, headers?: Record<string, string>): Promise<ApiResponse<T>>;
    protected createError(message: string, code: string, statusCode?: number, details?: any): ConnectorError;
    protected abstract getPlatform(): 'META' | 'GOOGLE';
    setSpendingLimits(limits: SpendingLimits): void;
    getSpendingStatus(): Promise<SpendingStatus | null>;
    protected checkSpendingLimits(additionalSpend: number): Promise<boolean>;
    healthCheck(): Promise<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        checks: {
            connection: boolean;
            rateLimits: boolean;
            spending: boolean;
        };
        details: any;
    }>;
}
//# sourceMappingURL=base-connector.d.ts.map