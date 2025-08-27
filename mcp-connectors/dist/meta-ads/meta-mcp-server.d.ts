/**
 * Meta Ads MCP Server
 * Provides MCP interface for Meta advertising platform integration
 */
export declare class MetaAdsMCPServer {
    private server;
    private connector;
    private logger;
    constructor();
    private setupHandlers;
    private ensureConnected;
    private handleConnect;
    private handleTestConnection;
    private handleHealthCheck;
    private handleCreateCampaign;
    private handleUpdateCampaign;
    private handleGetCampaign;
    private handleListCampaigns;
    private handlePauseCampaign;
    private handleResumeCampaign;
    private handleDeleteCampaign;
    private handleUploadCreative;
    private handleGetCreative;
    private handleListCreatives;
    private handleDeleteCreative;
    private handleGetCampaignPerformance;
    private handleGetAccountPerformance;
    private handleCreateAudience;
    private handleGetAudience;
    private handleListAudiences;
    private handleDeleteAudience;
    private handleValidateContent;
    private handleSetSpendingLimits;
    private handleGetSpendingStatus;
    run(): Promise<void>;
}
//# sourceMappingURL=meta-mcp-server.d.ts.map