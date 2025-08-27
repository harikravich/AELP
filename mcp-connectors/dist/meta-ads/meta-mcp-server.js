/**
 * Meta Ads MCP Server
 * Provides MCP interface for Meta advertising platform integration
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, ListToolsRequestSchema, ErrorCode, McpError, } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { MetaAdsConnector } from './meta-connector.js';
import { Logger, maskSensitiveData } from '../shared/utils.js';
export class MetaAdsMCPServer {
    server;
    connector = null;
    logger;
    constructor() {
        this.server = new Server({
            name: 'meta-ads-connector',
            version: '1.0.0',
        }, {
            capabilities: {
                tools: {},
            },
        });
        this.logger = new Logger('MetaAdsMCPServer');
        this.setupHandlers();
    }
    setupHandlers() {
        this.server.setRequestHandler(ListToolsRequestSchema, async () => {
            return {
                tools: [
                    // Connection Management
                    {
                        name: 'meta_connect',
                        description: 'Connect to Meta Ads API with credentials',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                appId: { type: 'string', description: 'Meta App ID' },
                                appSecret: { type: 'string', description: 'Meta App Secret' },
                                accessToken: { type: 'string', description: 'Access token' },
                                refreshToken: { type: 'string', description: 'Refresh token (optional)' },
                                accountId: { type: 'string', description: 'Ad Account ID' },
                                businessAccountId: { type: 'string', description: 'Business Account ID' },
                                rateLimitPerSecond: { type: 'number', default: 10 },
                                retryAttempts: { type: 'number', default: 3 },
                                timeout: { type: 'number', default: 30000 },
                                apiVersion: { type: 'string', default: 'v18.0' }
                            },
                            required: ['appId', 'appSecret', 'accessToken', 'accountId', 'businessAccountId']
                        }
                    },
                    {
                        name: 'meta_test_connection',
                        description: 'Test the connection to Meta Ads API',
                        inputSchema: { type: 'object', properties: {} }
                    },
                    {
                        name: 'meta_health_check',
                        description: 'Get health status of Meta Ads connector',
                        inputSchema: { type: 'object', properties: {} }
                    },
                    // Campaign Management
                    {
                        name: 'meta_create_campaign',
                        description: 'Create a new Meta advertising campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaign: {
                                    type: 'object',
                                    properties: {
                                        name: { type: 'string' },
                                        objective: {
                                            type: 'object',
                                            properties: {
                                                type: {
                                                    type: 'string',
                                                    enum: ['BRAND_AWARENESS', 'REACH', 'TRAFFIC', 'ENGAGEMENT', 'APP_INSTALLS',
                                                        'VIDEO_VIEWS', 'LEAD_GENERATION', 'MESSAGES', 'CONVERSIONS', 'SALES']
                                                }
                                            },
                                            required: ['type']
                                        },
                                        budget: {
                                            type: 'object',
                                            properties: {
                                                amount: { type: 'number' },
                                                currency: { type: 'string', default: 'USD' },
                                                type: { type: 'string', enum: ['DAILY', 'LIFETIME'] }
                                            },
                                            required: ['amount', 'type']
                                        },
                                        targeting: { type: 'object' },
                                        creatives: { type: 'array' },
                                        status: { type: 'string', enum: ['ACTIVE', 'PAUSED'], default: 'PAUSED' },
                                        startDate: { type: 'string', format: 'date' },
                                        endDate: { type: 'string', format: 'date' },
                                        bidStrategy: {
                                            type: 'object',
                                            properties: {
                                                type: { type: 'string', enum: ['AUTOMATIC', 'MANUAL_CPC', 'TARGET_CPA'] },
                                                amount: { type: 'number' }
                                            }
                                        }
                                    },
                                    required: ['name', 'objective', 'budget']
                                }
                            },
                            required: ['campaign']
                        }
                    },
                    {
                        name: 'meta_update_campaign',
                        description: 'Update an existing Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' },
                                updates: { type: 'object' }
                            },
                            required: ['campaignId', 'updates']
                        }
                    },
                    {
                        name: 'meta_get_campaign',
                        description: 'Get details of a specific Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' }
                            },
                            required: ['campaignId']
                        }
                    },
                    {
                        name: 'meta_list_campaigns',
                        description: 'List Meta campaigns with optional filters',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                filters: {
                                    type: 'object',
                                    properties: {
                                        status: { type: 'string' },
                                        limit: { type: 'number', default: 25 }
                                    }
                                }
                            }
                        }
                    },
                    {
                        name: 'meta_pause_campaign',
                        description: 'Pause a Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' }
                            },
                            required: ['campaignId']
                        }
                    },
                    {
                        name: 'meta_resume_campaign',
                        description: 'Resume a paused Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' }
                            },
                            required: ['campaignId']
                        }
                    },
                    {
                        name: 'meta_delete_campaign',
                        description: 'Delete a Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' }
                            },
                            required: ['campaignId']
                        }
                    },
                    // Creative Management
                    {
                        name: 'meta_upload_creative',
                        description: 'Upload a creative to Meta Ads',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                creative: {
                                    type: 'object',
                                    properties: {
                                        name: { type: 'string' },
                                        type: { type: 'string', enum: ['IMAGE', 'VIDEO', 'CAROUSEL', 'COLLECTION', 'TEXT'] },
                                        assets: {
                                            type: 'object',
                                            properties: {
                                                images: {
                                                    type: 'array',
                                                    items: {
                                                        type: 'object',
                                                        properties: {
                                                            url: { type: 'string' },
                                                            altText: { type: 'string' }
                                                        },
                                                        required: ['url']
                                                    }
                                                },
                                                videos: {
                                                    type: 'array',
                                                    items: {
                                                        type: 'object',
                                                        properties: {
                                                            url: { type: 'string' },
                                                            thumbnail: { type: 'string' }
                                                        },
                                                        required: ['url']
                                                    }
                                                },
                                                text: {
                                                    type: 'object',
                                                    properties: {
                                                        headline: { type: 'string' },
                                                        description: { type: 'string' },
                                                        callToAction: { type: 'string' }
                                                    }
                                                }
                                            }
                                        },
                                        destinationUrl: { type: 'string' }
                                    },
                                    required: ['name', 'type', 'assets']
                                }
                            },
                            required: ['creative']
                        }
                    },
                    {
                        name: 'meta_get_creative',
                        description: 'Get details of a specific Meta creative',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                creativeId: { type: 'string' }
                            },
                            required: ['creativeId']
                        }
                    },
                    {
                        name: 'meta_list_creatives',
                        description: 'List Meta creatives with optional filters',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                filters: {
                                    type: 'object',
                                    properties: {
                                        limit: { type: 'number', default: 25 }
                                    }
                                }
                            }
                        }
                    },
                    {
                        name: 'meta_delete_creative',
                        description: 'Delete a Meta creative',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                creativeId: { type: 'string' }
                            },
                            required: ['creativeId']
                        }
                    },
                    // Performance Monitoring
                    {
                        name: 'meta_get_campaign_performance',
                        description: 'Get performance metrics for a Meta campaign',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                campaignId: { type: 'string' },
                                dateRange: {
                                    type: 'object',
                                    properties: {
                                        start: { type: 'string', format: 'date' },
                                        end: { type: 'string', format: 'date' }
                                    },
                                    required: ['start', 'end']
                                },
                                metrics: {
                                    type: 'array',
                                    items: { type: 'string' }
                                }
                            },
                            required: ['campaignId', 'dateRange']
                        }
                    },
                    {
                        name: 'meta_get_account_performance',
                        description: 'Get performance metrics for all Meta campaigns',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                dateRange: {
                                    type: 'object',
                                    properties: {
                                        start: { type: 'string', format: 'date' },
                                        end: { type: 'string', format: 'date' }
                                    },
                                    required: ['start', 'end']
                                },
                                breakdown: { type: 'string' }
                            },
                            required: ['dateRange']
                        }
                    },
                    // Audience Management
                    {
                        name: 'meta_create_audience',
                        description: 'Create a custom audience in Meta',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                audience: {
                                    type: 'object',
                                    properties: {
                                        name: { type: 'string' },
                                        type: { type: 'string', enum: ['CUSTOM', 'LOOKALIKE', 'SAVED'] },
                                        description: { type: 'string' },
                                        retentionDays: { type: 'number', default: 180 },
                                        source: { type: 'object' }
                                    },
                                    required: ['name', 'type']
                                }
                            },
                            required: ['audience']
                        }
                    },
                    {
                        name: 'meta_get_audience',
                        description: 'Get details of a specific Meta audience',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                audienceId: { type: 'string' }
                            },
                            required: ['audienceId']
                        }
                    },
                    {
                        name: 'meta_list_audiences',
                        description: 'List Meta audiences with optional filters',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                filters: {
                                    type: 'object',
                                    properties: {
                                        limit: { type: 'number', default: 25 }
                                    }
                                }
                            }
                        }
                    },
                    {
                        name: 'meta_delete_audience',
                        description: 'Delete a Meta audience',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                audienceId: { type: 'string' }
                            },
                            required: ['audienceId']
                        }
                    },
                    // Content Validation
                    {
                        name: 'meta_validate_content',
                        description: 'Validate creative content against Meta policies',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                creative: { type: 'object' }
                            },
                            required: ['creative']
                        }
                    },
                    // Spending Control
                    {
                        name: 'meta_set_spending_limits',
                        description: 'Set spending limits for Meta campaigns',
                        inputSchema: {
                            type: 'object',
                            properties: {
                                limits: {
                                    type: 'object',
                                    properties: {
                                        dailyLimit: { type: 'number' },
                                        monthlyLimit: { type: 'number' },
                                        campaignLimit: { type: 'number' },
                                        currency: { type: 'string', default: 'USD' },
                                        alertThresholds: {
                                            type: 'object',
                                            properties: {
                                                warning: { type: 'number' },
                                                critical: { type: 'number' }
                                            },
                                            required: ['warning', 'critical']
                                        }
                                    },
                                    required: ['dailyLimit', 'monthlyLimit', 'alertThresholds']
                                }
                            },
                            required: ['limits']
                        }
                    },
                    {
                        name: 'meta_get_spending_status',
                        description: 'Get current spending status and alerts',
                        inputSchema: {
                            type: 'object',
                            properties: {}
                        }
                    }
                ],
            };
        });
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const { name, arguments: args } = request.params;
            try {
                this.logger.info(`Executing tool: ${name}`, maskSensitiveData(args));
                switch (name) {
                    case 'meta_connect':
                        return await this.handleConnect(args);
                    case 'meta_test_connection':
                        return await this.handleTestConnection();
                    case 'meta_health_check':
                        return await this.handleHealthCheck();
                    // Campaign Management
                    case 'meta_create_campaign':
                        return await this.handleCreateCampaign(args);
                    case 'meta_update_campaign':
                        return await this.handleUpdateCampaign(args);
                    case 'meta_get_campaign':
                        return await this.handleGetCampaign(args);
                    case 'meta_list_campaigns':
                        return await this.handleListCampaigns(args);
                    case 'meta_pause_campaign':
                        return await this.handlePauseCampaign(args);
                    case 'meta_resume_campaign':
                        return await this.handleResumeCampaign(args);
                    case 'meta_delete_campaign':
                        return await this.handleDeleteCampaign(args);
                    // Creative Management
                    case 'meta_upload_creative':
                        return await this.handleUploadCreative(args);
                    case 'meta_get_creative':
                        return await this.handleGetCreative(args);
                    case 'meta_list_creatives':
                        return await this.handleListCreatives(args);
                    case 'meta_delete_creative':
                        return await this.handleDeleteCreative(args);
                    // Performance Monitoring
                    case 'meta_get_campaign_performance':
                        return await this.handleGetCampaignPerformance(args);
                    case 'meta_get_account_performance':
                        return await this.handleGetAccountPerformance(args);
                    // Audience Management
                    case 'meta_create_audience':
                        return await this.handleCreateAudience(args);
                    case 'meta_get_audience':
                        return await this.handleGetAudience(args);
                    case 'meta_list_audiences':
                        return await this.handleListAudiences(args);
                    case 'meta_delete_audience':
                        return await this.handleDeleteAudience(args);
                    // Content Validation
                    case 'meta_validate_content':
                        return await this.handleValidateContent(args);
                    // Spending Control
                    case 'meta_set_spending_limits':
                        return await this.handleSetSpendingLimits(args);
                    case 'meta_get_spending_status':
                        return await this.handleGetSpendingStatus();
                    default:
                        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
                }
            }
            catch (error) {
                this.logger.error(`Tool execution failed: ${name}`, error);
                if (error instanceof McpError) {
                    throw error;
                }
                throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error.message}`);
            }
        });
    }
    async ensureConnected() {
        if (!this.connector) {
            throw new McpError(ErrorCode.InvalidRequest, 'Not connected to Meta Ads API. Call meta_connect first.');
        }
    }
    // Connection Management Handlers
    async handleConnect(config) {
        try {
            const connectorConfig = {
                apiKey: config.appId,
                accessToken: config.accessToken,
                refreshToken: config.refreshToken,
                clientId: config.appId,
                clientSecret: config.appSecret,
                accountId: config.accountId,
                businessAccountId: config.businessAccountId,
                rateLimitPerSecond: config.rateLimitPerSecond || 10,
                retryAttempts: config.retryAttempts || 3,
                timeout: config.timeout || 30000,
                apiVersion: config.apiVersion || 'v18.0'
            };
            this.connector = new MetaAdsConnector(connectorConfig);
            // Test the connection
            const testResult = await this.connector.testConnection();
            if (!testResult.success) {
                this.connector = null;
                throw new McpError(ErrorCode.InvalidRequest, `Connection failed: ${testResult.error?.message}`);
            }
            this.logger.info('Successfully connected to Meta Ads API');
            return {
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({
                            success: true,
                            message: 'Successfully connected to Meta Ads API',
                            accountId: config.accountId
                        })
                    }
                ]
            };
        }
        catch (error) {
            this.logger.error('Connection failed', error);
            throw new McpError(ErrorCode.InvalidRequest, `Connection failed: ${error.message}`);
        }
    }
    async handleTestConnection() {
        await this.ensureConnected();
        const result = await this.connector.testConnection();
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleHealthCheck() {
        await this.ensureConnected();
        const health = await this.connector.healthCheck();
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(health)
                }
            ]
        };
    }
    // Campaign Management Handlers
    async handleCreateCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.createCampaign(args.campaign);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleUpdateCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.updateCampaign(args.campaignId, args.updates);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleGetCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.getCampaign(args.campaignId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleListCampaigns(args) {
        await this.ensureConnected();
        const result = await this.connector.listCampaigns(args.filters);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handlePauseCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.pauseCampaign(args.campaignId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleResumeCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.resumeCampaign(args.campaignId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleDeleteCampaign(args) {
        await this.ensureConnected();
        const result = await this.connector.deleteCampaign(args.campaignId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    // Creative Management Handlers
    async handleUploadCreative(args) {
        await this.ensureConnected();
        const result = await this.connector.uploadCreative(args.creative);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleGetCreative(args) {
        await this.ensureConnected();
        const result = await this.connector.getCreative(args.creativeId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleListCreatives(args) {
        await this.ensureConnected();
        const result = await this.connector.listCreatives(args.filters);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleDeleteCreative(args) {
        await this.ensureConnected();
        const result = await this.connector.deleteCreative(args.creativeId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    // Performance Monitoring Handlers
    async handleGetCampaignPerformance(args) {
        await this.ensureConnected();
        const result = await this.connector.getCampaignPerformance(args.campaignId, args.dateRange, args.metrics);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleGetAccountPerformance(args) {
        await this.ensureConnected();
        const result = await this.connector.getAccountPerformance(args.dateRange, args.breakdown);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    // Audience Management Handlers
    async handleCreateAudience(args) {
        await this.ensureConnected();
        const result = await this.connector.createAudience(args.audience);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleGetAudience(args) {
        await this.ensureConnected();
        const result = await this.connector.getAudience(args.audienceId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleListAudiences(args) {
        await this.ensureConnected();
        const result = await this.connector.listAudiences(args.filters);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    async handleDeleteAudience(args) {
        await this.ensureConnected();
        const result = await this.connector.deleteAudience(args.audienceId);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    // Content Validation Handler
    async handleValidateContent(args) {
        await this.ensureConnected();
        const result = await this.connector.validateContent(args.creative);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(result)
                }
            ]
        };
    }
    // Spending Control Handlers
    async handleSetSpendingLimits(args) {
        await this.ensureConnected();
        this.connector.setSpendingLimits(args.limits);
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify({
                        success: true,
                        message: 'Spending limits updated successfully'
                    })
                }
            ]
        };
    }
    async handleGetSpendingStatus() {
        await this.ensureConnected();
        const status = await this.connector.getSpendingStatus();
        return {
            content: [
                {
                    type: 'text',
                    text: JSON.stringify({
                        success: true,
                        data: status
                    })
                }
            ]
        };
    }
    async run() {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        this.logger.info('Meta Ads MCP Server started');
    }
}
// For standalone execution
if (import.meta.url === `file://${process.argv[1]}`) {
    const server = new MetaAdsMCPServer();
    server.run().catch((error) => {
        console.error('Server failed to start:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=meta-mcp-server.js.map