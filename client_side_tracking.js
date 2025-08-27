/**
 * Client-Side Cross-Account Tracking for GAELP
 * 
 * This JavaScript module handles client-side tracking while ALWAYS having
 * server-side backup. NEVER relies only on client-side storage due to 
 * iOS privacy restrictions.
 * 
 * CRITICAL: Must work with server-side tracking system
 */

class GAELPClientTracker {
    constructor(options = {}) {
        this.serverEndpoint = options.serverEndpoint || 'https://track.teen-wellness-monitor.com/api';
        this.domain = options.domain || window.location.hostname;
        this.debug = options.debug || false;
        
        // Tracking parameters
        this.gaelpUid = null;
        this.sessionId = this.generateSessionId();
        this.isIOSDevice = this.detectIOSDevice();
        
        // Initialize immediately
        this.init();
    }
    
    /**
     * Initialize tracking system
     */
    init() {
        this.log('Initializing GAELP client tracker...');
        
        // Extract tracking parameters from URL
        this.extractTrackingParams();
        
        // Generate device fingerprint
        this.generateDeviceFingerprint().then(signature => {
            this.deviceSignature = signature;
            
            // Send page view event to server
            this.trackPageView();
            
            // Set up event listeners
            this.setupEventListeners();
            
            this.log('GAELP client tracker initialized', {
                gaelpUid: this.gaelpUid,
                isIOS: this.isIOSDevice,
                domain: this.domain
            });
        });
    }
    
    /**
     * Extract tracking parameters from URL
     */
    extractTrackingParams() {
        const urlParams = new URLSearchParams(window.location.search);
        
        // Extract GAELP parameters
        this.gaelpUid = urlParams.get('gaelp_uid') || this.getFromStorage('gaelp_uid');
        this.gaelpSource = urlParams.get('gaelp_source') || 'unknown';
        this.gaelpCampaign = urlParams.get('gaelp_campaign') || 'unknown';
        this.gaelpCreative = urlParams.get('gaelp_creative') || 'unknown';
        this.gaelpTimestamp = urlParams.get('gaelp_timestamp') || Date.now().toString();
        
        // Extract platform click IDs
        this.gclid = urlParams.get('gclid');
        this.fbclid = urlParams.get('fbclid');
        
        // Store in multiple places for persistence (iOS backup)
        if (this.gaelpUid) {
            this.setInMultipleStorages('gaelp_uid', this.gaelpUid);
            this.setInMultipleStorages('gaelp_data', JSON.stringify({
                source: this.gaelpSource,
                campaign: this.gaelpCampaign,
                creative: this.gaelpCreative,
                timestamp: this.gaelpTimestamp,
                gclid: this.gclid,
                fbclid: this.fbclid
            }));
        }
    }
    
    /**
     * Generate comprehensive device fingerprint
     */
    async generateDeviceFingerprint() {
        const signature = {
            userAgent: navigator.userAgent,
            language: navigator.language,
            platform: navigator.platform,
            screenResolution: `${screen.width}x${screen.height}`,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            
            // Enhanced fingerprinting for iOS
            touchSupport: 'ontouchstart' in window,
            deviceMemory: navigator.deviceMemory || null,
            hardwareConcurrency: navigator.hardwareConcurrency || null,
        };
        
        // Generate canvas fingerprint (if not blocked)
        try {
            signature.canvasFingerprint = await this.generateCanvasFingerprint();
        } catch (e) {
            this.log('Canvas fingerprinting blocked or failed');
        }
        
        // Generate audio fingerprint (if not blocked)
        try {
            signature.audioFingerprint = await this.generateAudioFingerprint();
        } catch (e) {
            this.log('Audio fingerprinting blocked or failed');
        }
        
        // Get available fonts
        try {
            signature.fontsHash = await this.generateFontsHash();
        } catch (e) {
            this.log('Fonts detection blocked or failed');
        }
        
        // Hash the signature for privacy
        signature.ipHash = await this.hashClientIP();
        signature.userAgentHash = await this.hash(signature.userAgent);
        
        return signature;
    }
    
    /**
     * Generate canvas fingerprint
     */
    async generateCanvasFingerprint() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Draw complex pattern
        ctx.textBaseline = 'top';
        ctx.font = '14px Arial';
        ctx.fillStyle = '#f60';
        ctx.fillRect(125, 1, 62, 20);
        ctx.fillStyle = '#069';
        ctx.fillText('GAELP Tracking 🔒', 2, 15);
        ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
        ctx.fillText('Cross-domain attribution', 4, 17);
        
        // Add geometric shapes
        ctx.arc(50, 50, 20, 0, Math.PI * 2, true);
        ctx.fill();
        
        return await this.hash(canvas.toDataURL());
    }
    
    /**
     * Generate audio fingerprint
     */
    async generateAudioFingerprint() {
        return new Promise((resolve, reject) => {
            try {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioCtx.createOscillator();
                const analyser = audioCtx.createAnalyser();
                const gainNode = audioCtx.createGain();
                const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
                
                oscillator.type = 'triangle';
                oscillator.frequency.value = 10000;
                
                gainNode.gain.value = 0;
                
                oscillator.connect(analyser);
                analyser.connect(scriptProcessor);
                scriptProcessor.connect(gainNode);
                gainNode.connect(audioCtx.destination);
                
                scriptProcessor.onaudioprocess = function(bins) {
                    const freqData = new Float32Array(analyser.frequencyBinCount);
                    analyser.getFloatFrequencyData(freqData);
                    
                    const fingerprint = Array.from(freqData).slice(0, 30).join('');
                    
                    oscillator.disconnect();
                    scriptProcessor.disconnect();
                    audioCtx.close();
                    
                    resolve(this.hash(fingerprint));
                }.bind(this);
                
                oscillator.start(0);
                
                setTimeout(() => {
                    reject(new Error('Audio fingerprint timeout'));
                }, 1000);
                
            } catch (e) {
                reject(e);
            }
        });
    }
    
    /**
     * Generate fonts hash
     */
    async generateFontsHash() {
        const testFonts = [
            'Arial', 'Arial Black', 'Arial Narrow', 'Arial Rounded MT Bold',
            'Bookman Old Style', 'Bradley Hand ITC', 'Century', 'Century Gothic',
            'Comic Sans MS', 'Courier', 'Courier New', 'Georgia', 'Gentium',
            'Helvetica', 'Helvetica Neue', 'Impact', 'King', 'Lucida Console',
            'Lalit', 'Modena', 'Monotype Corsiva', 'Papyrus', 'Tahoma', 'TeX',
            'Times', 'Times New Roman', 'Trebuchet MS', 'Verdana', 'Verona'
        ];
        
        const availableFonts = [];
        const testString = 'mmmmmmmmmmlli';
        const testSize = '72px';
        
        // Create baseline measurements
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = testSize + ' monospace';
        const baselineWidth = context.measureText(testString).width;
        
        // Test each font
        for (const font of testFonts) {
            context.font = testSize + ' ' + font + ', monospace';
            const width = context.measureText(testString).width;
            
            if (width !== baselineWidth) {
                availableFonts.push(font);
            }
        }
        
        return await this.hash(availableFonts.sort().join(','));
    }
    
    /**
     * Hash client IP (approximate based on timezone and other signals)
     */
    async hashClientIP() {
        // Since we can't get actual IP client-side, use timezone + connection info
        const ipApprox = [
            this.deviceSignature?.timezone || 'unknown',
            navigator.connection?.effectiveType || 'unknown',
            navigator.connection?.downlink || 'unknown'
        ].join('|');
        
        return await this.hash(ipApprox);
    }
    
    /**
     * Track page view event
     */
    async trackPageView() {
        const eventData = {
            event_type: 'page_view',
            gaelp_uid: this.gaelpUid,
            page_url: window.location.href,
            page_title: document.title,
            referrer: document.referrer,
            device_signature: this.deviceSignature,
            tracking_params: {
                source: this.gaelpSource,
                campaign: this.gaelpCampaign,
                creative: this.gaelpCreative,
                timestamp: this.gaelpTimestamp,
                gclid: this.gclid,
                fbclid: this.fbclid
            },
            client_info: {
                session_id: this.sessionId,
                is_ios: this.isIOSDevice,
                timestamp: Date.now()
            }
        };
        
        // Send to server (CRITICAL - never rely only on client-side)
        await this.sendToServer('/track', eventData);
        
        // Also send to GA4 directly (backup)
        if (window.gtag) {
            this.sendToGA4('page_view', {
                page_title: document.title,
                page_location: window.location.href,
                gaelp_uid: this.gaelpUid,
                gaelp_source: this.gaelpSource,
                gaelp_campaign: this.gaelpCampaign
            });
        }
        
        this.log('Page view tracked', eventData);
    }
    
    /**
     * Track conversion event
     */
    async trackConversion(conversionData) {
        const eventData = {
            event_type: 'conversion',
            gaelp_uid: this.gaelpUid,
            conversion_data: conversionData,
            device_signature: this.deviceSignature,
            tracking_params: {
                source: this.gaelpSource,
                campaign: this.gaelpCampaign,
                creative: this.gaelpCreative
            },
            client_info: {
                session_id: this.sessionId,
                timestamp: Date.now()
            }
        };
        
        // Send to server (CRITICAL)
        await this.sendToServer('/conversion', eventData);
        
        // Send to GA4
        if (window.gtag) {
            this.sendToGA4('purchase', {
                transaction_id: conversionData.transaction_id,
                value: conversionData.value,
                currency: conversionData.currency || 'USD',
                gaelp_uid: this.gaelpUid,
                gaelp_source: this.gaelpSource,
                gaelp_campaign: this.gaelpCampaign
            });
        }
        
        this.log('Conversion tracked', eventData);
    }
    
    /**
     * Prepare redirect to Aura with preserved tracking
     */
    prepareAuraRedirect() {
        if (!this.gaelpUid) {
            this.log('Warning: No GAELP UID for Aura redirect');
            return null;
        }
        
        const auraParams = new URLSearchParams({
            ref: 'gaelp',
            gaelp_uid: this.gaelpUid,
            gaelp_source: this.gaelpSource,
            gaelp_campaign: this.gaelpCampaign,
            gaelp_ts: Date.now().toString()
        });
        
        // Add platform click IDs
        if (this.gclid) auraParams.set('gclid', this.gclid);
        if (this.fbclid) auraParams.set('fbclid', this.fbclid);
        
        const auraUrl = `https://aura.com/parental-controls?${auraParams.toString()}`;
        
        this.log('Prepared Aura redirect URL', { url: auraUrl });
        return auraUrl;
    }
    
    /**
     * Setup event listeners for user interactions
     */
    setupEventListeners() {
        // Track CTA clicks
        document.addEventListener('click', (event) => {
            const target = event.target;
            
            // Check if it's a CTA button
            if (target.matches('.cta-button, .signup-button, [data-track-cta]')) {
                this.trackCTAClick(target);
            }
            
            // Check if it's an Aura redirect link
            if (target.matches('a[href*="aura.com"], [data-aura-redirect]')) {
                this.handleAuraRedirect(event, target);
            }
        });
        
        // Track form submissions
        document.addEventListener('submit', (event) => {
            const form = event.target;
            if (form.matches('form[data-track-form], .signup-form')) {
                this.trackFormSubmission(form);
            }
        });
        
        // Track page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'hidden') {
                this.trackPageExit();
            }
        });
        
        // Track scroll depth
        let maxScroll = 0;
        window.addEventListener('scroll', () => {
            const scrollPercent = Math.round(
                (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
            );
            
            if (scrollPercent > maxScroll && scrollPercent % 25 === 0) {
                maxScroll = scrollPercent;
                this.trackScrollDepth(scrollPercent);
            }
        });
    }
    
    /**
     * Track CTA button clicks
     */
    async trackCTAClick(element) {
        const eventData = {
            event_type: 'cta_click',
            gaelp_uid: this.gaelpUid,
            cta_text: element.textContent.trim(),
            cta_id: element.id || null,
            cta_class: element.className || null,
            page_url: window.location.href,
            device_signature: this.deviceSignature,
            timestamp: Date.now()
        };
        
        await this.sendToServer('/track', eventData);
        
        if (window.gtag) {
            this.sendToGA4('click', {
                link_text: element.textContent.trim(),
                link_url: element.href || window.location.href,
                gaelp_uid: this.gaelpUid
            });
        }
        
        this.log('CTA click tracked', eventData);
    }
    
    /**
     * Handle Aura redirect with preserved tracking
     */
    handleAuraRedirect(event, element) {
        event.preventDefault();
        
        // Get redirect URL with tracking
        const auraUrl = this.prepareAuraRedirect();
        
        if (auraUrl) {
            // Track the redirect attempt
            this.trackAuraRedirect();
            
            // Redirect after short delay to ensure tracking is sent
            setTimeout(() => {
                window.location.href = auraUrl;
            }, 100);
        } else {
            // Fallback to original URL if tracking fails
            window.location.href = element.href;
        }
    }
    
    /**
     * Track Aura redirect
     */
    async trackAuraRedirect() {
        const eventData = {
            event_type: 'aura_redirect',
            gaelp_uid: this.gaelpUid,
            from_url: window.location.href,
            device_signature: this.deviceSignature,
            timestamp: Date.now()
        };
        
        await this.sendToServer('/track', eventData);
        
        if (window.gtag) {
            this.sendToGA4('begin_checkout', {
                gaelp_uid: this.gaelpUid,
                gaelp_source: this.gaelpSource
            });
        }
        
        this.log('Aura redirect tracked', eventData);
    }
    
    /**
     * Track form submissions
     */
    async trackFormSubmission(form) {
        const formData = new FormData(form);
        const eventData = {
            event_type: 'form_submit',
            gaelp_uid: this.gaelpUid,
            form_id: form.id || null,
            form_action: form.action || null,
            form_fields: Object.fromEntries(formData),
            page_url: window.location.href,
            timestamp: Date.now()
        };
        
        // Remove sensitive data before sending
        delete eventData.form_fields.password;
        delete eventData.form_fields.email; // Hash instead
        if (eventData.form_fields.email) {
            eventData.form_fields.email_hash = await this.hash(eventData.form_fields.email);
        }
        
        await this.sendToServer('/track', eventData);
        
        this.log('Form submission tracked', eventData);
    }
    
    /**
     * Track scroll depth milestones
     */
    async trackScrollDepth(percent) {
        const eventData = {
            event_type: 'scroll_depth',
            gaelp_uid: this.gaelpUid,
            scroll_percent: percent,
            page_url: window.location.href,
            timestamp: Date.now()
        };
        
        await this.sendToServer('/track', eventData);
        
        if (window.gtag && percent >= 75) {
            this.sendToGA4('scroll', {
                percent_scrolled: percent,
                gaelp_uid: this.gaelpUid
            });
        }
    }
    
    /**
     * Track page exit
     */
    async trackPageExit() {
        const eventData = {
            event_type: 'page_exit',
            gaelp_uid: this.gaelpUid,
            session_duration: Date.now() - this.sessionStartTime,
            page_url: window.location.href,
            timestamp: Date.now()
        };
        
        // Use sendBeacon for reliable exit tracking
        this.sendBeacon('/track', eventData);
    }
    
    /**
     * Send event to server via AJAX
     */
    async sendToServer(endpoint, data) {
        try {
            const response = await fetch(`${this.serverEndpoint}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            this.log('Data sent to server successfully');
            
        } catch (error) {
            this.log('Failed to send to server', error);
            
            // Store in local queue for retry
            this.queueForRetry(endpoint, data);
        }
    }
    
    /**
     * Send event via beacon (for page exit)
     */
    sendBeacon(endpoint, data) {
        try {
            if (navigator.sendBeacon) {
                navigator.sendBeacon(
                    `${this.serverEndpoint}${endpoint}`,
                    JSON.stringify(data)
                );
            } else {
                // Fallback for older browsers
                this.sendToServer(endpoint, data);
            }
        } catch (error) {
            this.log('Beacon send failed', error);
        }
    }
    
    /**
     * Send to GA4 directly
     */
    sendToGA4(eventName, parameters) {
        try {
            if (window.gtag) {
                gtag('event', eventName, parameters);
                this.log('Sent to GA4', { eventName, parameters });
            }
        } catch (error) {
            this.log('GA4 send failed', error);
        }
    }
    
    /**
     * Queue failed requests for retry
     */
    queueForRetry(endpoint, data) {
        const queue = this.getFromStorage('retry_queue') || '[]';
        const queueArray = JSON.parse(queue);
        
        queueArray.push({
            endpoint,
            data,
            timestamp: Date.now()
        });
        
        // Keep only last 10 items
        const trimmedQueue = queueArray.slice(-10);
        this.setInMultipleStorages('retry_queue', JSON.stringify(trimmedQueue));
    }
    
    /**
     * Retry failed requests
     */
    async retryQueuedRequests() {
        const queue = this.getFromStorage('retry_queue') || '[]';
        const queueArray = JSON.parse(queue);
        
        for (const item of queueArray) {
            // Only retry items from last 24 hours
            if (Date.now() - item.timestamp < 24 * 60 * 60 * 1000) {
                try {
                    await this.sendToServer(item.endpoint, item.data);
                } catch (error) {
                    // Keep in queue if still failing
                    continue;
                }
            }
        }
        
        // Clear processed queue
        this.setInMultipleStorages('retry_queue', '[]');
    }
    
    /**
     * Store data in multiple locations for iOS compatibility
     */
    setInMultipleStorages(key, value) {
        try {
            // LocalStorage
            if (window.localStorage) {
                localStorage.setItem(`gaelp_${key}`, value);
            }
            
            // SessionStorage
            if (window.sessionStorage) {
                sessionStorage.setItem(`gaelp_${key}`, value);
            }
            
            // Cookie (as fallback)
            document.cookie = `gaelp_${key}=${encodeURIComponent(value)}; path=/; max-age=${30 * 24 * 60 * 60}`;
            
        } catch (error) {
            this.log('Storage failed', error);
        }
    }
    
    /**
     * Get data from storage (try multiple sources)
     */
    getFromStorage(key) {
        try {
            // Try localStorage first
            if (window.localStorage) {
                const value = localStorage.getItem(`gaelp_${key}`);
                if (value) return value;
            }
            
            // Try sessionStorage
            if (window.sessionStorage) {
                const value = sessionStorage.getItem(`gaelp_${key}`);
                if (value) return value;
            }
            
            // Try cookie as fallback
            const cookies = document.cookie.split(';');
            for (const cookie of cookies) {
                const [name, value] = cookie.trim().split('=');
                if (name === `gaelp_${key}`) {
                    return decodeURIComponent(value);
                }
            }
            
        } catch (error) {
            this.log('Storage read failed', error);
        }
        
        return null;
    }
    
    /**
     * Detect iOS device
     */
    detectIOSDevice() {
        return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
               (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    }
    
    /**
     * Generate session ID
     */
    generateSessionId() {
        return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Hash function for fingerprinting
     */
    async hash(input) {
        if (!crypto || !crypto.subtle) {
            // Fallback hash for older browsers
            let hash = 0;
            for (let i = 0; i < input.length; i++) {
                const char = input.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash; // Convert to 32bit integer
            }
            return Math.abs(hash).toString(36);
        }
        
        const encoder = new TextEncoder();
        const data = encoder.encode(input);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        return hashHex.substring(0, 16); // First 16 characters
    }
    
    /**
     * Log debug messages
     */
    log(message, data = null) {
        if (this.debug) {
            console.log(`[GAELP Tracker] ${message}`, data);
        }
    }
}

// Initialize tracker when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a tracking-enabled page
    if (document.querySelector('[data-gaelp-track]') || 
        window.location.search.includes('gaelp_uid')) {
        
        window.gaelpTracker = new GAELPClientTracker({
            serverEndpoint: 'https://track.teen-wellness-monitor.com/api',
            debug: window.location.search.includes('debug=1')
        });
        
        console.log('GAELP Client Tracker initialized');
    }
});

// Make available globally
window.GAELPClientTracker = GAELPClientTracker;