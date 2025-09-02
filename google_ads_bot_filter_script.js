
function main() {
    // Display Campaign Placement Exclusions - Bot Filter
    var exclusions = [
  {
    "url": "fraudulent-network-1.com",
    "reason": "Bot score: 0.90",
    "sessions_saved": 669
  },
  {
    "url": "bot-traffic-source.tk",
    "reason": "Bot score: 0.90",
    "sessions_saved": 478
  },
  {
    "url": "fake-content-network.ml",
    "reason": "Bot score: 0.90",
    "sessions_saved": 382
  },
  {
    "url": "suspicious-ads-platform.ga",
    "reason": "Bot score: 0.90",
    "sessions_saved": 382
  }
];
    
    var campaigns = AdsApp.campaigns()
        .withCondition("CampaignType = DISPLAY")
        .get();
    
    while (campaigns.hasNext()) {
        var campaign = campaigns.next();
        Logger.log("Processing campaign: " + campaign.getName());
        
        exclusions.forEach(function(exclusion) {
            try {
                campaign.createNegativeKeyword(exclusion.url);
                Logger.log("Excluded: " + exclusion.url);
            } catch (e) {
                Logger.log("Failed to exclude " + exclusion.url + ": " + e.message);
            }
        });
    }
    
    Logger.log("Bot filtering complete. Excluded " + exclusions.length + " placements.");
}
