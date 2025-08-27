
function main() {
    // Display Campaign Placement Exclusions - Bot Filter
    var exclusions = [
  {
    "url": "suspicious-ads-network.com",
    "reason": "Bot score: 0.90",
    "sessions_saved": 45000
  },
  {
    "url": "fake-traffic-source.tk",
    "reason": "Bot score: 0.90",
    "sessions_saved": 30000
  },
  {
    "url": "bot-generated-content.ml",
    "reason": "Bot score: 0.90",
    "sessions_saved": 25000
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
