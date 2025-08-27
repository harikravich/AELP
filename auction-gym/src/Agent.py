import numpy as np

from BidderAllocation import PyTorchLogisticRegressionAllocator, OracleAllocator
from Impression import ImpressionOpportunity
from Models import sigmoid


class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, num_items, item_values, allocator, bidder, memory=0, 
                 use_competitive_intel=True, quality_score=7.0, daily_budget=1000.0):
        self.rng = rng
        self.name = name
        self.num_items = num_items

        # Value distribution
        self.item_values = item_values

        self.net_utility = .0
        self.gross_utility = .0

        self.logs = []

        self.allocator = allocator
        self.bidder = bidder

        self.memory = memory
        
        # Competitive intelligence settings
        self.use_competitive_intel = use_competitive_intel
        self.quality_score = quality_score
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        self.competitive_adjustments = []  # Track bid adjustments
        
        # Performance tracking for safety integration
        self.win_rate_history = []
        self.avg_cpc_history = []
        self.spend_history = []

    def select_item(self, context):
        # Estimate CTR for all items
        estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        # Pick the best item (according to TS)
        best_item = np.argmax(estim_values)

        # If we do Thompson Sampling, don't propagate the noisy bid amount but bid using the MAP estimate
        if type(self.allocator) == PyTorchLogisticRegressionAllocator and self.allocator.thompson_sampling:
            estim_CTRs_MAP = self.allocator.estimate_CTR(context, sample=False)
            return best_item, estim_CTRs_MAP[best_item]

        return best_item, estim_CTRs[best_item]

    def bid(self, context):
        # First, pick what item we want to choose
        best_item, estimated_CTR = self.select_item(context)

        # Sample value for this item
        value = self.item_values[best_item]

        # Get the bid
        bid = self.bidder.bid(value, context, estimated_CTR)

        # Log what we know so far
        self.logs.append(ImpressionOpportunity(context=context,
                                               item=best_item,
                                               estimated_CTR=estimated_CTR,
                                               value=value,
                                               bid=bid,
                                               # These will be filled out later
                                               best_expected_value=0.0,
                                               true_CTR=0.0,
                                               price=0.0,
                                               second_price=0.0,
                                               outcome=0,
                                               won=False))

        return bid, best_item

    def charge(self, price, second_price, outcome):
        self.logs[-1].set_price_outcome(price, second_price, outcome, won=True)
        last_value = self.logs[-1].value * outcome
        self.net_utility += (last_value - price)
        self.gross_utility += last_value
        
        # Track daily spend for budget management
        self.daily_spend += price
        self.spend_history.append(price)

    def set_price(self, price):
        self.logs[-1].set_price(price)

    def update(self, iteration, plot=False, figsize=(8,5), fontsize=14):
        # Gather relevant logs
        contexts = np.array(list(opp.context for opp in self.logs))
        items = np.array(list(opp.item for opp in self.logs))
        values = np.array(list(opp.value for opp in self.logs))
        bids = np.array(list(opp.bid for opp in self.logs))
        prices = np.array(list(opp.price for opp in self.logs))
        outcomes = np.array(list(opp.outcome for opp in self.logs))
        estimated_CTRs = np.array(list(opp.estimated_CTR for opp in self.logs))

        # Update response model with data from winning bids
        won_mask = np.array(list(opp.won for opp in self.logs))
        self.allocator.update(contexts[won_mask], items[won_mask], outcomes[won_mask], iteration, plot, figsize, fontsize, self.name)

        # Update bidding model with all data
        self.bidder.update(contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, self.name)
        
        # Update performance metrics for safety monitoring
        self._update_performance_metrics(won_mask, prices)
        
        # Reset daily spend counter (simplified daily reset)
        if iteration % 100 == 0:  # Reset every 100 iterations (simulating daily reset)
            self.daily_spend = 0.0

    def get_allocation_regret(self):
        ''' How much value am I missing out on due to suboptimal allocation? '''
        return np.sum(list(opp.best_expected_value - opp.true_CTR * opp.value for opp in self.logs))

    def get_estimation_regret(self):
        ''' How much am I overpaying due to over-estimation of the value? '''
        return np.sum(list(opp.estimated_CTR * opp.value - opp.true_CTR * opp.value for opp in self.logs))

    def get_overbid_regret(self):
        ''' How much am I overpaying because I could shade more? '''
        return np.sum(list((opp.price - opp.second_price) * opp.won for opp in self.logs))

    def get_underbid_regret(self):
        ''' How much have I lost because I could have shaded less? '''
        # The difference between the winning price and our bid -- for opportunities we lost, and where we could have won without overpaying
        # Important to mention that this assumes a first-price auction! i.e. the price is the winning bid
        return np.sum(list((opp.price - opp.bid) * (not opp.won) * (opp.price < (opp.true_CTR * opp.value)) for opp in self.logs))

    def get_CTR_RMSE(self):
        return np.sqrt(np.mean(list((opp.true_CTR - opp.estimated_CTR)**2 for opp in self.logs)))

    def get_CTR_bias(self):
        return np.mean(list((opp.estimated_CTR / opp.true_CTR) for opp in filter(lambda opp: opp.won, self.logs)))

    def clear_utility(self):
        self.net_utility = .0
        self.gross_utility = .0

    def _update_performance_metrics(self, won_mask, prices):
        """Update performance metrics for safety monitoring"""
        try:
            if len(won_mask) > 0:
                win_rate = np.mean(won_mask)
                self.win_rate_history.append(win_rate)
                
                if np.sum(won_mask) > 0:
                    avg_cpc = np.mean(prices[won_mask])
                    self.avg_cpc_history.append(avg_cpc)
                
                # Keep only recent history
                max_history = 100
                self.win_rate_history = self.win_rate_history[-max_history:]
                self.avg_cpc_history = self.avg_cpc_history[-max_history:]
                self.spend_history = self.spend_history[-max_history:]
        except Exception as e:
            print(f"Warning: Performance metrics update failed for {self.name}: {e}")
    
    def get_performance_metrics(self):
        """Get current performance metrics for monitoring"""
        return {
            'agent_name': self.name,
            'daily_spend': self.daily_spend,
            'daily_budget': self.daily_budget,
            'budget_utilization': self.daily_spend / self.daily_budget if self.daily_budget > 0 else 0,
            'recent_win_rate': np.mean(self.win_rate_history[-10:]) if self.win_rate_history else 0,
            'recent_avg_cpc': np.mean(self.avg_cpc_history[-10:]) if self.avg_cpc_history else 0,
            'quality_score': self.quality_score,
            'use_competitive_intel': self.use_competitive_intel,
            'total_auctions': len(self.logs)
        }
    
    def clear_logs(self):
        if not self.memory:
            self.logs = []
        else:
            self.logs = self.logs[-self.memory:]
        self.bidder.clear_logs(memory=self.memory)
        
        # Clear competitive adjustment history but keep some for analysis
        if len(self.competitive_adjustments) > 1000:
            self.competitive_adjustments = self.competitive_adjustments[-500:]

