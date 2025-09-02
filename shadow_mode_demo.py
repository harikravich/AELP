#!/usr/bin/env python3
"""
SHADOW MODE DEMONSTRATION
Shows shadow mode testing in action without spending real money
"""

import asyncio
import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/home/hariravichandran/AELP')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShadowModeDemo:
    """
    Demonstration of shadow mode testing capabilities
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        
        # Import required components
        from shadow_mode_state import DynamicEnrichedState, create_synthetic_state_for_testing
        self.DynamicEnrichedState = DynamicEnrichedState
        self.create_synthetic_state = create_synthetic_state_for_testing
        
        logger.info("Shadow Mode Demo initialized")
    
    def create_shadow_models(self):
        """Create different shadow models for comparison"""
        
        class ShadowModel:
            def __init__(self, name, config):
                self.name = name
                self.config = config
                self.decisions_made = 0
                self.total_spend = 0.0
                self.total_revenue = 0.0
                self.impressions = 0
                self.clicks = 0
                self.conversions = 0
            
            def make_decision(self, user_state, context):
                """Make a bidding decision based on model characteristics"""
                self.decisions_made += 1
                
                # Base bid calculation
                base_bid = 2.0 * self.config['bid_multiplier']
                
                # Adjust for user segment value
                segment_multiplier = 1.0 + (user_state.segment_cvr - 0.02) * 10
                base_bid *= segment_multiplier
                
                # Adjust for competition
                competition_level = context.get('competition_level', 0.5)
                if self.config['risk_tolerance'] == 'high':
                    base_bid *= (1.0 + competition_level * 0.8)
                elif self.config['risk_tolerance'] == 'low':
                    base_bid *= (1.0 + competition_level * 0.3)
                else:
                    base_bid *= (1.0 + competition_level * 0.5)
                
                # Add exploration noise
                if np.random.random() < self.config['exploration_rate']:
                    noise = np.random.normal(0, base_bid * 0.2)
                    base_bid += noise
                
                # Creative selection based on model preference
                if self.config['creative_strategy'] == 'conservative':
                    creative_id = np.random.randint(0, 20)  # Lower creative IDs
                elif self.config['creative_strategy'] == 'aggressive':
                    creative_id = np.random.randint(30, 50)  # Higher creative IDs
                else:
                    creative_id = np.random.randint(0, 50)
                
                # Channel selection
                channels = ['paid_search', 'display', 'social', 'email']
                if self.config['channel_preference'] == 'search_focused':
                    channel = np.random.choice(channels, p=[0.6, 0.2, 0.1, 0.1])
                elif self.config['channel_preference'] == 'display_focused':
                    channel = np.random.choice(channels, p=[0.1, 0.6, 0.2, 0.1])
                else:
                    channel = np.random.choice(channels)
                
                # Final bid amount
                bid_amount = max(0.5, min(15.0, base_bid))
                
                return {
                    'model_name': self.name,
                    'bid_amount': bid_amount,
                    'creative_id': creative_id,
                    'channel': channel,
                    'confidence': np.random.uniform(0.4, 0.9),
                    'timestamp': datetime.now()
                }
            
            def simulate_outcome(self, decision, user_state, context):
                """Simulate auction and user interaction outcomes"""
                
                # Simulate auction
                competition_bids = [
                    np.random.lognormal(0.7, 0.3) for _ in range(np.random.randint(3, 8))
                ]
                
                won_auction = decision['bid_amount'] > max(competition_bids)
                
                if won_auction:
                    self.impressions += 1
                    price_paid = max(competition_bids) + 0.01
                    self.total_spend += price_paid
                    
                    # Simulate click
                    base_ctr = 0.025
                    ctr_multiplier = 1.0
                    
                    # Adjust CTR based on creative strategy alignment
                    if user_state.segment_name == 'crisis_parent' and self.config['creative_strategy'] == 'aggressive':
                        ctr_multiplier *= 1.3
                    elif user_state.segment_name == 'researching_parent' and self.config['creative_strategy'] == 'conservative':
                        ctr_multiplier *= 1.2
                    
                    final_ctr = base_ctr * ctr_multiplier
                    clicked = np.random.random() < final_ctr
                    
                    if clicked:
                        self.clicks += 1
                        
                        # Simulate conversion
                        base_cvr = user_state.segment_cvr
                        cvr_multiplier = 1.0
                        
                        # Channel effectiveness
                        channel_multipliers = {
                            'paid_search': 1.2,
                            'email': 1.1,
                            'display': 0.8,
                            'social': 0.9
                        }
                        cvr_multiplier *= channel_multipliers.get(decision['channel'], 1.0)
                        
                        final_cvr = base_cvr * cvr_multiplier
                        converted = np.random.random() < final_cvr
                        
                        if converted:
                            self.conversions += 1
                            revenue = np.random.lognormal(4.6, 0.3)  # ~$100 average
                            self.total_revenue += revenue
                    
                    return {
                        'won_auction': won_auction,
                        'price_paid': price_paid,
                        'clicked': clicked,
                        'converted': converted if clicked else False,
                        'revenue': revenue if (clicked and converted) else 0.0
                    }
                else:
                    return {
                        'won_auction': False,
                        'price_paid': 0.0,
                        'clicked': False,
                        'converted': False,
                        'revenue': 0.0
                    }
            
            def get_metrics(self):
                """Get performance metrics"""
                return {
                    'decisions': self.decisions_made,
                    'impressions': self.impressions,
                    'clicks': self.clicks,
                    'conversions': self.conversions,
                    'spend': self.total_spend,
                    'revenue': self.total_revenue,
                    'win_rate': self.impressions / max(1, self.decisions_made),
                    'ctr': self.clicks / max(1, self.impressions),
                    'cvr': self.conversions / max(1, self.clicks),
                    'roas': self.total_revenue / max(0.01, self.total_spend),
                    'cost_per_click': self.total_spend / max(1, self.clicks),
                    'cost_per_conversion': self.total_spend / max(1, self.conversions)
                }
        
        # Define different model configurations
        models = {
            'production': ShadowModel('production', {
                'bid_multiplier': 1.0,
                'risk_tolerance': 'medium',
                'exploration_rate': 0.05,
                'creative_strategy': 'conservative',
                'channel_preference': 'balanced'
            }),
            
            'shadow_aggressive': ShadowModel('shadow_aggressive', {
                'bid_multiplier': 1.2,
                'risk_tolerance': 'high',
                'exploration_rate': 0.15,
                'creative_strategy': 'aggressive',
                'channel_preference': 'search_focused'
            }),
            
            'shadow_conservative': ShadowModel('shadow_conservative', {
                'bid_multiplier': 0.9,
                'risk_tolerance': 'low',
                'exploration_rate': 0.08,
                'creative_strategy': 'conservative',
                'channel_preference': 'display_focused'
            }),
            
            'baseline': ShadowModel('baseline', {
                'bid_multiplier': 0.8,
                'risk_tolerance': 'medium',
                'exploration_rate': 0.3,
                'creative_strategy': 'balanced',
                'channel_preference': 'balanced'
            })
        }
        
        return models
    
    def run_shadow_comparison(self, models, num_users=1000):
        """Run shadow testing with multiple users"""
        
        logger.info(f"Starting shadow testing with {len(models)} models and {num_users} users")
        
        # Track comparisons
        comparisons = []
        
        for user_id in range(num_users):
            # Generate user
            user_state = self.create_synthetic_state()
            context = {
                'competition_level': np.random.beta(2, 2),
                'avg_competitor_bid': np.random.lognormal(0.8, 0.3),
                'is_peak_hour': np.random.choice([True, False], p=[0.3, 0.7])
            }
            
            # Get decisions from all models
            decisions = {}
            outcomes = {}
            
            for model_name, model in models.items():
                decision = model.make_decision(user_state, context)
                outcome = model.simulate_outcome(decision, user_state, context)
                
                decisions[model_name] = decision
                outcomes[model_name] = outcome
            
            # Compare decisions
            if 'production' in decisions and 'shadow_aggressive' in decisions:
                prod_decision = decisions['production']
                shadow_decision = decisions['shadow_aggressive']
                
                bid_divergence = abs(shadow_decision['bid_amount'] - prod_decision['bid_amount']) / prod_decision['bid_amount']
                creative_divergence = prod_decision['creative_id'] != shadow_decision['creative_id']
                channel_divergence = prod_decision['channel'] != shadow_decision['channel']
                
                # Calculate value difference
                prod_value = outcomes['production']['revenue'] - outcomes['production']['price_paid']
                shadow_value = outcomes['shadow_aggressive']['revenue'] - outcomes['shadow_aggressive']['price_paid']
                
                comparisons.append({
                    'user_id': user_id,
                    'user_segment': user_state.segment_name,
                    'bid_divergence': bid_divergence,
                    'creative_divergence': creative_divergence,
                    'channel_divergence': channel_divergence,
                    'production_bid': prod_decision['bid_amount'],
                    'shadow_bid': shadow_decision['bid_amount'],
                    'production_value': prod_value,
                    'shadow_value': shadow_value,
                    'significant_divergence': bid_divergence > 0.2 or creative_divergence or channel_divergence
                })
        
        return models, comparisons
    
    def analyze_results(self, models, comparisons):
        """Analyze shadow testing results"""
        
        logger.info("Analyzing shadow testing results...")
        
        print("\n" + "="*80)
        print("SHADOW MODE TESTING RESULTS")
        print("="*80)
        
        # Model performance comparison
        print(f"\n{'MODEL PERFORMANCE':<60}")
        print("-" * 80)
        print(f"{'Model':<20} {'Decisions':<10} {'Win Rate':<10} {'CTR':<8} {'CVR':<8} {'ROAS':<8} {'CPC':<8}")
        print("-" * 80)
        
        for model_name, model in models.items():
            metrics = model.get_metrics()
            print(f"{model_name:<20} "
                  f"{metrics['decisions']:<10} "
                  f"{metrics['win_rate']:<10.3f} "
                  f"{metrics['ctr']:<8.3f} "
                  f"{metrics['cvr']:<8.3f} "
                  f"{metrics['roas']:<8.2f} "
                  f"${metrics['cost_per_click']:<7.2f}")
        
        # Comparison analysis
        if comparisons:
            print(f"\nCOMPARISON ANALYSIS (Production vs Shadow Aggressive):")
            print("-" * 60)
            
            total_comparisons = len(comparisons)
            significant_divergences = sum(1 for c in comparisons if c['significant_divergence'])
            
            avg_bid_divergence = np.mean([c['bid_divergence'] for c in comparisons])
            creative_divergence_rate = sum(1 for c in comparisons if c['creative_divergence']) / total_comparisons
            channel_divergence_rate = sum(1 for c in comparisons if c['channel_divergence']) / total_comparisons
            
            production_values = [c['production_value'] for c in comparisons]
            shadow_values = [c['shadow_value'] for c in comparisons]
            
            avg_production_value = np.mean(production_values)
            avg_shadow_value = np.mean(shadow_values)
            
            shadow_wins = sum(1 for i in range(len(production_values)) 
                            if shadow_values[i] > production_values[i])
            shadow_win_rate = shadow_wins / total_comparisons
            
            print(f"Total Comparisons: {total_comparisons}")
            print(f"Significant Divergences: {significant_divergences} ({significant_divergences/total_comparisons:.1%})")
            print(f"Average Bid Divergence: {avg_bid_divergence:.3f}")
            print(f"Creative Divergence Rate: {creative_divergence_rate:.1%}")
            print(f"Channel Divergence Rate: {channel_divergence_rate:.1%}")
            print(f"")
            print(f"Average Production Value: ${avg_production_value:.2f}")
            print(f"Average Shadow Value: ${avg_shadow_value:.2f}")
            print(f"Value Lift: {((avg_shadow_value - avg_production_value) / max(0.01, abs(avg_production_value))):.1%}")
            print(f"Shadow Win Rate: {shadow_win_rate:.1%}")
        
        # Segment breakdown
        segment_breakdown = {}
        for comp in comparisons:
            segment = comp['user_segment']
            if segment not in segment_breakdown:
                segment_breakdown[segment] = {
                    'count': 0, 
                    'shadow_wins': 0,
                    'avg_bid_divergence': 0,
                    'total_bid_divergence': 0
                }
            
            segment_breakdown[segment]['count'] += 1
            segment_breakdown[segment]['total_bid_divergence'] += comp['bid_divergence']
            
            if comp['shadow_value'] > comp['production_value']:
                segment_breakdown[segment]['shadow_wins'] += 1
        
        print(f"\nSEGMENT BREAKDOWN:")
        print("-" * 60)
        print(f"{'Segment':<20} {'Users':<8} {'Shadow Wins':<12} {'Avg Divergence':<15}")
        print("-" * 60)
        
        for segment, data in segment_breakdown.items():
            if data['count'] > 0:
                win_rate = data['shadow_wins'] / data['count']
                avg_divergence = data['total_bid_divergence'] / data['count']
                print(f"{segment:<20} {data['count']:<8} {win_rate:<12.1%} {avg_divergence:<15.3f}")
        
        print("\n" + "="*80)
        print("SHADOW TESTING INSIGHTS:")
        print("="*80)
        
        best_model = max(models.items(), key=lambda x: x[1].get_metrics()['roas'])
        print(f"🏆 Best Performing Model: {best_model[0]} (ROAS: {best_model[1].get_metrics()['roas']:.2f}x)")
        
        if avg_shadow_value > avg_production_value:
            print(f"✅ Shadow model shows {((avg_shadow_value - avg_production_value) / abs(avg_production_value)):.1%} value improvement")
        else:
            print(f"❌ Shadow model shows {((avg_production_value - avg_shadow_value) / abs(avg_production_value)):.1%} value decrease")
        
        if significant_divergences > total_comparisons * 0.3:
            print(f"⚠️  High divergence rate ({significant_divergences/total_comparisons:.1%}) - requires careful monitoring")
        else:
            print(f"✅ Acceptable divergence rate ({significant_divergences/total_comparisons:.1%})")
        
        return {
            'model_metrics': {name: model.get_metrics() for name, model in models.items()},
            'comparison_summary': {
                'total_comparisons': total_comparisons,
                'significant_divergences': significant_divergences,
                'avg_bid_divergence': avg_bid_divergence,
                'shadow_win_rate': shadow_win_rate,
                'value_lift': (avg_shadow_value - avg_production_value) / max(0.01, abs(avg_production_value))
            },
            'segment_breakdown': segment_breakdown
        }
    
    def save_results(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"shadow_mode_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_types(results)
        
        with open(filename, 'w') as f:
            json.dump(results_clean, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename

async def main():
    """Main demonstration function"""
    
    print("\n" + "="*80)
    print("GAELP SHADOW MODE TESTING DEMONSTRATION")
    print("="*80)
    print("This demonstrates shadow mode testing without spending real money")
    print("Multiple models run in parallel, decisions are compared")
    print("="*80)
    
    # Initialize demo
    demo = ShadowModeDemo()
    
    # Create shadow models
    logger.info("Creating shadow models...")
    models = demo.create_shadow_models()
    logger.info(f"Created {len(models)} shadow models: {list(models.keys())}")
    
    # Run shadow testing
    logger.info("Running shadow comparison...")
    start_time = time.time()
    
    models, comparisons = demo.run_shadow_comparison(models, num_users=2000)
    
    duration = time.time() - start_time
    logger.info(f"Shadow testing completed in {duration:.2f} seconds")
    
    # Analyze results
    results = demo.analyze_results(models, comparisons)
    
    # Save results
    filename = demo.save_results(results)
    
    print(f"\n🎉 SHADOW MODE DEMONSTRATION COMPLETED!")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"📊 Users tested: 2000")
    print(f"🤖 Models compared: {len(models)}")
    print(f"📈 Comparisons made: {len(comparisons)}")
    print(f"💾 Results saved: {filename}")
    
    print(f"\n💡 Key Benefits Demonstrated:")
    print(f"   ✅ No real money spent")
    print(f"   ✅ Parallel model comparison")
    print(f"   ✅ Bid divergence detection")
    print(f"   ✅ Performance prediction")
    print(f"   ✅ Statistical analysis")
    print(f"   ✅ Segment-specific insights")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)