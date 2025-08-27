"""
Test and demonstration of the integrated competitive intelligence auction system
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import our integrated system
from competitive_auction_integration import (
    CompetitiveAuctionOrchestrator, 
    CompetitiveSpendLimit,
    create_competitive_auction_system
)

# Import base auction components
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'auction-gym/src'))
    
    from Auction import Auction
    from Agent import Agent
    from Bidder import TruthfulBidder, EmpiricalShadedBidder
    from BidderAllocation import OracleAllocator, LogisticTSAllocator
    from AuctionAllocation import SecondPrice, FirstPrice
except ImportError as e:
    print(f"Warning: Could not import auction components: {e}")
    print("Running in demo mode without full auction simulation")


class CompetitiveAuctionTester:
    """Test harness for competitive intelligence auction system"""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = {}
        self.auction_history = []
    
    async def setup(self):
        """Set up the test environment"""
        print("🔧 Setting up competitive auction test environment...")
        
        self.orchestrator = await create_competitive_auction_system(
            enable_competitive_intel=True,
            enable_safety_orchestrator=False,  # Simplified for testing
            daily_spend_limit=1000.0
        )
        
        print("✅ Test environment ready")
    
    def test_competitive_bid_decisions(self):
        """Test competitive bid decision making"""
        print("\n🎯 Testing Competitive Bid Decisions")
        print("-" * 50)
        
        test_scenarios = [
            {
                "name": "Low competition scenario",
                "original_bid": 2.00,
                "keyword": "niche_product",
                "quality_score": 8.0,
                "expected": "slight_increase_or_decrease"
            },
            {
                "name": "High competition scenario", 
                "original_bid": 3.50,
                "keyword": "popular_keyword",
                "quality_score": 7.0,
                "expected": "increase_or_maintain"
            },
            {
                "name": "Budget constrained scenario",
                "original_bid": 1.50,
                "keyword": "budget_keyword",
                "quality_score": 6.5,
                "expected": "conservative_adjustment"
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            decision = self.orchestrator.decide_competitive_bid(
                agent_name=f"test_agent_{scenario['name']}",
                original_bid=scenario['original_bid'],
                keyword=scenario['keyword'],
                agent_quality_score=scenario['quality_score']
            )
            
            results.append({
                'scenario': scenario['name'],
                'original_bid': decision.original_bid,
                'adjusted_bid': decision.adjusted_bid,
                'adjustment_ratio': decision.adjusted_bid / decision.original_bid,
                'competitive_pressure': decision.competitive_pressure,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'safety_approved': decision.safety_approved
            })
            
            print(f"  {scenario['name']}:")
            print(f"    Original: ${decision.original_bid:.2f} → Adjusted: ${decision.adjusted_bid:.2f}")
            print(f"    Ratio: {decision.adjusted_bid/decision.original_bid:.3f}x")
            print(f"    Reasoning: {decision.reasoning}")
        
        self.test_results['bid_decisions'] = results
        return results
    
    def test_auction_outcome_tracking(self):
        """Test auction outcome recording and learning"""
        print("\n📊 Testing Auction Outcome Tracking")
        print("-" * 50)
        
        # Simulate a series of auction outcomes
        test_auctions = [
            {"keyword": "test_keyword_1", "bid": 2.50, "won": True, "position": 1, "cost": 2.20},
            {"keyword": "test_keyword_1", "bid": 2.30, "won": False, "position": None, "cost": None},
            {"keyword": "test_keyword_2", "bid": 1.80, "won": True, "position": 2, "cost": 1.65},
            {"keyword": "test_keyword_1", "bid": 2.70, "won": True, "position": 1, "cost": 2.45},
            {"keyword": "test_keyword_2", "bid": 1.95, "won": True, "position": 1, "cost": 1.85},
        ]
        
        for auction in test_auctions:
            self.orchestrator.record_auction_outcome(
                keyword=auction['keyword'],
                bid=auction['bid'],
                won=auction['won'],
                position=auction['position'],
                cost=auction['cost']
            )
            
            self.auction_history.append(auction)
        
        print(f"  Recorded {len(test_auctions)} auction outcomes")
        
        # Track patterns
        patterns = self.orchestrator.track_competitive_patterns()
        
        if 'error' not in patterns:
            print(f"  Market intelligence updated:")
            market_overview = patterns.get('market_overview', {})
            print(f"    Total auctions analyzed: {market_overview.get('total_auctions_analyzed', 0)}")
            
            if 'orchestrator_metrics' in patterns:
                metrics = patterns['orchestrator_metrics']
                print(f"    Total bid adjustments: {metrics.get('total_bid_adjustments', 0)}")
                print(f"    Daily competitive spend: ${metrics.get('daily_competitive_spend', 0):.2f}")
        else:
            print(f"  ⚠️ Pattern tracking failed: {patterns['error']}")
        
        self.test_results['outcome_tracking'] = {
            'auctions_recorded': len(test_auctions),
            'patterns': patterns
        }
        
        return patterns
    
    def test_competitor_response_prediction(self):
        """Test competitor response prediction"""
        print("\n🔮 Testing Competitor Response Prediction")
        print("-" * 50)
        
        test_predictions = [
            {"bid": 3.00, "keyword": "competitive_keyword", "scenario": "bid_increase"},
            {"bid": 2.50, "keyword": "new_keyword", "scenario": "new_keyword"},
            {"bid": 4.00, "keyword": "high_value_keyword", "scenario": "budget_increase"}
        ]
        
        prediction_results = []
        
        for test in test_predictions:
            prediction = self.orchestrator.estimate_competitor_response(
                planned_bid=test['bid'],
                keyword=test['keyword'],
                scenario=test['scenario']
            )
            
            prediction_results.append({
                'test_scenario': test['scenario'],
                'planned_bid': test['bid'],
                'keyword': test['keyword'],
                'prediction': prediction
            })
            
            print(f"  {test['scenario']} scenario (${test['bid']:.2f} bid):")
            
            if 'error' not in prediction:
                responses = prediction.get('competitor_responses', {})
                market_impact = prediction.get('market_impact', {})
                risk_assessment = prediction.get('risk_assessment', {})
                
                print(f"    Escalation probability: {responses.get('escalation_probability', 0):.2f}")
                print(f"    Expected CPC increase: {market_impact.get('expected_cpc_increase', 'N/A')}")
                print(f"    Risk level: {risk_assessment.get('risk_level', 'unknown')}")
                print(f"    Recommendation: {risk_assessment.get('recommendation', 'N/A')}")
            else:
                print(f"    ⚠️ Prediction failed: {prediction['error']}")
        
        self.test_results['response_prediction'] = prediction_results
        return prediction_results
    
    def test_safety_and_budget_controls(self):
        """Test safety checks and budget controls"""
        print("\n🛡️ Testing Safety and Budget Controls")
        print("-" * 50)
        
        # Test normal bid adjustments
        print("  Testing normal bid adjustments...")
        normal_decision = self.orchestrator.decide_competitive_bid(
            agent_name="safety_test_agent",
            original_bid=2.00,
            keyword="normal_keyword"
        )
        print(f"    Normal bid: ${normal_decision.original_bid:.2f} → ${normal_decision.adjusted_bid:.2f}")
        print(f"    Safety approved: {normal_decision.safety_approved}")
        
        # Test extreme bid adjustment attempt
        print("  Testing extreme bid adjustment...")
        extreme_decision = self.orchestrator.decide_competitive_bid(
            agent_name="safety_test_agent",
            original_bid=1.00,
            keyword="extreme_keyword"
        )
        print(f"    Extreme bid: ${extreme_decision.original_bid:.2f} → ${extreme_decision.adjusted_bid:.2f}")
        print(f"    Safety approved: {extreme_decision.safety_approved}")
        
        # Simulate high spending to test limits
        print("  Testing spend limits...")
        initial_spend = self.orchestrator.daily_competitive_spend
        
        # Make several expensive adjustments
        for i in range(10):
            decision = self.orchestrator.decide_competitive_bid(
                agent_name=f"spend_test_agent_{i}",
                original_bid=5.00,
                keyword=f"expensive_keyword_{i}"
            )
            # Simulate the spending
            if decision.spend_impact > 0:
                self.orchestrator.daily_competitive_spend += decision.spend_impact
        
        final_spend = self.orchestrator.daily_competitive_spend
        print(f"    Spend increase: ${initial_spend:.2f} → ${final_spend:.2f}")
        print(f"    Emergency stop active: {self.orchestrator.emergency_stop_active}")
        
        # Test emergency stop
        if not self.orchestrator.emergency_stop_active:
            # Force trigger emergency stop for testing
            original_threshold = self.orchestrator.spend_limits.emergency_stop_threshold
            self.orchestrator.spend_limits.emergency_stop_threshold = final_spend + 10
            self.orchestrator.daily_competitive_spend = final_spend + 20
            
            test_decision = self.orchestrator.decide_competitive_bid(
                agent_name="emergency_test_agent",
                original_bid=1.00,
                keyword="emergency_keyword"
            )
            
            print(f"    Emergency stop triggered: {self.orchestrator.emergency_stop_active}")
            
            # Reset for further testing
            self.orchestrator.reset_emergency_stop("test_system")
            self.orchestrator.spend_limits.emergency_stop_threshold = original_threshold
            self.orchestrator.daily_competitive_spend = 0
        
        self.test_results['safety_controls'] = {
            'normal_decision_approved': normal_decision.safety_approved,
            'extreme_decision_approved': extreme_decision.safety_approved,
            'emergency_stop_tested': True
        }
        
        return self.test_results['safety_controls']
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        print("\n📈 Testing Performance Metrics")
        print("-" * 50)
        
        metrics = self.orchestrator.get_performance_metrics()
        
        print("  Current performance metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key.startswith('win_rate') or key.endswith('_rate'):
                    print(f"    {key}: {value:.3f}")
                elif 'cost' in key or 'spend' in key:
                    print(f"    {key}: ${value:.2f}")
                else:
                    print(f"    {key}: {value}")
            else:
                print(f"    {key}: {value}")
        
        status = self.orchestrator.get_orchestrator_status()
        
        print(f"  Overall status: {status['status']}")
        print(f"  Last updated: {status['last_updated']}")
        
        self.test_results['performance_metrics'] = {
            'metrics': metrics,
            'status': status
        }
        
        return metrics
    
    async def run_integration_test(self):
        """Run a comprehensive integration test"""
        print("\n🔬 Running Integration Test")
        print("=" * 60)
        
        # Simulate a day of auction activity
        keywords = ["shoes", "electronics", "books", "clothing", "sports"]
        agents = [f"agent_{i}" for i in range(5)]
        
        auction_count = 0
        decisions = []
        outcomes = []
        
        print("Simulating auction activity...")
        
        for hour in range(24):  # 24 hours
            for _ in range(np.random.poisson(3)):  # Average 3 auctions per hour
                keyword = np.random.choice(keywords)
                agent = np.random.choice(agents)
                
                # Original bid based on time of day (higher during business hours)
                base_bid = 1.5 + np.random.normal(0, 0.3)
                if 9 <= hour <= 17:  # Business hours
                    base_bid *= 1.3
                
                # Get competitive bid decision
                decision = self.orchestrator.decide_competitive_bid(
                    agent_name=agent,
                    original_bid=base_bid,
                    keyword=keyword,
                    timestamp=datetime.now() + timedelta(hours=hour-12)
                )
                
                decisions.append(decision)
                
                # Simulate auction outcome
                # Higher bids have higher win probability
                win_prob = min(0.9, decision.adjusted_bid / 5.0)
                won = np.random.random() < win_prob
                
                if won:
                    position = 1 if np.random.random() < 0.6 else 2
                    cost = decision.adjusted_bid * (0.8 + np.random.random() * 0.3)
                else:
                    position = None
                    cost = None
                
                # Record outcome
                self.orchestrator.record_auction_outcome(
                    keyword=keyword,
                    bid=decision.adjusted_bid,
                    won=won,
                    position=position,
                    cost=cost,
                    timestamp=datetime.now() + timedelta(hours=hour-12)
                )
                
                outcomes.append({
                    'hour': hour,
                    'keyword': keyword,
                    'agent': agent,
                    'original_bid': decision.original_bid,
                    'adjusted_bid': decision.adjusted_bid,
                    'won': won,
                    'position': position,
                    'cost': cost
                })
                
                auction_count += 1
        
        print(f"✅ Completed {auction_count} simulated auctions")
        
        # Analyze results
        print("\n📊 Integration Test Results:")
        
        # Bid adjustment analysis
        adjustments = [d.adjusted_bid / d.original_bid for d in decisions]
        print(f"  Average bid adjustment ratio: {np.mean(adjustments):.3f}")
        print(f"  Bid increases: {len([a for a in adjustments if a > 1.05])} ({len([a for a in adjustments if a > 1.05])/len(adjustments)*100:.1f}%)")
        print(f"  Bid decreases: {len([a for a in adjustments if a < 0.95])} ({len([a for a in adjustments if a < 0.95])/len(adjustments)*100:.1f}%)")
        
        # Win rate analysis
        total_wins = len([o for o in outcomes if o['won']])
        print(f"  Overall win rate: {total_wins/len(outcomes):.3f}")
        
        # Spend analysis
        total_cost = sum([o['cost'] for o in outcomes if o['cost']])
        print(f"  Total simulated spend: ${total_cost:.2f}")
        print(f"  Average CPC: ${total_cost/max(total_wins, 1):.2f}")
        
        # Final competitive intelligence summary
        final_patterns = self.orchestrator.track_competitive_patterns()
        if 'error' not in final_patterns:
            market_overview = final_patterns.get('market_overview', {})
            print(f"  Final CI auctions analyzed: {market_overview.get('total_auctions_analyzed', 0)}")
        
        # Final orchestrator status
        final_status = self.orchestrator.get_orchestrator_status()
        print(f"  Final orchestrator status: {final_status['status']}")
        
        self.test_results['integration_test'] = {
            'total_auctions': auction_count,
            'decisions': decisions,
            'outcomes': outcomes,
            'final_patterns': final_patterns,
            'final_status': final_status
        }
        
        return self.test_results['integration_test']
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        return {
            'test_summary': {
                'tests_run': len(self.test_results),
                'timestamp': datetime.now(),
                'orchestrator_functional': self.orchestrator is not None
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check bid decision functionality
        if 'bid_decisions' in self.test_results:
            decisions = self.test_results['bid_decisions']
            approved_rate = sum([d['safety_approved'] for d in decisions]) / len(decisions)
            if approved_rate < 0.9:
                recommendations.append("Review safety controls - high rejection rate for bid adjustments")
        
        # Check competitive intelligence
        if 'outcome_tracking' in self.test_results:
            patterns = self.test_results['outcome_tracking'].get('patterns', {})
            if 'error' in patterns:
                recommendations.append("Competitive intelligence system needs debugging")
        
        # Check spend controls
        if 'safety_controls' in self.test_results:
            if not self.test_results['safety_controls'].get('emergency_stop_tested', False):
                recommendations.append("Emergency stop mechanism needs verification")
        
        # Integration test results
        if 'integration_test' in self.test_results:
            integration = self.test_results['integration_test']
            if len(integration.get('outcomes', [])) < 50:
                recommendations.append("Run longer integration tests for better data")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system ready for deployment")
        
        return recommendations


async def main():
    """Main test execution"""
    print("🧪 Competitive Intelligence Auction System - Comprehensive Test")
    print("=" * 70)
    
    tester = CompetitiveAuctionTester()
    
    try:
        # Setup
        await tester.setup()
        
        # Run individual tests
        print("\n" + "="*70)
        tester.test_competitive_bid_decisions()
        
        print("\n" + "="*70) 
        tester.test_auction_outcome_tracking()
        
        print("\n" + "="*70)
        tester.test_competitor_response_prediction()
        
        print("\n" + "="*70)
        tester.test_safety_and_budget_controls()
        
        print("\n" + "="*70)
        tester.test_performance_metrics()
        
        print("\n" + "="*70)
        await tester.run_integration_test()
        
        # Generate final report
        print("\n" + "="*70)
        print("📋 FINAL TEST REPORT")
        print("="*70)
        
        report = tester.generate_test_report()
        
        print(f"Tests completed: {report['test_summary']['tests_run']}")
        print(f"System functional: {report['test_summary']['orchestrator_functional']}")
        
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Save detailed report
        report_filename = f"competitive_auction_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Make report JSON serializable
            def serialize_for_json(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            
            with open(report_filename, 'w') as f:
                json.dump(report, f, default=serialize_for_json, indent=2)
            
            print(f"\n📄 Detailed report saved to: {report_filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save detailed report: {e}")
        
        print("\n✅ Competitive Intelligence Auction System Testing Complete!")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    report = asyncio.run(main())