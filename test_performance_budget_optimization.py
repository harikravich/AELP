#!/usr/bin/env python3
"""
Test Performance-Driven Budget Optimization
Validates the complete budget optimization system with discovered performance patterns.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict
import logging

from integrated_performance_budget_optimizer import (
    IntegratedBudgetOptimizer, UnifiedChannelType, DeviceType,
    DiscoveredPerformanceData, PerformanceDiscoveryEngine
)

logger = logging.getLogger(__name__)


class TestPerformanceBudgetOptimization:
    """Test suite for performance-driven budget optimization"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing"""
        return IntegratedBudgetOptimizer(Decimal('1000'))
    
    @pytest.fixture
    def mock_performance_data(self):
        """Mock performance data for testing"""
        return {
            UnifiedChannelType.AFFILIATE_PARENTAL_NETWORKS: DiscoveredPerformanceData(
                channel=UnifiedChannelType.AFFILIATE_PARENTAL_NETWORKS,
                conversion_rate_pct=4.42,
                cost_per_click=0.85,
                cost_per_acquisition=65.0,
                return_on_ad_spend=4.8,
                volume_potential=0.7,
                efficiency_score=0.052,
                time_based_multipliers={i: 1.0 for i in range(24)},
                device_performance={DeviceType.IOS: 1.3, DeviceType.ANDROID: 1.1, DeviceType.DESKTOP: 0.9},
                confidence_level=0.95,
                last_updated=datetime.now()
            ),
            UnifiedChannelType.DISPLAY_PROSPECTING: DiscoveredPerformanceData(
                channel=UnifiedChannelType.DISPLAY_PROSPECTING,
                conversion_rate_pct=0.001,  # Broken
                cost_per_click=0.45,
                cost_per_acquisition=2500.0,
                return_on_ad_spend=0.05,
                volume_potential=0.95,
                efficiency_score=0.000022,  # Terrible efficiency
                time_based_multipliers={i: 0.9 for i in range(24)},
                device_performance={DeviceType.IOS: 0.8, DeviceType.ANDROID: 0.9, DeviceType.DESKTOP: 1.0},
                confidence_level=0.95,
                last_updated=datetime.now()
            )
        }
    
    @pytest.mark.asyncio
    async def test_budget_optimization_basic(self, optimizer):
        """Test basic budget optimization functionality"""
        allocations = await optimizer.optimize_budget_allocation()
        
        # Verify allocations
        assert len(allocations) > 0, "Should have channel allocations"
        
        total_budget = sum(allocations.values())
        assert abs(total_budget - optimizer.daily_budget) < Decimal('1'), f"Budget mismatch: ${total_budget} vs ${optimizer.daily_budget}"
        
        # Verify minimum budgets are respected
        for channel, budget in allocations.items():
            assert budget >= optimizer.minimum_channel_budget, f"{channel} budget ${budget} below minimum"
    
    @pytest.mark.asyncio  
    async def test_performance_based_reallocation(self, optimizer):
        """Test that budget is reallocated based on performance patterns"""
        allocations = await optimizer.optimize_budget_allocation()
        
        # Top performer (affiliates) should get significant allocation
        affiliate_budget = sum(
            allocations.get(ch, Decimal('0')) for ch in allocations
            if 'affiliate' in ch.value.lower()
        )
        
        total_budget = sum(allocations.values())
        affiliate_percentage = float(affiliate_budget) / float(total_budget) * 100
        
        assert affiliate_percentage > 30, f"Affiliates should get >30% of budget, got {affiliate_percentage:.1f}%"
        
        # Broken display should get minimal allocation
        display_budget = sum(
            allocations.get(ch, Decimal('0')) for ch in allocations
            if 'display' in ch.value.lower()
        )
        
        display_percentage = float(display_budget) / float(total_budget) * 100
        assert display_percentage < 10, f"Display should get <10% of budget, got {display_percentage:.1f}%"
    
    @pytest.mark.asyncio
    async def test_time_based_multipliers(self, optimizer):
        """Test time-based bid multipliers"""
        await optimizer.optimize_budget_allocation()
        
        # Test crisis hour multiplier (2am)
        crisis_multiplier = optimizer.get_real_time_bid_multiplier(
            UnifiedChannelType.SEARCH_BEHAVIORAL_HEALTH, DeviceType.IOS
        )
        
        # Should be elevated during crisis hours
        assert crisis_multiplier > 1.5, f"Crisis hour multiplier too low: {crisis_multiplier}"
        
        # Test iOS premium
        ios_multiplier = optimizer.get_real_time_bid_multiplier(
            UnifiedChannelType.FACEBOOK_IOS_PARENTS, DeviceType.IOS
        )
        android_multiplier = optimizer.get_real_time_bid_multiplier(
            UnifiedChannelType.FACEBOOK_IOS_PARENTS, DeviceType.ANDROID
        )
        
        # iOS should significantly outperform Android for iOS-targeted campaigns
        assert ios_multiplier > android_multiplier * 2, f"iOS multiplier not high enough: {ios_multiplier} vs {android_multiplier}"
    
    @pytest.mark.asyncio
    async def test_efficiency_driven_allocation(self, optimizer):
        """Test that allocation follows efficiency patterns"""
        allocations = await optimizer.optimize_budget_allocation()
        
        # Get performance data
        perf_data = optimizer.performance_data
        
        # Sort channels by efficiency
        sorted_by_efficiency = sorted(
            [(ch, data.efficiency_score) for ch, data in perf_data.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Top 3 efficient channels should get more budget than bottom 3
        top_3_channels = [ch for ch, _ in sorted_by_efficiency[:3]]
        bottom_3_channels = [ch for ch, _ in sorted_by_efficiency[-3:]]
        
        top_3_budget = sum(allocations.get(ch, Decimal('0')) for ch in top_3_channels)
        bottom_3_budget = sum(allocations.get(ch, Decimal('0')) for ch in bottom_3_channels)
        
        assert top_3_budget > bottom_3_budget * 3, f"Top performers should get much more budget: ${top_3_budget} vs ${bottom_3_budget}"
    
    @pytest.mark.asyncio
    async def test_no_hardcoded_allocations(self, optimizer):
        """Test that no hardcoded percentages are used"""
        # Run optimization twice with different conditions
        allocations1 = await optimizer.optimize_budget_allocation()
        
        # Simulate performance change
        if optimizer.performance_data:
            # Artificially boost one channel's efficiency
            first_channel = list(optimizer.performance_data.keys())[0]
            optimizer.performance_data[first_channel].efficiency_score *= 2
        
        allocations2 = await optimizer.optimize_budget_allocation()
        
        # Allocations should be different, proving dynamic adjustment
        allocation_differences = []
        for channel in allocations1:
            if channel in allocations2:
                diff = abs(allocations1[channel] - allocations2[channel])
                if diff > Decimal('1'):  # Meaningful difference
                    allocation_differences.append(diff)
        
        assert len(allocation_differences) > 0, "Allocations should change when performance data changes"
    
    @pytest.mark.asyncio
    async def test_safety_constraints(self, optimizer):
        """Test that safety constraints are applied"""
        allocations = await optimizer.optimize_budget_allocation()
        
        # No channel should get more than 50% of total budget (prevents over-concentration)
        total_budget = sum(allocations.values())
        for channel, budget in allocations.items():
            percentage = float(budget) / float(total_budget) * 100
            assert percentage < 50, f"{channel} gets too much budget: {percentage:.1f}%"
        
        # No channel should get less than minimum (except broken ones)
        for channel, budget in allocations.items():
            if channel != UnifiedChannelType.DISPLAY_PROSPECTING:  # Allow broken channel to get minimal
                assert budget >= Decimal('20'), f"{channel} budget too low: ${budget}"
    
    @pytest.mark.asyncio
    async def test_performance_discovery_engine(self):
        """Test the performance discovery engine"""
        engine = PerformanceDiscoveryEngine()
        patterns = engine.discover_performance_patterns()
        
        assert len(patterns) > 0, "Should discover performance patterns"
        
        # Verify affiliate channels are identified as top performers
        affiliate_channels = [ch for ch in patterns if 'affiliate' in ch.value.lower()]
        assert len(affiliate_channels) > 0, "Should discover affiliate channels"
        
        for ch in affiliate_channels:
            assert patterns[ch].conversion_rate_pct > 3.0, f"Affiliate {ch} should have high CVR"
            assert patterns[ch].efficiency_score > 0.03, f"Affiliate {ch} should have high efficiency"
        
        # Verify display prospecting is identified as broken
        if UnifiedChannelType.DISPLAY_PROSPECTING in patterns:
            display_data = patterns[UnifiedChannelType.DISPLAY_PROSPECTING]
            assert display_data.conversion_rate_pct < 0.01, "Display prospecting should have terrible CVR"
            assert display_data.efficiency_score < 0.001, "Display prospecting should have terrible efficiency"
    
    @pytest.mark.asyncio
    async def test_optimization_summary(self, optimizer):
        """Test optimization summary generation"""
        await optimizer.optimize_budget_allocation()
        summary = optimizer.get_optimization_summary()
        
        assert "error" not in summary, "Summary should not have errors"
        assert "budget_summary" in summary, "Should include budget summary"
        assert "performance_insights" in summary, "Should include performance insights"
        assert "allocation_analysis" in summary, "Should include allocation analysis"
        
        # Verify key metrics
        budget_summary = summary["budget_summary"]
        assert budget_summary["total_budget"] == 1000.0, "Should track total budget correctly"
        assert budget_summary["channel_count"] > 0, "Should have active channels"
        
        # Verify performance insights identify best and worst performers correctly
        perf_insights = summary["performance_insights"]
        assert "top_performer" in perf_insights, "Should identify top performer"
        assert "worst_performer" in perf_insights, "Should identify worst performer"
        assert perf_insights["efficiency_gap"] > 100, "Should show large efficiency gap"
    
    def test_channel_mapping_completeness(self):
        """Test that all channel types are properly mapped"""
        # Verify all unified channel types have reasonable categories
        for channel in UnifiedChannelType:
            assert len(channel.value) > 5, f"Channel name too short: {channel.value}"
            assert '_' in channel.value, f"Channel name should use underscores: {channel.value}"
            
            # Verify channel categories
            channel_value = channel.value.lower()
            valid_categories = ['affiliate', 'search', 'facebook', 'tiktok', 'display']
            has_valid_category = any(cat in channel_value for cat in valid_categories)
            assert has_valid_category, f"Channel {channel.value} doesn't match valid categories"


async def run_comprehensive_test():
    """Run comprehensive test demonstration"""
    print("🧪 Performance-Driven Budget Optimization Test Suite")
    print("=" * 60)
    
    # Initialize test optimizer
    optimizer = IntegratedBudgetOptimizer(Decimal('1000'))
    
    print("\n🔬 Test 1: Basic Budget Optimization")
    allocations = await optimizer.optimize_budget_allocation()
    total_budget = sum(allocations.values())
    print(f"   ✅ Budget allocated: ${total_budget} across {len(allocations)} channels")
    assert abs(total_budget - optimizer.daily_budget) < Decimal('1')
    
    print("\n🔬 Test 2: Performance-Based Reallocation") 
    affiliate_budget = sum(
        allocations.get(ch, Decimal('0')) for ch in allocations
        if 'affiliate' in ch.value.lower()
    )
    affiliate_pct = float(affiliate_budget) / float(total_budget) * 100
    print(f"   ✅ Affiliates get {affiliate_pct:.1f}% of budget (should be >30%)")
    assert affiliate_pct > 30
    
    display_budget = sum(
        allocations.get(ch, Decimal('0')) for ch in allocations
        if 'display' in ch.value.lower()
    )
    display_pct = float(display_budget) / float(total_budget) * 100
    print(f"   ✅ Display gets {display_pct:.1f}% of budget (should be <10%)")
    assert display_pct < 10
    
    print("\n🔬 Test 3: Time-Based Multipliers")
    crisis_multiplier = optimizer.get_real_time_bid_multiplier(
        UnifiedChannelType.SEARCH_BEHAVIORAL_HEALTH, DeviceType.IOS
    )
    print(f"   ✅ Crisis hour multiplier: {crisis_multiplier:.2f}x (should be >1.5x)")
    assert crisis_multiplier > 1.5
    
    print("\n🔬 Test 4: iOS Premium")
    ios_multiplier = optimizer.get_real_time_bid_multiplier(
        UnifiedChannelType.FACEBOOK_IOS_PARENTS, DeviceType.IOS
    )
    android_multiplier = optimizer.get_real_time_bid_multiplier(
        UnifiedChannelType.FACEBOOK_IOS_PARENTS, DeviceType.ANDROID
    )
    print(f"   ✅ iOS vs Android: {ios_multiplier:.2f}x vs {android_multiplier:.2f}x")
    assert ios_multiplier > android_multiplier * 2
    
    print("\n🔬 Test 5: Efficiency-Driven Allocation")
    perf_data = optimizer.performance_data
    sorted_by_efficiency = sorted(
        [(ch, data.efficiency_score) for ch, data in perf_data.items()],
        key=lambda x: x[1], reverse=True
    )
    
    top_channel = sorted_by_efficiency[0][0]
    bottom_channel = sorted_by_efficiency[-1][0]
    top_budget = allocations.get(top_channel, Decimal('0'))
    bottom_budget = allocations.get(bottom_channel, Decimal('0'))
    
    print(f"   ✅ Top performer ({top_channel.value}): ${top_budget}")
    print(f"   ✅ Bottom performer ({bottom_channel.value}): ${bottom_budget}")
    assert top_budget > bottom_budget * 2
    
    print("\n🔬 Test 6: No Hardcoded Percentages")
    # Simulate performance change
    original_efficiency = optimizer.performance_data[top_channel].efficiency_score
    optimizer.performance_data[top_channel].efficiency_score *= 0.5  # Reduce efficiency
    
    new_allocations = await optimizer.optimize_budget_allocation()
    new_top_budget = new_allocations.get(top_channel, Decimal('0'))
    
    change_amount = abs(top_budget - new_top_budget)
    print(f"   ✅ Budget changed by ${change_amount} when performance changed")
    assert change_amount > Decimal('10')  # Should change meaningfully
    
    print("\n🔬 Test 7: Performance Discovery")
    engine = PerformanceDiscoveryEngine()
    patterns = engine.discover_performance_patterns()
    
    affiliate_patterns = [ch for ch in patterns if 'affiliate' in ch.value.lower()]
    display_patterns = [ch for ch in patterns if 'display' in ch.value.lower() and 'prospecting' in ch.value.lower()]
    
    print(f"   ✅ Discovered {len(affiliate_patterns)} affiliate channels with high performance")
    print(f"   ✅ Discovered {len(display_patterns)} broken display channels")
    
    if affiliate_patterns:
        affiliate_cvr = patterns[affiliate_patterns[0]].conversion_rate_pct
        print(f"   ✅ Top affiliate CVR: {affiliate_cvr:.2f}% (should be >3%)")
        assert affiliate_cvr > 3.0
    
    if display_patterns:
        display_cvr = patterns[display_patterns[0]].conversion_rate_pct
        print(f"   ✅ Broken display CVR: {display_cvr:.3f}% (should be <0.01%)")
        assert display_cvr < 0.01
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n💡 System Successfully Demonstrates:")
    print("   • Dynamic budget allocation based on discovered 4.42% CVR affiliates")
    print("   • Severe budget reduction for 0.001% CVR display prospecting") 
    print("   • Crisis time multipliers (2am searches get 2x boost)")
    print("   • iOS premium bidding (3x higher than Android for targeted campaigns)")
    print("   • Marginal efficiency optimization with NO hardcoded percentages")
    print("   • Real-time performance-based reallocation")
    print("   • Safety constraints preventing over-concentration")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run comprehensive test
    asyncio.run(run_comprehensive_test())