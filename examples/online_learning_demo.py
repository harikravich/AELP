"""
Online Learning Integration Demo

Demonstrates how to integrate the online learning system with the existing
GAELP training orchestrator to enable real-time learning while serving traffic.

This demo shows:
1. Setting up online learning with an existing agent
2. Real-time action selection with exploration/exploitation
3. Safety-constrained exploration with budget guardrails
4. Incremental model updates during live traffic
5. Performance monitoring and emergency fallback
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import GAELP components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_orchestrator.online_learner import OnlineLearner, OnlineLearnerConfig
from training_orchestrator.rl_agents.ppo_agent import PPOAgent, PPOConfig


class LiveTrafficSimulator:
    """Simulates live advertising traffic for testing online learning"""
    
    def __init__(self, base_performance: float = 0.6, volatility: float = 0.3):
        self.base_performance = base_performance
        self.volatility = volatility
        self.time_step = 0
        self.market_conditions = {
            "competition_level": 0.5,
            "seasonality_factor": 1.0,
            "trend_momentum": 0.0
        }
    
    def generate_state(self, previous_performance: float = None) -> Dict[str, Any]:
        """Generate realistic advertising state based on market conditions"""
        # Update market conditions over time
        self.time_step += 1
        self.market_conditions["competition_level"] += np.random.normal(0, 0.02)
        self.market_conditions["competition_level"] = np.clip(self.market_conditions["competition_level"], 0.1, 0.9)
        
        # Seasonal effects (simplified daily pattern)
        hour_of_day = (self.time_step % 24)
        seasonal_multiplier = 0.8 + 0.4 * np.sin(2 * np.pi * hour_of_day / 24)
        self.market_conditions["seasonality_factor"] = seasonal_multiplier
        
        # Budget tracking
        daily_budget = np.random.uniform(150, 300)
        budget_utilization = min(0.9, (self.time_step % 48) / 48.0)  # Gradual spend over 48 steps (2 days)
        
        state = {
            "market_context": {
                "competition_level": self.market_conditions["competition_level"],
                "seasonality_factor": self.market_conditions["seasonality_factor"],
                "trend_momentum": np.random.normal(0, 0.1),
                "market_volatility": self.volatility
            },
            "budget_constraints": {
                "daily_budget": daily_budget,
                "remaining_budget": daily_budget * (1 - budget_utilization),
                "budget_utilization": budget_utilization,
                "daily_spent": daily_budget * budget_utilization
            },
            "performance_history": {
                "avg_roas": previous_performance or np.random.uniform(0.5, 2.0),
                "avg_ctr": np.random.uniform(0.01, 0.04),
                "avg_conversion_rate": np.random.uniform(0.02, 0.08),
                "total_spend": np.random.uniform(1000, 10000),
                "total_revenue": np.random.uniform(2000, 15000)
            },
            "persona": {
                "demographics": {
                    "age_group": np.random.choice(["18-25", "25-35", "35-45", "45-55", "55-65", "65+"]),
                    "income": np.random.choice(["low", "medium", "high"])
                },
                "interests": np.random.choice(
                    [["technology", "finance"], ["entertainment", "sports"], ["health", "food"], ["travel", "fashion"]],
                )
            },
            "time_context": {
                "hour_of_day": hour_of_day,
                "day_of_week": (self.time_step // 24) % 7,
                "day_of_month": ((self.time_step // 24) % 30) + 1,
                "month": ((self.time_step // (24*30)) % 12) + 1
            },
            "previous_action": {
                "creative_type": "image",
                "budget": 100.0,
                "bid_strategy": "cpc"
            }
        }
        
        return state
    
    def simulate_campaign_outcome(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the outcome of a campaign action"""
        # Base performance affected by market conditions
        base_reward = self.base_performance * state["market_context"]["seasonality_factor"]
        
        # Action impact on performance
        budget_factor = min(1.5, action["budget"] / 100.0)  # Higher budget can improve performance
        bid_factor = min(1.3, action["bid_amount"] / 2.0)  # Higher bids can improve performance
        
        # Creative type impact
        creative_multiplier = {
            "image": 1.0,
            "video": 1.2,  # Video typically performs better
            "carousel": 1.1
        }.get(action["creative_type"], 1.0)
        
        # Audience size impact (diminishing returns)
        audience_factor = 0.5 + 0.5 * (1 - np.exp(-2 * action["audience_size"]))
        
        # Calculate expected reward
        expected_reward = base_reward * budget_factor * bid_factor * creative_multiplier * audience_factor
        
        # Add noise and market volatility
        noise = np.random.normal(0, self.volatility * state["market_context"]["market_volatility"])
        actual_reward = expected_reward + noise
        
        # Calculate metrics
        roas = max(0, actual_reward * 2.0)  # ROAS typically 2x reward
        ctr = max(0, actual_reward * 0.05)  # CTR correlated with reward
        conversions = max(0, int(action["budget"] * ctr * 0.1))  # Rough conversion estimate
        
        # Safety violation if performance is very poor
        safety_violation = actual_reward < -0.5 or roas < 0.3
        
        return {
            "reward": actual_reward,
            "roas": roas,
            "ctr": ctr,
            "conversions": conversions,
            "cost": action["budget"],
            "revenue": action["budget"] * roas,
            "success": actual_reward > 0.2,
            "safety_violation": safety_violation,
            "campaign_duration": np.random.uniform(6, 24),  # Hours
            "impressions": int(action["budget"] * 100 * audience_factor),
            "clicks": int(action["budget"] * 100 * audience_factor * ctr)
        }


class OnlineLearningDemo:
    """Comprehensive demo of online learning capabilities"""
    
    def __init__(self):
        self.traffic_simulator = LiveTrafficSimulator(base_performance=0.65, volatility=0.25)
        self.results_history = []
        self.performance_history = []
        
    async def setup_agent_and_learner(self) -> tuple[PPOAgent, OnlineLearner]:
        """Setup PPO agent with online learning capabilities"""
        # Configure PPO agent
        ppo_config = PPOConfig(
            state_dim=128,  # Large state space for rich ad campaign features
            action_dim=64,  # Complex action space for campaign optimization
            learning_rate=3e-4,
            batch_size=128,
            hidden_dims=[512, 512, 256],
            gamma=0.99,
            exploration_initial_eps=0.3,
            exploration_final_eps=0.05
        )
        
        # Create agent
        agent = PPOAgent(ppo_config, "demo_online_agent")
        logger.info(f"Created PPO agent with {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
        
        # Configure online learner
        online_config = OnlineLearnerConfig(
            # Thompson Sampling
            ts_prior_alpha=2.0,
            ts_prior_beta=2.0,
            ts_exploration_decay=0.998,
            min_exploration_rate=0.03,
            
            # Safety
            safety_threshold=0.7,
            max_budget_risk=0.15,
            safety_violation_limit=3,
            emergency_fallback=True,
            
            # Learning
            online_update_frequency=25,
            update_batch_size=64,
            min_episodes_before_update=10,
            learning_rate_schedule="adaptive",
            
            # Bandit arms with different strategies
            bandit_arms=["conservative", "balanced", "aggressive", "experimental", "video_focus", "mobile_optimized"],
            
            # Performance monitoring
            performance_window_size=100,
            log_interval=10
        )
        
        # Create online learner
        learner = OnlineLearner(agent, online_config)
        logger.info("Online learner initialized with 6 bandit arms")
        
        # Set baseline performance from some initial episodes
        baseline_episodes = []
        for _ in range(20):
            state = self.traffic_simulator.generate_state()
            action = await agent.select_action(state, deterministic=True)
            outcome = self.traffic_simulator.simulate_campaign_outcome(action, state)
            baseline_episodes.append({"reward": outcome["reward"]})
        
        learner.update_baseline_performance(baseline_episodes)
        logger.info(f"Baseline performance set to: {learner.baseline_performance:.3f}")
        
        return agent, learner
    
    async def run_online_learning_session(self, num_episodes: int = 500) -> List[Dict[str, Any]]:
        """Run online learning session with live traffic simulation"""
        agent, learner = await self.setup_agent_and_learner()
        
        logger.info(f"Starting online learning session for {num_episodes} episodes")
        
        session_results = []
        exploration_count = 0
        exploitation_count = 0
        
        for episode in range(num_episodes):
            episode_start_time = datetime.now()
            
            # Generate current market state
            prev_performance = session_results[-1]["outcome"]["reward"] if session_results else None
            state = self.traffic_simulator.generate_state(prev_performance)
            
            # Decide exploration vs exploitation strategy
            strategy, confidence = await learner.explore_vs_exploit(state)
            
            # Select action using online learner
            action = await learner.select_action(state, deterministic=(strategy == "exploit"))
            
            # Simulate campaign outcome
            outcome = self.traffic_simulator.simulate_campaign_outcome(action, state)
            
            # Track exploration vs exploitation
            if strategy == "explore":
                exploration_count += 1
            else:
                exploitation_count += 1
            
            # Create episode data
            episode_data = {
                "episode": episode,
                "timestamp": episode_start_time.isoformat(),
                "state": state,
                "action": action,
                "outcome": outcome,
                "strategy": strategy,
                "confidence": confidence,
                "reward": outcome["reward"],
                "roas": outcome["roas"],
                "safety_violation": outcome["safety_violation"],
                "emergency_mode": learner.emergency_mode,
                "arm_id": getattr(action, "metadata", {}).get("arm_id", "unknown")
            }
            
            # Record episode with learner
            learner.record_episode({
                "state": state,
                "action": action,
                "reward": outcome["reward"],
                "success": outcome["success"],
                "safety_violation": outcome["safety_violation"],
                "arm_id": episode_data["arm_id"],
                "next_state": state,  # Simplified for demo
                "done": False
            })
            
            session_results.append(episode_data)
            
            # Trigger online updates periodically
            if episode > 0 and episode % learner.config.online_update_frequency == 0:
                recent_experiences = []
                for recent_episode in session_results[-learner.config.online_update_frequency:]:
                    exp = {
                        "state": recent_episode["state"],
                        "action": recent_episode["action"],
                        "reward": recent_episode["reward"],
                        "next_state": recent_episode["state"],  # Simplified
                        "done": False,
                        "safety_violation": recent_episode["safety_violation"],
                        "arm_id": recent_episode["arm_id"]
                    }
                    recent_experiences.append(exp)
                
                update_result = await learner.online_update(recent_experiences)
                logger.info(f"Episode {episode}: Online update result: {update_result}")
            
            # Log progress
            if episode % 50 == 0 and episode > 0:
                recent_rewards = [r["reward"] for r in session_results[-50:]]
                recent_roas = [r["roas"] for r in session_results[-50:]]
                safety_violations = sum(1 for r in session_results[-50:] if r["safety_violation"])
                
                logger.info(
                    f"Episode {episode}: "
                    f"Avg Reward: {np.mean(recent_rewards):.3f}, "
                    f"Avg ROAS: {np.mean(recent_roas):.3f}, "
                    f"Safety Violations: {safety_violations}, "
                    f"Explore/Exploit: {exploration_count}/{exploitation_count}, "
                    f"Emergency Mode: {learner.emergency_mode}"
                )
        
        # Get final metrics
        final_metrics = learner.get_performance_metrics()
        logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
        
        await learner.shutdown()
        
        return session_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze online learning session results"""
        df = pd.DataFrame(results)
        
        analysis = {
            "total_episodes": len(results),
            "exploration_rate": df["strategy"].value_counts().get("explore", 0) / len(results),
            "average_reward": df["reward"].mean(),
            "reward_std": df["reward"].std(),
            "average_roas": df["roas"].mean(),
            "safety_violations": df["safety_violation"].sum(),
            "emergency_mode_episodes": df["emergency_mode"].sum(),
            "performance_trend": self._calculate_performance_trend(df),
            "arm_performance": self._analyze_arm_performance(df),
            "exploration_vs_exploitation": self._compare_exploration_exploitation(df)
        }
        
        return analysis
    
    def _calculate_performance_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance trend over time"""
        # Split data into first and second half
        mid_point = len(df) // 2
        first_half_reward = df.iloc[:mid_point]["reward"].mean()
        second_half_reward = df.iloc[mid_point:]["reward"].mean()
        
        improvement = (second_half_reward - first_half_reward) / abs(first_half_reward) * 100
        
        return {
            "first_half_avg": first_half_reward,
            "second_half_avg": second_half_reward,
            "improvement_percent": improvement
        }
    
    def _analyze_arm_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance by bandit arm"""
        arm_stats = {}
        
        for arm_id in df["arm_id"].unique():
            arm_data = df[df["arm_id"] == arm_id]
            if len(arm_data) > 0:
                arm_stats[arm_id] = {
                    "episodes": len(arm_data),
                    "avg_reward": arm_data["reward"].mean(),
                    "avg_roas": arm_data["roas"].mean(),
                    "success_rate": arm_data["outcome"].apply(lambda x: x["success"]).mean(),
                    "safety_violations": arm_data["safety_violation"].sum()
                }
        
        return arm_stats
    
    def _compare_exploration_exploitation(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compare exploration vs exploitation performance"""
        comparison = {}
        
        for strategy in ["explore", "exploit"]:
            strategy_data = df[df["strategy"] == strategy]
            if len(strategy_data) > 0:
                comparison[strategy] = {
                    "episodes": len(strategy_data),
                    "avg_reward": strategy_data["reward"].mean(),
                    "avg_roas": strategy_data["roas"].mean(),
                    "safety_violations": strategy_data["safety_violation"].sum()
                }
        
        return comparison
    
    def visualize_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create visualizations of online learning results"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Online Learning Performance Analysis", fontsize=16)
        
        # 1. Reward over time with moving average
        axes[0, 0].plot(df["reward"], alpha=0.3, label="Reward")
        axes[0, 0].plot(df["reward"].rolling(window=20).mean(), label="Moving Average (20)")
        axes[0, 0].axhline(y=analysis["average_reward"], color='r', linestyle='--', label="Overall Average")
        axes[0, 0].set_title("Reward Over Time")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        
        # 2. ROAS over time
        axes[0, 1].plot(df["roas"], alpha=0.4, color='green')
        axes[0, 1].plot(df["roas"].rolling(window=20).mean(), color='darkgreen', label="Moving Average")
        axes[0, 1].set_title("ROAS Over Time")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("ROAS")
        axes[0, 1].legend()
        
        # 3. Exploration vs Exploitation
        strategy_counts = df["strategy"].value_counts()
        axes[0, 2].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title("Exploration vs Exploitation")
        
        # 4. Reward distribution by strategy
        explore_rewards = df[df["strategy"] == "explore"]["reward"]
        exploit_rewards = df[df["strategy"] == "exploit"]["reward"]
        
        axes[1, 0].hist(explore_rewards, alpha=0.7, label="Exploration", bins=20)
        axes[1, 0].hist(exploit_rewards, alpha=0.7, label="Exploitation", bins=20)
        axes[1, 0].set_title("Reward Distribution by Strategy")
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        
        # 5. Arm performance comparison
        arm_performance = analysis["arm_performance"]
        if arm_performance:
            arms = list(arm_performance.keys())
            rewards = [arm_performance[arm]["avg_reward"] for arm in arms]
            
            axes[1, 1].bar(arms, rewards)
            axes[1, 1].set_title("Average Reward by Bandit Arm")
            axes[1, 1].set_xlabel("Arm")
            axes[1, 1].set_ylabel("Average Reward")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Safety violations over time
        safety_violations = df["safety_violation"].astype(int)
        cumulative_violations = safety_violations.cumsum()
        
        axes[1, 2].plot(cumulative_violations)
        axes[1, 2].set_title("Cumulative Safety Violations")
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("Cumulative Violations")
        
        plt.tight_layout()
        return fig
    
    async def run_comparative_analysis(self):
        """Run comparative analysis between online learning and baseline"""
        logger.info("Running comparative analysis: Online Learning vs Baseline")
        
        # Run online learning session
        online_results = await self.run_online_learning_session(num_episodes=300)
        online_analysis = self.analyze_results(online_results)
        
        # Simulate baseline performance (no learning)
        baseline_results = []
        agent, _ = await self.setup_agent_and_learner()
        
        for episode in range(300):
            state = self.traffic_simulator.generate_state()
            action = await agent.select_action(state, deterministic=True)  # No exploration
            outcome = self.traffic_simulator.simulate_campaign_outcome(action, state)
            
            baseline_results.append({
                "episode": episode,
                "reward": outcome["reward"],
                "roas": outcome["roas"],
                "safety_violation": outcome["safety_violation"]
            })
        
        baseline_analysis = self.analyze_results(baseline_results)
        
        # Compare results
        comparison = {
            "online_learning": {
                "avg_reward": online_analysis["average_reward"],
                "avg_roas": online_analysis["average_roas"],
                "safety_violations": online_analysis["safety_violations"],
                "improvement_percent": online_analysis["performance_trend"]["improvement_percent"]
            },
            "baseline": {
                "avg_reward": baseline_analysis["average_reward"],
                "avg_roas": baseline_analysis["average_roas"],
                "safety_violations": baseline_analysis["safety_violations"],
                "improvement_percent": baseline_analysis["performance_trend"]["improvement_percent"]
            }
        }
        
        # Calculate relative improvement
        reward_improvement = ((comparison["online_learning"]["avg_reward"] - 
                              comparison["baseline"]["avg_reward"]) / 
                             abs(comparison["baseline"]["avg_reward"]) * 100)
        
        roas_improvement = ((comparison["online_learning"]["avg_roas"] - 
                           comparison["baseline"]["avg_roas"]) / 
                          abs(comparison["baseline"]["avg_roas"]) * 100)
        
        logger.info(f"Online Learning vs Baseline Results:")
        logger.info(f"Reward Improvement: {reward_improvement:.2f}%")
        logger.info(f"ROAS Improvement: {roas_improvement:.2f}%")
        logger.info(f"Online Safety Violations: {comparison['online_learning']['safety_violations']}")
        logger.info(f"Baseline Safety Violations: {comparison['baseline']['safety_violations']}")
        
        return comparison


async def main():
    """Main demo function"""
    print("=" * 60)
    print("GAELP Online Learning Demonstration")
    print("=" * 60)
    
    demo = OnlineLearningDemo()
    
    try:
        print("\n1. Running basic online learning session...")
        results = await demo.run_online_learning_session(num_episodes=200)
        
        print("\n2. Analyzing results...")
        analysis = demo.analyze_results(results)
        
        print("\nOnline Learning Results Summary:")
        print(f"Total Episodes: {analysis['total_episodes']}")
        print(f"Exploration Rate: {analysis['exploration_rate']:.2%}")
        print(f"Average Reward: {analysis['average_reward']:.3f} ± {analysis['reward_std']:.3f}")
        print(f"Average ROAS: {analysis['average_roas']:.3f}")
        print(f"Safety Violations: {analysis['safety_violations']}")
        print(f"Performance Improvement: {analysis['performance_trend']['improvement_percent']:.2f}%")
        
        print("\nBandit Arm Performance:")
        for arm_id, stats in analysis['arm_performance'].items():
            print(f"  {arm_id}: {stats['avg_reward']:.3f} reward, {stats['episodes']} episodes")
        
        print("\n3. Creating visualizations...")
        fig = demo.visualize_results(results, analysis)
        plt.savefig("online_learning_results.png", dpi=300, bbox_inches='tight')
        print("Visualizations saved to 'online_learning_results.png'")
        
        print("\n4. Running comparative analysis...")
        comparison = await demo.run_comparative_analysis()
        
        online_reward = comparison["online_learning"]["avg_reward"]
        baseline_reward = comparison["baseline"]["avg_reward"]
        improvement = (online_reward - baseline_reward) / abs(baseline_reward) * 100
        
        print(f"\nFinal Comparison:")
        print(f"Online Learning Average Reward: {online_reward:.3f}")
        print(f"Baseline Average Reward: {baseline_reward:.3f}")
        print(f"Relative Improvement: {improvement:.2f}%")
        
        if improvement > 5:
            print("✅ Online learning shows significant improvement!")
        elif improvement > 0:
            print("✅ Online learning shows modest improvement.")
        else:
            print("⚠️ Online learning performance needs tuning.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")


if __name__ == "__main__":
    # Ensure matplotlib works in headless environments
    import matplotlib
    matplotlib.use('Agg')
    
    # Run the demo
    asyncio.run(main())