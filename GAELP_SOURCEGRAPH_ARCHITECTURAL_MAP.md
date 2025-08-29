=== GAELP CODEBASE ARCHITECTURAL MAP ===
Generated: Wed Aug 27 04:41:00 UTC 2025

## 1. CORE RL COMPONENTS
✱ 30+ results for "repo:github.com/harikravich/AELP class.*Agent|class.*Environment|class.*PPO" in 78ms
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/run_full_demo.py)
github.com/harikravich/AELP › run_full_demo.py (1 matches)
--------------------------------------------------------------------------------
      20 |  class MockAgent:
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/demo_monte_carlo.py)
github.com/harikravich/AELP › demo_monte_carlo.py (1 matches)
--------------------------------------------------------------------------------
      34 |  class DemoAgent:
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/competitor_agents.py)
github.com/harikravich/AELP › competitor_agents.py (9 matches)
--------------------------------------------------------------------------------
     103 |  class BaseCompetitorAgent(ABC):
     104 |      """Base class for all competitor agents"""
  ------------------------------------------------------------------------------
      40 |  class AgentType(Enum):
  ------------------------------------------------------------------------------

## 2. INTEGRATIONS
✱ 30+ results for "repo:github.com/harikravich/AELP import.*RecSim|import.*AuctionGym|import.*stable_baselines" in 120ms
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/gaelp_master_integration_simple.py)
github.com/harikravich/AELP › gaelp_master_integration_simple.py (2 matches)
--------------------------------------------------------------------------------
     104 |  recsim_imports = safe_import('recsim_auction_bridge', 
  ------------------------------------------------------------------------------
     277 |          recsim_class = recsim_imports.get('RecSimAuctionBridge')
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/debug_auction.py)
github.com/harikravich/AELP › debug_auction.py (1 matches)
--------------------------------------------------------------------------------
      19 |      from auction_gym_integration import AuctionGymWrapper
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/COMPETITOR_AGENTS_SUMMARY.md)

## 3. TRAINING METHODS
✱ 30+ results for "repo:github.com/harikravich/AELP def.*train|def.*learn|def.*update" in 111ms
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/train_aura_agent.py)
github.com/harikravich/AELP › train_aura_agent.py (3 matches)
--------------------------------------------------------------------------------
     221 |      async def train(self, num_episodes: int = 100):
  ------------------------------------------------------------------------------
     158 |      async def train_episode(self, episode_num: int) -> Dict[str, Any]:
  ------------------------------------------------------------------------------
     264 |      def analyze_learning(self):
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/attribution_models.py)
github.com/harikravich/AELP › attribution_models.py (3 matches)
--------------------------------------------------------------------------------
     206 |      def train(self, journeys: List[Journey], validation_split: float = 0.2):
  ------------------------------------------------------------------------------
     434 |      def train_data_driven_model(self, journeys: List[Journey]):
  ------------------------------------------------------------------------------
     246 |      def _prepare_training_data(self, journeys: List[Journey]) -> Tuple[np.ndarray, np.ndarray]:
--------------------------------------------------------------------------------

## 4. DATA PIPELINES
✱ 30+ results for "repo:github.com/harikravich/AELP class.*Database|class.*Journey|def.*get_data" in 151ms
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/attribution_models.py)
github.com/harikravich/AELP › attribution_models.py (1 matches)
--------------------------------------------------------------------------------
      45 |  class Journey:
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/conversion_lag_model.py)
github.com/harikravich/AELP › conversion_lag_model.py (1 matches)
--------------------------------------------------------------------------------
      36 |  class ConversionJourney:
--------------------------------------------------------------------------------
(https://gaelp.sourcegraph.app/github.com/harikravich/AELP/-/blob/user_journey_database.py)
github.com/harikravich/AELP › user_journey_database.py (4 matches)
--------------------------------------------------------------------------------

## 5. MAIN ENTRY POINTS
