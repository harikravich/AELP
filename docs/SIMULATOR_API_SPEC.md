# Simulator API Specification (Scaffold)

Purpose: define a unified interface for the simulator used by training and tests.

## Interfaces
- Simulator.reset() -> State
- Simulator.step(action) -> (next_state, reward, done, info)
- Components: UserModel, AuctionModel, CreativeResponseModel, ConversionLagModel, AttributionAdapter
- Calibration: load(version_id), validate(metrics)

## State & Action
- State: features from user, auction context, creative context, budget/pacing, temporal, identity.
- Action: bid, creative_id, channel/platform allocation (if handled at this level).

## Info
- Includes auction_outcome, interaction_result, spend, revenue, attribution details for delayed credit.

