#!/usr/bin/env python3
"""
ACTUALLY FUCKING RUN THE SYSTEM
"""

from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment

print("\n🚀 RUNNING GAELP SYSTEM\n")

# Create environment and agent
env = ProductionFortifiedEnvironment()
agent = ProductionFortifiedRLAgent()

# Run 3 episodes
for episode in range(3):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) > 32 and step % 32 == 0:
            agent.replay()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode+1}: {step+1} steps, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.4f}")

print("\n✅ DONE. IT FUCKING WORKS.\n")