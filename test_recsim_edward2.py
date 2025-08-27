#!/usr/bin/env python3
"""
Test RecSim/edward2 issue
"""

print("Testing RecSim/edward2 compatibility...")

try:
    print("1. Importing edward2...")
    import edward2 as ed
    print("   ✅ edward2 imported")
    
    print("2. Checking dirichlet_multinomial...")
    # The issue is that scipy.stats.dirichlet_multinomial doesn't have rvs method
    from scipy.stats import dirichlet_multinomial_gen
    print(f"   dirichlet_multinomial methods: {dir(dirichlet_multinomial_gen)[:5]}")
    
    print("3. Importing recsim_ng.core.value...")
    import recsim_ng.core.value as value
    print("   ✅ recsim_ng.core.value imported")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to patch the issue...")

# The issue is that edward2 expects scipy.stats.dirichlet_multinomial to have rvs method
# but it doesn't. This is a compatibility issue between versions.

# We can work around this by patching before import
import scipy.stats
if not hasattr(scipy.stats.dirichlet_multinomial, 'rvs'):
    print("Patching dirichlet_multinomial.rvs...")
    
    def rvs_patch(self, alpha, n, size=1, random_state=None):
        """Patch for missing rvs method in dirichlet_multinomial"""
        import numpy as np
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample from Dirichlet to get probabilities
        probs = np.random.dirichlet(alpha)
        # Sample from multinomial with those probabilities
        return np.random.multinomial(n, probs, size=size)
    
    scipy.stats.dirichlet_multinomial.rvs = rvs_patch
    print("✅ Patched dirichlet_multinomial.rvs")

# Now try importing again
try:
    import recsim_ng.core.value as value
    print("✅ RecSim imports work after patch!")
except Exception as e:
    print(f"❌ Still failing: {e}")