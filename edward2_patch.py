#!/usr/bin/env python3
"""
Patch for edward2/scipy compatibility issue
Must be imported BEFORE edward2
"""

import scipy.stats
import numpy as np

# Patch the missing rvs method in dirichlet_multinomial
if not hasattr(scipy.stats.dirichlet_multinomial, 'rvs'):
    def rvs_patch(alpha, n, size=1, random_state=None):
        """Patch for missing rvs method in dirichlet_multinomial"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Handle different input formats
        alpha = np.asarray(alpha)
        if alpha.ndim == 1:
            # Single alpha vector
            results = []
            for _ in range(size if isinstance(size, int) else np.prod(size)):
                # Sample from Dirichlet to get probabilities
                probs = np.random.dirichlet(alpha)
                # Sample from multinomial with those probabilities
                sample = np.random.multinomial(n, probs)
                results.append(sample)
            
            if size == 1:
                return results[0]
            else:
                return np.array(results).reshape(size + alpha.shape)
        else:
            # Multiple alpha vectors
            results = []
            for a in alpha:
                probs = np.random.dirichlet(a)
                sample = np.random.multinomial(n, probs)
                results.append(sample)
            return np.array(results)
    
    # Add the method to the dirichlet_multinomial object
    scipy.stats.dirichlet_multinomial.rvs = rvs_patch
    print("✅ Patched scipy.stats.dirichlet_multinomial.rvs for edward2 compatibility")

# Also patch edward2.traceable if it doesn't exist
try:
    import edward2
    if not hasattr(edward2, 'traceable'):
        # Add a dummy traceable function for compatibility
        def traceable(func):
            """Compatibility wrapper for missing edward2.traceable"""
            return func
        edward2.traceable = traceable
        print("✅ Patched edward2.traceable for compatibility")
except ImportError:
    pass