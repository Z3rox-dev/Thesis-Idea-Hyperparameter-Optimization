# Elastic Space Physics Module - Analysis Summary

## Overview

This module implements physics-based adaptive sampling using a "stiffness field"
that learns from the optimization landscape. It was designed as a "free" alternative
to geometry probing that doesn't require dedicated evaluations.

## Key Results

### Standalone Performance
- On Ellipsoid function: **+10.8% improvement** vs uniform random search (50% win rate)
- Successfully learns anisotropic structure (correlation ~0.98 with true gradient weights)
- All 7 diagnostic tests pass

### Integration with ALBA
- **Does NOT help**: -15% to -41% on various functions
- Root cause: ALBA already has sophisticated local search that doesn't benefit from additional modulation

## Why It Works Standalone but Not with ALBA

1. **Exploration vs Local Search Mismatch**
   - Stiffness learned during exploration (sparse, global) ≠ stiffness needed for local search (dense, local)
   - Near optimum, all gradients → 0, so global gradient info is irrelevant

2. **Double Modulation**
   - ALBA's local search already uses shrinking std over progress
   - Adding elastic modulation on top creates interference

3. **Phase Disconnect**
   - Elastic space learns from ALL points
   - But only the local search phase could use it
   - Exploration points "pollute" the learned stiffness

## When to Use Elastic Space

✅ **Standalone local search** (not with ALBA)
✅ **Simple optimization loops** (random search + local refinement)
✅ **Budget >= 200** (needs data to learn)
✅ **Anisotropic functions** (Ellipsoid, ill-conditioned quadratics)

❌ **With ALBA** (conflicting mechanisms)
❌ **Mixed categorical/continuous** (only learns continuous)
❌ **Low budget** (insufficient data)

## Recommendations for ALBA Integration

If we want physics-based improvements in ALBA, better approaches would be:

1. **Cube Diffusion** (`cube_diffusion.py`)
   - Diffuse quality info between adjacent cubes
   - Uses ALBA's native structure
   - Guides exploration without affecting local search

2. **Geodesics** (not implemented)
   - During local search, follow gradient field
   - Would require online gradient estimation
   - More complex but theoretically sound

3. **Wave Probing** (not implemented)
   - Sinusoidal perturbations to detect resonance
   - Could identify sensitive dimensions cheaply
   - Needs careful integration with ALBA phases

## Files Created

- `elastic_space.py` - Main module
- `cube_diffusion.py` - Alternative for ALBA integration
- `diagnostics/test_elastic_space.py` - Standalone tests (ALL PASS)
- `diagnostics/test_alba_elastic_space.py` - ALBA integration tests
- `diagnostics/elastic_space_diagnostic.py` - Deep analysis

## Conclusion

The Elastic Space module is mathematically sound and works well as a standalone
sampler, but **should NOT be enabled in ALBA** (`elastic_space=False`).

The module remains in the codebase as a potential tool for:
- Standalone optimization tasks
- Future research on physics-based HPO
- Educational purposes (demonstrates stiffness/diffusion concepts)
