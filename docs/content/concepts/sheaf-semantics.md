# Sheaf Semantics

## Definition

**Sheaf semantics** interprets the resolution pipeline through the lens of
sheaf cohomology. The constraint topology — where open sets correspond to
compatible subsets of constraints — carries a natural
{@class https://uor.foundation/cohomology/Sheaf} of resolution data. This
viewpoint unifies local constraint satisfaction with global resolution.

## Local vs Global Consistency

The distinction between local and global consistency is captured by the
sheaf structure:

- **Stalks**: A {@class https://uor.foundation/cohomology/Stalk} at a single
  constraint holds the local resolution data — the fibers pinned by that
  constraint alone. Local consistency means each stalk is individually
  satisfiable.
- **Global sections**: A {@class https://uor.foundation/cohomology/Section}
  over the entire constraint space represents a globally consistent
  resolution — an assignment that simultaneously satisfies all constraints.

Local consistency does not imply global consistency. The gap between the two
is precisely what cohomology measures.

## Gluing Obstructions

When local sections over overlapping open sets cannot be assembled into a
global section, a {@class https://uor.foundation/cohomology/GluingObstruction}
arises. These obstructions live in the first cohomology group H^1:

- **H^0** (global sections): captures the space of fully resolved states.
  A nonzero H^0 means at least one global resolution exists.
- **H^1** (gluing obstructions): classifies the ways local resolutions fail
  to glue. A nontrivial H^1 signals that the constraint set has intrinsic
  conflicts visible only at the global level.

## Connection to the Resolution Pipeline

The sheaf-cohomological perspective connects to the resolution pipeline
through stages psi_5 and psi_6 of the structural reasoning pipeline:

1. **psi_5** dualizes the chain complex into a cochain complex, lifting
   boundary data to coboundary data.
2. **psi_6** computes cohomology from the cochain complex, producing
   obstruction classes.

The iterative resolution loop (from `resolver/`) can then be understood as
an attempt to kill cohomology classes: each refinement step reduces H^1
until all obstructions vanish and a global section (complete resolution)
exists.

## Practical Interpretation

| Cohomology group | Resolution meaning |
|------------------|--------------------|
| H^0 = 0 | No global resolution exists |
| H^0 nontrivial | At least one global resolution exists |
| H^1 = 0 | Local solutions always glue to global solutions |
| H^1 nontrivial | Gluing obstructions present; iterative refinement needed |
