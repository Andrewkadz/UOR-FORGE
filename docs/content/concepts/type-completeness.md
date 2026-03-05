# Type Completeness

## Definition

**Type completeness** is the formal property of a
{@class https://uor.foundation/type/ConstrainedType} that guarantees resolution
always terminates in O(1) time. A type is complete when its constraint nerve
satisfies the completeness criterion IT\_7d: the Euler characteristic of the
constraint nerve equals the quantum level _n_ and all Betti numbers β\_k are
zero.

Completeness is certified by the kernel via the
{@class https://uor.foundation/cert/CompletenessCertificate} pathway. The
certified entity is a {@class https://uor.foundation/type/CompleteType} —
a subclass of `ConstrainedType` that has passed the full ψ pipeline.

## Certification Pipeline

The completeness certification lifecycle proceeds in four stages:

1. **Candidate registration** — A `ConstrainedType` is promoted to a
   {@class https://uor.foundation/type/CompletenessCandidate} by associating
   it with a resolver:ResolutionState and an
   observable:ConstraintNerve via the
   {@prop https://uor.foundation/type/candidateNerve} property.

2. **Witness accumulation** — Each constraint application that closes at
   least one fiber produces a
   {@class https://uor.foundation/type/CompletenessWitness}. The witness
   records the applied constraint via
   {@prop https://uor.foundation/type/witnessConstraint} and the
   {@prop https://uor.foundation/type/fibersClosed} count.

3. **Resolver evaluation** — A
   {@class https://uor.foundation/resolver/CompletenessResolver} reads the
   cached {@prop https://uor.foundation/resolver/nerveEulerCharacteristic}
   from the ResolutionState. If χ(N(C)) = n and all β\_k = 0, the kernel
   issues a CompletenessCertificate; otherwise it emits a
   resolver:RefinementSuggestion.

4. **Certificate issuance** — The
   {@class https://uor.foundation/cert/CompletenessCertificate} links to the
   certified {@class https://uor.foundation/type/CompleteType} via
   {@prop https://uor.foundation/cert/certifiedType} and records provenance
   via an {@class https://uor.foundation/cert/CompletenessAuditTrail}.

## Termination Criterion IT\_7d

IT\_7d requires:

- χ(N(C)) = n (Euler characteristic of the constraint nerve equals the
  quantum level)
- β\_k = 0 for all k ≥ 0 (no topological obstructions)

When IT\_7d holds, the residual entropy S = freeCount × ln 2 drops to zero,
meaning no unconstrained fibers remain.

## Related

- {@class https://uor.foundation/resolver/CompletenessResolver}
- {@class https://uor.foundation/cert/CompletenessCertificate}
- {@class https://uor.foundation/cert/CompletenessAuditTrail}
- [Session Resolution](session-resolution.html)
