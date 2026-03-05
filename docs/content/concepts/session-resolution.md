# Session Resolution

## Definition

**Session resolution** is the multi-turn inference protocol in which a sequence
of {@class https://uor.foundation/query/RelationQuery} evaluations shares a
common {@class https://uor.foundation/state/Context}. Each resolved query
appends a {@class https://uor.foundation/state/Binding} to a
{@class https://uor.foundation/state/BindingAccumulator}, monotonically
reducing the aggregate free fiber space for subsequent queries.

A {@class https://uor.foundation/state/Session} is the bounded collection of
query/response pairs that share this accumulator. Sessions are the unit of
coherent multi-turn reasoning in Prism deployments.

## Monotonicity Invariant

The binding monotonicity invariant (SR\_1) guarantees:

> freeCount(B\_{i+1}) ≤ freeCount(B\_i) for all i in a Session S.

The {@prop https://uor.foundation/state/aggregateFiberDeficit} on the
BindingAccumulator tracks the total remaining free fibers across all accumulated
bindings. This value can only decrease as the session progresses — accumulated
knowledge cannot increase uncertainty.

## SessionResolver

A {@class https://uor.foundation/resolver/SessionResolver} is the top-level
resolver for multi-turn Prism deployments. It maintains the BindingAccumulator
across evaluations via the
{@prop https://uor.foundation/resolver/sessionAccumulator} property.

## SessionQuery

A {@class https://uor.foundation/query/SessionQuery} extends RelationQuery
by explicitly declaring session membership via the
{@prop https://uor.foundation/query/sessionMembership} property. This
allows the conformance suite to validate session-scoped fiber reduction.

## Session Boundaries

A {@class https://uor.foundation/state/SessionBoundary} marks a context-reset
event within a session stream. The {@class https://uor.foundation/state/SessionBoundaryType}
vocabulary enum classifies the reason:

| Individual | Meaning |
|------------|---------|
| `state:ExplicitReset` | Caller requested context reset |
| `state:ConvergenceBoundary` | No further queries can reduce the aggregate fiber deficit |
| `state:ContradictionBoundary` | New query produced a type contradiction with an accumulated binding |

Each boundary records the {@prop https://uor.foundation/state/priorContext}
(the state before reset) and the {@prop https://uor.foundation/state/freshContext}
(the clean state for subsequent queries).

## Session Identity Algebra (SR_ series)

Amendment 27 adds five SR\_ identity individuals formalizing the session
resolution algebra:

| Identity | Statement |
|----------|-----------|
| SR\_1 | Binding monotonicity: freeCount(B\_{i+1}) ≤ freeCount(B\_i) |
| SR\_2 | Empty session is identity: freeCount(B\_0) = total fiber space |
| SR\_3 | Convergence: session terminates iff freeCount reaches minimum |
| SR\_4 | Disjoint bindings compose without fiber conflict |
| SR\_5 | ContradictionBoundary fires iff ∃ conflicting bindings at same address |

## Related

- {@class https://uor.foundation/state/Session}
- {@class https://uor.foundation/state/BindingAccumulator}
- {@class https://uor.foundation/state/SessionBoundary}
- {@class https://uor.foundation/resolver/SessionResolver}
- [Type Completeness](type-completeness.html)
