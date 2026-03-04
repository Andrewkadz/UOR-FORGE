/// SHACL test 23: Identity grounding spot-check — verificationStatus and verificationPath.
pub const TEST23_IDENTITY_GROUNDING: &str = r#"
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix op:   <https://uor.foundation/op/> .

op:R_A1 a op:Identity ;
    op:lhs "add(x, add(y, z))" ;
    op:rhs "add(add(x, y), z)" ;
    op:forAll "x, y, z ∈ R_n" ;
    op:verificationStatus "verifiable" ;
    op:verificationPath "Ring associativity — direct computation" .

op:C_1 a op:Identity ;
    op:lhs "pins(compose(A, B))" ;
    op:rhs "pins(A) ∪ pins(B)" ;
    op:forAll "constraints A, B" ;
    op:verificationStatus "derivable" ;
    op:verificationPath "Constraint composition — set union lemma" .

op:F_1 a op:Identity ;
    op:lhs "pinned fiber" ;
    op:rhs "cannot be unpinned" ;
    op:forAll "FiberCoordinate" ;
    op:verificationStatus "derivable" ;
    op:verificationPath "Fiber monotonicity — lattice argument" .

op:DC_1 a op:Identity ;
    op:lhs "∂_R f(x)" ;
    op:rhs "f(succ(x)) - f(x)" ;
    op:forAll "f : R_n → R_n, x ∈ R_n" ;
    op:verificationStatus "derivable" ;
    op:verificationPath "Discrete derivative — finite difference definition" .

op:psi_1 a op:Identity ;
    op:lhs "ψ₁(κ_k, constraint_k)" ;
    op:rhs "fiber pinning state" ;
    op:forAll "curvature κ_k, constraint_k" ;
    op:verificationStatus "derivable" ;
    op:verificationPath "Curvature → Fiber — index bridge" .
"#;
