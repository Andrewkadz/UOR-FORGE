/// SHACL test 36: Q1Ring and QuantumLevel chain — Amendment 26.
pub const TEST36_Q1_RING: &str = r#"
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix owl:    <http://www.w3.org/2002/07/owl#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix schema: <https://uor.foundation/schema/> .

schema:ex_q1ring_36 a owl:NamedIndividual, schema:Q1Ring ;
    schema:Q1bitWidth  "16"^^xsd:positiveInteger ;
    schema:Q1capacity  "65536"^^xsd:positiveInteger .

schema:Q0 schema:nextLevel schema:Q1 .
"#;
