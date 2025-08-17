# Reviewer Notes (Confidential)

## 1. Structure (Mathematics Paper Standards)
- Current organization mixes syntax, semantics, and operational discussion; recommend a clean partition:
	1. Preliminaries (basic probability, BN factorization, junction tree notions).
	2. Syntax (grammar, well-formedness, construction of formula DAG).
	3. Denotational Semantics (interpretation function, normalization operator, lifting operator) with precise definitions.
	4. Operational Semantics (evaluation rules, cache transition system) given as formal inference rules.
	5. Main Results (Soundness, Completeness, Locality, Normalization Invariance, Complexity Bounds) with proofs or clearly labeled proof sketches.
	6. Extensions / Further Directions.
- Contributions list exists but does not distinguish clearly between known concepts and purportedly novel aspects (runtime symbolic DAG, fine-grained cache invalidation). Add an explicit “Main Contributions” paragraph closing the Introduction.
- Appendices should contain full proofs; presently only informal sketches appear inline.
- Notation summary is helpful but inconsistent (uses \Ref although macro replaced by \fref) and should be moved just after Preliminaries or before Appendices.

## 2. Technical Correctness / Rigor
- Denotational semantics: Every product/marginal is normalized; must supply an invariant relating these normalized intermediates to classical unnormalized potentials. Provide a formal scaling lemma with induction (base cases: unit, node potential, evidence; inductive steps: product, marginalization).
- Operational semantics: Should present formal evaluation judgments with rules. Currently “Cached”, “Product”, etc. are prose only; missing explicit premises and side-conditions (e.g., dependency graph acyclicity).
- Dependency graph: Not formally defined; need construction function mapping formulas to edges; prove acyclicity (smart-constructor invariants) and finiteness.
- Joint query procedure: Ratio P(A∪E)/P(E) requires explicit assumption P(E)>0. State behavior for contradictory evidence (define result or raise error). Define temporary join construction within the syntax (new operator or derived rule) or show reduction to existing rules.
- Reference Elision lemma and notation summary mismatch macros; fix for internal coherence.
- Marginal–Product commutation lemma: conditions and proof vague (“up to normalization”). Provide precise statement using proportionality \propto or explicit normalization function.
- Complexity claims lack bounds on number of formulas, maximum cache size, and depth of dependency chains. Need lemmas bounding these by structural parameters (number of cliques, separators, treewidth).
- Soft evidence only mentioned; semantics not formalized (e.g., replacing indicator with likelihood—adjust normalization points).
- Grammar: Nonterminals CPT, PotentialSpec are placeholders with no formal production rules; either define them or remove from grammar to avoid ambiguity.

## 3. Novelty / Positioning (Theoretical Context)
- Must clearly separate what is original: (i) a general algebraic formulation of lazy propagation with persistent symbolic DAG built once per structure; (ii) a recursive cache invalidation strategy proven minimal; (iii) a join query decomposition using symbolic retraction logic. These need formal statements.
- Overlap with existing lazy propagation and arithmetic circuit frameworks is only cursorily discussed; add a comparative subsection specifying which algebraic identities or operational guarantees are not covered in earlier work.
- Efficiency claims should be reframed as complexity theorems (worst-case, amortized) rather than empirical promises (experiments are optional in a math paper but complexity rigor is required).

## 4. Clarity & Exposition
- Inconsistent symbol choices (PI/SIGMA/PHI/EPS ASCII stand-ins vs Unicode in math). Adopt a single canonical form; if ASCII is for grammar meta-language, state this explicitly.
- Undefined operators: “Lift” and “Norm” invoked without formal signatures. Define: Lift_{D→U}: Pot(D)→Pot(U) plus normalization policy; Norm: nonnegative function → probability distribution or zero-preserving map.
- Normalization with zero total mass: define explicitly (e.g., leave vector zero and treat later normalization invariance carefully) or prohibit via well-formedness (no all-zero potentials).
- Provide a table summarizing constructors, domains, and simplification rules (smart constructor rewrites) with proof of termination and confluence (or justification why not needed).
- Ensure macro consistency: notation summary must use \fref not \Ref.

## 5. Literature Coverage (Mathematical / Theoretical)
- Expand citations: arithmetic circuits (later developments beyond 2003), AND/OR search and caching (Dechter, Mateescu), probabilistic circuits and knowledge compilation (e.g., Kisa, Chavira), incremental inference frameworks, lazy propagation variants, factor graph normalization methods.
- Cite canonical sources for normalization invariance, factor algebra identities, and any specialized summation techniques (if Kahan or stable summation is part of the theoretical guarantee, cite an authoritative source or remove from theory if purely implementation detail).

## 6. Red Flags (Blocking Issues Before Acceptance)
- Core theorems lack formal statements and proofs (soundness, completeness, locality, normalization invariance).
- Undefined grammar components (CPT, PotentialSpec) break formal self-containment.
- Ambiguous handling of impossible evidence (P(E)=0) and soft evidence; must resolve.
- Normalization strategy’s effect on algebraic laws (associativity, distributivity) is not analyzed; need lemmas showing normalized product still composes correctly (or constrained conditions).
- Notation inconsistencies risk misinterpretation (Ref vs fref, ASCII vs Unicode operators).
- No formal complexity theorems bounding space/time in terms of treewidth and formula count.

## 7. Required Remediations (Mathematics-Focused)
- Supply formal inference system for evaluation; include derivation height bounds.
- Add complete inductive proofs (appendix) for: normalization invariance, locality of invalidation, completeness (every posterior marginal produced), and soundness (denotation matches BN semantics).
- Define all constructors and rewrite (smart) rules; prove they preserve semantics and terminate.
- Provide explicit complexity theorem(s) with clearly stated parameters (n variables, treewidth w, max domain size K, number of cliques c) and proof sketches.
- Formalize soft evidence extension or explicitly declare it out of scope.
- Clarify behavior for zero-probability evidence; add theorem stating conditions ensuring P(E)>0 under which results hold, or supply fallback semantics.
- Complete grammar: define CPT as a function schema (perhaps as a labeled multidimensional array) or remove from presented grammar and treat as external data.
- Unify notation: decide on either Unicode mathematical symbols in prose and abstract ASCII in grammar (with mapping table) or use consistent Unicode throughout.

## 8. Overall Recommendation
Conceptually promising but **Major Revision Required** for mathematical publication: insufficient formal rigor, incomplete grammar, missing proofs, and positioning gaps.

---
End of reviewer notes.
