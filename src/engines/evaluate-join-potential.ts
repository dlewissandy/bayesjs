import { FastNode } from './FastNode'
import { propagateJoinMessages } from './join-propagation'
import { evaluate, evaluateMarginalPure } from './evaluation'
import { FastClique } from './FastClique'
import { Formula } from './Formula'
import { FastPotential } from './FastPotential'
import { sum } from 'ramda'

/** Construct the join of an arbitrary collection of variables in a Bayesian Network,
 * conditioned on an optional set of parent variables.  Returns the potential function
 * of the joint distribution over the head and parent variables subject to any evidence.
 * @param nodes: The collection of nodes in the bayesian network
 * @param cliques: The collection of cliques in the junction tree for the bayesian
 *   network.
 * @param formulas: The collection of formulas for computing the posterior
 *   distributions for the cliques, nodes and separators  of the Bayes Network, and
 *   all the intermediate potentials.
 * @param potentialFunction: The collection of potential functions for the cliques
 *   nodes and separators of the Bayes Network and the intermediate potentials.
 * @param headVariables: The indices of the variables that occur as head variables of
 *   the joint distribution being constructed.  These indices must be valid references
 *   to variables in the network.
 * @param parentVariables: The idices of the variables that occur as parent variables
 *   of the joint distribution being constructed.   These indices must be valid
 *   references for the variables in the network, and must be distinct from the
 *   head variables of the joint being constructed.
 * @returns: The potential function for the join which satisfies the conditions
 *   that for every combination of parents, the potentials sum to unity.  The
 *   case when there are no parents (unconditioned joint distribution), corresponds
 *   to the trivial case where there is only a single (empty) combination of
 *   parent values.
 */
export function evaluateJoinPotentials (nodes: FastNode[], cliques: FastClique[], connectedComponents: number[][], separators: number[][], formulas: Formula[], potentials: (FastPotential | null)[], headVariables: number[], parentVariables: number[]): { potentials: FastPotential; supplementalFormulas: Formula[]; supplementalPotentials: (FastPotential | null)[]} {
  // We start by propagating the join messages to construct a formula for the new
  // join distribution.  This may create supplementary formulas that were not
  // encountered in the original message passing to make the cliques consistent.
  const { joinFormulaId, supplementalFormulas } = propagateJoinMessages(nodes, cliques, connectedComponents, formulas, separators, headVariables, parentVariables)
  // We initialize an array of potential to contain any new computations and seed it
  // with the previous the previously computed potentials, and enough new elements to cover
  // the formulas added during message passing.   Then we evaluate the joint formula using
  // that context.
  const amendedPotentials: (FastPotential | null)[] = [...potentials, ...Array(supplementalFormulas.length).fill(null)]
  const amendedFormulas = [...formulas, ...supplementalFormulas]

  // Now we can evaluate the formula for the joint distribution.  This distribution
  // will be equivalent to the requested join up to a permutation of the variables
  // in the domain.
  const joinPotentials = evaluate(joinFormulaId, nodes, amendedFormulas, amendedPotentials)
  const joinFormula = amendedFormulas[joinFormulaId]
  const joinDomain = [...headVariables, ...parentVariables]

  // if any new potentials were computed, then update the cache.
  amendedPotentials.slice(0, potentials.length).forEach((ps, i) => {
    if (potentials[i] == null) potentials[i] = ps
  })
  // If required, we permute the elements in the potential function to agree with the
  // requested order of variables in the domain and normalize the potentials to ensure
  // that they sum to unity.
  let result: FastPotential = []
  if (joinFormula.domain.every((n, i) => n === joinDomain[i])) {
    const total = sum(joinPotentials)
    result = joinPotentials.map(p => total !== 0 ? p / total : p)
  } else {
    result = evaluateMarginalPure(joinPotentials, joinFormula.domain, joinFormula.domain.map(i => nodes[i].levels.length), joinDomain, joinDomain.map(i => nodes[i].levels.length), joinPotentials.length, true)
  }
  return {
    potentials: result,
    supplementalFormulas: amendedFormulas.slice(formulas.length),
    supplementalPotentials: amendedPotentials.slice(formulas.length),
  }
}
