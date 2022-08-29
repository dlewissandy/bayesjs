import {
  FastPotential,
  combinationToIndex,
  indexToCombination,
} from './FastPotential'
import { InferenceEngine } from '..'
import { product, partition } from 'ramda'
import { evaluateMarginalPure } from './evaluation'
import { normalize } from './util'
import { FastClique } from './FastClique'
import { FastNode } from './FastNode'
import { Formula } from './Formula'
import { evaluateFormula } from './evaluate-join-probability'

/** A CliqueInfo object is collection of information about a
 * clique that facilitates efficient random sampling of that
 * clique's variables.
 */
type CliqueInfo = {
  /** The ordinal position of the clique in the Bayes Network's cliques collection. */
  id: number;
  /** The identifiers for the nodes that belong to the clique.  This collection is
   * ordered such that the first n elements are the head variables of the clique's
   * conditional distribution.
   */
  domain: number[];
  /** The names of the variables in the clique.   This list is ordered in the same
   * order as the domain.
   */
  nodeNames: string[];
  /** The number of head variables in the clique's conditional distribution */
  numberOfHeadVariables: number;
  /** The number of levels of each of the variables in the clique's domain.  This
   * list is ordered in the same order as the domain.
   */
  numbersOfLevels: number[];
  /** The levels of each variable that occurs in the clique's domain.   This list is
   * ordered first by variable, then by level.
   */
  levels: string[][];
  /** The number of rows in each block of the clique's conditional probability
   * distribution
   */
  blockSize: number;
  /** The conditional probability distribution for the clique */
  conditional: FastPotential;
};

/** Given a clique, collect the information necessary to generate
 * random observations from the clique.   Specifically, we need to
 * refactor the clique's posterior joint probability distribution,
 * P(X) to a conditional probability distribution P(X - Y| Y) where
 * the parent variables are chosen to be any of the variables
 * which are members of any previously visited clique in the
 * traversal ordering.
 * @param clique: The clique of interest
 * @param nodes: The nodes of the original Bayes Network
 * @param potentials: The posterior (consistent) potentials of the
 *   Bayes Network
 * @param formulas: The collection of symbolic representations of
 *   each of the posterior distributions in the network.
 * @param visitedNodes: A collection of node identifiers for those
 *   nodes which belong to one or more of the cliques that have
 *   already been visited.
 */
function getCliqueInfo (
  clique: FastClique,
  nodes: FastNode[],
  potentials: (FastPotential | null)[],
  formulas: Formula[],
  visitedNodes: number[],
): CliqueInfo {
  const { domain: formulaDomain } = formulas[clique.posterior]
  const [headVariables, parentVariables] = partition(
    x => !visitedNodes.includes(x),
    formulaDomain,
  )
  if (headVariables.length === 0) {
    throw new Error(
      'Cannot generate random sample.  Clique has no head variables.',
    )
  }
  const conditionalDomain = headVariables.concat(parentVariables)
  const numbersOfLevels = conditionalDomain.map(i => nodes[i].levels.length)
  if (numbersOfLevels.some(x => x === 0)) {
    throw new Error(
      'Cannot generate random sample.  Some of the variables have no levels.',
    )
  }

  const numberOfHeadVariables = headVariables.length
  // construct the conditional distribution for the clique from the posterior
  // joint probability distribution.
  evaluateFormula(
    clique.posterior,
    nodes,
    formulas,
    potentials,
    formulas.length,
    formulas.map(x => x.size),
    nodes.map(() => null),
  )
  const posterior = potentials[clique.posterior] as FastPotential
  const posteriorDomain = formulas[clique.posterior].domain
  const posteriorNumLvls = posteriorDomain.map(x => nodes[x].levels.length)
  const blockSize = product(headVariables.map(x => nodes[x].levels.length))
  // compute the potential of the parents by marginalizing the posterior joint
  // probability distribution.
  const parentNumOfLvls = parentVariables.map(x => nodes[x].levels.length)
  const parentSize = product(parentNumOfLvls)
  // ensure that the clique potential is computed!
  const parentPotential = evaluateMarginalPure(
    posterior,
    posteriorDomain,
    posteriorNumLvls,
    parentVariables,
    parentNumOfLvls,
    parentSize,
    false,
  )
  let conditional: FastPotential = []
  // The order of the variables in the posterior may not be in the
  // head/parent variable order of the conditional.   if required,
  // we permute the elements of the posterior to put them into the
  // correct order.
  const permuted = posteriorDomain.some((x, i) => x !== conditionalDomain[i])
    ? posterior.map((_, i) => {
      const permutedCombo = indexToCombination(i, numbersOfLevels)
      const posteriorCombo = conditionalDomain.map(
        j => permutedCombo[posteriorDomain.indexOf(j)],
      )
      const posteriorIndex = combinationToIndex(
        posteriorCombo,
        posteriorNumLvls,
      )
      return posterior[posteriorIndex]
    })
    : posterior
  for (let offset = 0; offset * blockSize < posterior.length; offset++) {
    // We proceed block by block of the conditional distribution, ensuring that
    // each block is normalized and satisifies P(X|Y) = P(X,Y) / P(Y).
    const block = permuted.slice(offset * blockSize, (offset + 1) * blockSize)
    conditional = conditional.concat(
      normalize(
        block.map(
          p =>
            // This ugliness is to avoid division by zero when the probability
            // of the parent event is zero (which can occur when the user
            // provides inconsistent evidence).   This will populate the block
            // with a uniform distribution so that it can produce any
            // possible outcome of head variables.   However, it may change the
            // probability of extremely rare events.
            Math.round(
              ((p + 1e-64) / (parentPotential[offset] + 1e-64)) * 1e16,
            ) / 1e16,
        ),
      ),
    )
  }
  return {
    id: clique.id,
    domain: conditionalDomain,
    nodeNames: conditionalDomain.map(i => nodes[i].name),
    numberOfHeadVariables,
    numbersOfLevels,
    levels: conditionalDomain.map(i => nodes[i].levels),
    blockSize,
    conditional,
  }
}

/** Aggregate the information about each clique and determine a
 * forward traversal order that can be used for the generation of
 * random samples.
 * @param engine: The inference engine containing the Bayes Network
 *   of interest.
 * @returns: A list of clique information objects in ascending
 *   traversal order.
 */
function getTraversalOrder (engine: InferenceEngine): CliqueInfo[] {
  // deconstruct the inference engine to facilitate working with
  // the hidden parameters.
  const {
    _cliques,
    _potentials,
    _nodes,
    _connectedComponents,
    _formulas,
  } = engine.toJSON()
  // Perform a topological sorting on each of the connected components
  // of the clique graph (junction tree graph), choosing an arbitrary
  // clique from the connected connected components and peforming a
  // depth first search starting from that clique.

  const result: CliqueInfo[] = []
  _connectedComponents.forEach(cliques => {
    // keep track of the cliques that have been visited to prevent
    // infinite looping.
    const visitedCliques: number[] = []
    let visitedNodes: number[] = []
    // pick a root clique and initialize the queue.
    let queue = [cliques[0]]
    while (queue.length > 0) {
      const cliqueId = queue.shift() as number
      const clique: FastClique = _cliques[cliqueId]
      const info = getCliqueInfo(
        clique,
        _nodes,
        _potentials,
        _formulas,
        visitedNodes,
      )
      visitedCliques.push(cliqueId)
      visitedNodes = visitedNodes.concat(
        info.domain.slice(0, info.numberOfHeadVariables),
      )
      result.push(info)
      queue = queue.concat(
        clique.neighbors.filter(x => !visitedCliques.includes(x)),
      )
    }
  })
  return result
}

function getRandomIndex (block: FastPotential): number {
  let r = Math.random()
  for (let i = 0; i < block.length; i++) {
    // if this level is the one randomly selected, then return it.
    if (r <= block[i]) return i
    // otherwise, decrease the random value by the probability of this level
    // and move on to the next level for the head variable.
    r -= block[i]
  }
  // We should not be able to get here, but if we do, then return the last
  // level for the head variable.
  return block.length - 1
}

function getRandomObservation (
  cliqueInfo: CliqueInfo[],
): Record<string, string> {
  const fastObservation: Record<number, number> = {}
  const result: Record<string, string> = {}
  cliqueInfo.forEach(
    ({
      domain,
      numberOfHeadVariables,
      numbersOfLevels,
      levels,
      blockSize,
      conditional,
      nodeNames,
    }) => {
      const parentCombo = domain
        .slice(numberOfHeadVariables)
        .map(i => fastObservation[i])
      const offset = combinationToIndex(
        parentCombo,
        numbersOfLevels.slice(numberOfHeadVariables),
      )
      const block = conditional.slice(
        offset * blockSize,
        (offset + 1) * blockSize,
      )
      const headIdx = getRandomIndex(block)
      const headCombo = indexToCombination(
        headIdx,
        numbersOfLevels.slice(0, numberOfHeadVariables),
      )
      headCombo.forEach((lvlIdx, headIdx) => {
        const nodeIdx = domain[headIdx]
        const nodeName = nodeNames[headIdx]
        fastObservation[nodeIdx] = lvlIdx
        result[nodeName] = levels[headIdx][lvlIdx]
      })
    },
  )

  return result
}

/** Generate a random sample from a Bayes network such that it
 * reflects any of the currently provided evidence.   This method
 * uses a variation on forward prior sampling, where we traverse
 * the clique graph (rather than the Bayes Network itself).  This
 * ensures that evidence is propigated to all nodes which are
 * d-connected.
 * @param engine: The inference engine containing the Bayes Network
 *   to sample
 * @param sampleSize: The number of observations in the requested
 *   sample.
 * Note: This function will throw an error if the posterior
 *   clique distributions cannot be computed, the sample size is
 *   less than zero, or if any of nodes have no levels.
 */
export function getRandomSample (
  engine: InferenceEngine,
  sampleSize: number,
): Record<string, string>[] {
  // Basic sanity checks
  if (sampleSize < 0) {
    throw new Error(
      'Cannot generate random sample.   Sample size must be greater than zero.',
    )
  }
  if (sampleSize === 0) return []
  // Force the computation of all the posterior joint distributions for the cliques
  engine.inferAll()
  const cliqueInfo: CliqueInfo[] = getTraversalOrder(engine)
  return Array(sampleSize)
    .fill(null)
    .map(() => getRandomObservation(cliqueInfo))
}
