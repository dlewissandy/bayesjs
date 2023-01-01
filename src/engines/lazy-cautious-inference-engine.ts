import {
  INetworkResult,
  IInferenceEngine, IInferAllOptions, ICptWithParents, ICptWithoutParents,
} from '../types'
import { clone, uniq } from 'ramda'
import roundTo = require('round-to')

import { FastPotential, indexToCombination } from './FastPotential'
import { FastClique } from './FastClique'
import { FastNode } from './FastNode'
import { NodeId, FormulaId, FastEvent } from './common'
import { Formula, EvidenceFunction, updateReferences } from './Formula'
import { propagatePotentials } from './symbolic-propagation'
import { evaluate } from './evaluation'
import { getNetworkInfo, initializeCliques, initializeEvidence, initializeNodeParents, initializeNodes, initializePosteriorCliquePotentials, initializePosteriorNodePotentials, initializePriorNodePotentials, initializeSeparatorPotentials, NetworkInfo, upsertFormula, setDistribution, pickRootClique, kahanSum } from './util'
import { Distribution } from './Distribution'
import { evaluateJoinPotentials } from './evaluate-join-potential'
import { inferJoinProbability } from './evaluate-join-probability'
import { getRandomSample } from './random-sample'

/** This inference engine uses a modified version of the lazy cautious message
 * propigation strategy described in:
 * "Lazy Propagation: A Junction Tree Inference Algorithm based on Lazy evaluation"
 * by Madsen and Jensen.
 *
 * This implementation extends the algorithm described by the authors by using
 * a symbolic message passing architecture.   Message passing populates a
 * collection of formulas for each message and posterior marginals using the
 * AST that is made explicit in the Formula type.  The syntax of the ASTs has
 * been chosen so that it can be performed once, and need not be updated with
 * changes in either hard or soft evidence.
 *
 * The potentials are evaluated upon demand whenever the infer function is
 * called.   The inference results and the results of any intermediate
 * computations are stored in the potentials cache to facilitate subsequent
 * inferences.   This separation of concerns between message passing and
 * evaluation allows not only fast retraction of hard evidence, but also
 * replacement of potential functions for individual nodes of the Bayes
 * network without invalidating the entire cache.
 *
 */
export class LazyPropagationEngine implements IInferenceEngine {
  private _cliques: FastClique[]
  private _nodes: FastNode[]
  private _potentials: (FastPotential | null)[]
  private _formulas: Formula[]
  private _connectedComponents: NodeId[][]
  private _separators: NodeId[][]
  private _separatorPotentials: FormulaId[] = []

  /** This function recursively clears the cached potentials starting from the given
   * formula id.   It terminates the recursion down any branch when it encounters
   * a cached value that is null, because this implies that all cached values that
   * are depends of that potential are also null.
   */
  private clearCachedValues = (root: FormulaId) => {
    const f = this._formulas[root]
    const p = this._potentials[root]
    // If the cache rooted on this node has already been cleared, then
    // exit.
    if (!p) return
    // otherwise, clear this node and (recursively) all its children
    this._potentials[root] = null
    f.refrerencedBy.forEach(childId => this.clearCachedValues(childId))
  }

  constructor (network: {[name: string]: {
    levels: string[];
    parents: string[];
    potentialFunction?: FastPotential;
    distribution?: Distribution;
    cpt?: ICptWithParents | ICptWithoutParents;
  };}) {
    const info: NetworkInfo = getNetworkInfo(network)
    const { connectedComponents, separators } = info
    const { nodeMap, cliqueMap } = info
    const { numberOfCliques, numberOfNodes } = info

    const fs: Formula[] = []
    const ns: FastNode[] = Array(numberOfNodes)
    const cs: FastClique[] = Array(numberOfCliques)
    const ccs: number[][] = connectedComponents.map(cc => cc.map(cliqueName => cliqueMap[cliqueName]))
    const upsert = upsertFormula(fs, {})

    initializeNodes(network, nodeMap, upsert, ns)
    initializeNodeParents(ns)
    initializeEvidence(ns, upsert)
    initializeCliques(info, upsert, ns, cs, fs)
    const messages = propagatePotentials(cs, separators, upsert, fs, info.roots)
    initializePosteriorCliquePotentials(upsert, ns, cs, messages, fs)
    this._separatorPotentials = initializeSeparatorPotentials(info, upsert, cs, fs)
    initializePosteriorNodePotentials(upsert, ns, cs, fs, this._separatorPotentials)
    const ps: FastPotential[] = new Array(fs.length)
    initializePriorNodePotentials(network, ps, ns)
    updateReferences(fs)

    // Set the values of the private properties.
    this._nodes = ns
    this._cliques = cs
    this._formulas = fs
    this._potentials = ps
    this._connectedComponents = ccs
    this._separators = separators
  }

  // Implmentation of the hasVariable interface function.  Returns true
  // if and only if there is a variable with the given name
  hasVariable = (name: string) => this._nodes.map(x => x.name).includes(name)

  // Implementation of the getVariables interface function.
  getVariables = () => this._nodes.map(x => x.name)

  // Implementation of teh getParents interface function.
  getParents = (name: string) => {
    const node = this._nodes.find(x => x.name === name)
    if (!node) return []
    return node.parents.map(i => this._nodes[i].name)
  }

  // Implementation of the hasParent interface function
  hasParent = (name: string, parent: string) => this.getParents(name).includes(parent)

  // implementation of the getLevels interface function
  getLevels = (name: string) => {
    const node = this._nodes.find(x => x.name === name)
    if (!node) return []
    return node.levels
  }

  // implementation of the hasLevel interface function
  hasLevel = (name: string, level: string) => this.getLevels(name).includes(level)

  // implementation of the getDistribution interface function.
  // this returns the prior local distribution for the given variable.
  getPriorDistribution = (name: string) => {
    const headNode = this._nodes.find(x => x.name === name)
    if (headNode == null) throw new Error(`Cannot get prior distribution for ${name}.  Variable does not exist.`)
    const parents = headNode?.parents.map(i => this._nodes[i]).map(p => ({ name: p.name, levels: p.levels })) || []
    const potentials = this._potentials[headNode.id] || Array(this._formulas[headNode.id].size).fill(1 / headNode.levels.length)
    const headLevels = headNode.levels
    return new Distribution([{ name, levels: headLevels }], parents, potentials)
  }

  getPosteriorDistribution = (name: string) => {
    const headNode = this._nodes.find(x => x.name === name)
    if (headNode == null) throw new Error(`Cannot get posterior distribution for ${name}.  Variable does not exist.`)
    const parents = headNode?.parents.map(i => this._nodes[i]).map(p => p.name) || []
    return this.getJointDistribution([name], parents)
  }

  /**
   * Compute the joint distribution for a collection of variables, optionally conditioned upon
   * a collection of parent variables.   This algorithm uses a mondified symbolic message passing
   * scheme to reuse as many of the previously computed potentials as possible..
   * @param headVariables A non-empty list of variables in the join
   * @param parentVariables A possibly empty list of parent variables.   The parent and head
   *   variables must be disjoint.
   * @returns a new distribution object with the requested conditioned joint distribution.
   * @throws This function throws if either the head or parent variables are not distinct, or if they contain
   *   repeated elements or elements that are not variables of the Bayes network.   Also
   *   throws when the parent and head variables are not disjoint.
   */
  getJointDistribution = (headVariables: string[], parentVariables: string[]): Distribution => {
    const parentIdxs: number[] = parentVariables.map(s => this._nodes.findIndex(node => node.name === s))
    const headIdxs: number[] = headVariables.map(s => this._nodes.findIndex(node => node.name === s))
    const { potentials } = evaluateJoinPotentials(this._nodes, this._cliques, this._connectedComponents, this._separators, this._formulas, this._potentials, headIdxs, parentIdxs)
    const dist = new Distribution(headIdxs.map(n => this._nodes[n]), parentIdxs.map(n => this._nodes[n]), potentials)
    return dist
  }

  // implementation of the setDistribution interface function.
  // note that this clears the cache for any potential that is dependendent
  // either directly or indirectly on this value.
  setDistribution = (distribution: Distribution): boolean => {
    const nodeId = setDistribution(distribution, this._nodes, this._potentials)
    const p = this._potentials[nodeId]
    this.clearCachedValues(nodeId)
    this._potentials[nodeId] = p
    return true
  }

  // implementation of the hasEvidenceFor interface function.
  hasEvidenceFor = (name: string) => {
    const node = this._nodes.find(x => x.name === name)
    if (!node) {
      return false
    }
    return (this._formulas[node.evidenceFunction] as EvidenceFunction).levels != null
  }

  // implementation of the getEvidence interface function
  getEvidence = (name: string) => {
    let result: string[] | null = null
    const node = this._nodes.find(x => x.name === name)
    if (node) {
      const lvls = (this._formulas[node.evidenceFunction] as EvidenceFunction).levels
      if (lvls !== null) result = lvls.map(lvl => node.levels[lvl])
    }
    return result
  }

  // Get all the evidence provided for the network
  getAllEvidence = () => {
    const result: { [name: string]: string[]} = {}
    this.getVariables().forEach(name => {
      const ls: string[] | null = this.getEvidence(name)
      if (ls && ls.length > 0) { result[name] = ls }
    },
    )
    return result
  }

  /**
   * Update the evidence for the given variable, replacing the existing
   * evidence for that variable with the new value.   Evidence for any other
   * variables is left unchanged.
   * @param nodeId the identifier for the node for which the evidence is
   *   being updated
   * @param levelIds The identifiers for the levels ove the variable which
   *   are evidenced
   * NOTE: This function assumes that the nodeId and levelIds are valid and
   *    that the level identifiers are distinct.
   *   These preconditions are not checked, and must be ensured by the caller.
   */
  private updateEvidencePrimitive = (nodeId: number, levelIds: number[]) => {
    const node = this._nodes[nodeId]
    const evidenceFunc = this._formulas[node.evidenceFunction] as EvidenceFunction
    const newLevels = levelIds.sort()
    if (!evidenceFunc.levels || evidenceFunc.levels.length !== newLevels.length || evidenceFunc.levels !== newLevels) {
      evidenceFunc.levels = newLevels
      this.clearCachedValues(evidenceFunc.id)
    }
  }

  /**
   * Update the evidence for the given variables, replacing the existing
   * evidence for that variable with the new value.   Evidence for any other
   * variables is left unchanged.
   * @param evidence A maping from variables to evidenced values for the
   *   variable.
   */
  updateEvidence = (evidence: { [name: string]: string[]}) => {
    Object.keys(evidence).forEach(name => {
      const node = this._nodes.find(x => x.name === name)
      if (node) {
        const lvlIdxs: number[] = uniq(evidence[name].map(l => node.levels.indexOf(l)))

        if (lvlIdxs.length > 0 && lvlIdxs[0] === -1) throw new Error(`Cannot update the evidence.   One of levels provided for ${node.name} is not valid.`)
        if (lvlIdxs.length > 0) {
          this.updateEvidencePrimitive(node.id, uniq(lvlIdxs))
        }
      }
    })
  }

  // Implementation of the setEvidence interface function.
  setEvidence = (evidence: { [name: string]: string[]}) => {
    this.removeAllEvidence()
    this.updateEvidence(evidence)
  }

  /** Remove any hard evidence for the given variable.   If the
   * variable has no hard evidence, then the cache will remain
   * unchanged.   Otherwise, all cached values that depend either directly
   * or indirectly on the evidence will be cleared.  NOTE:
   * This could be further improved by using the d-connecteness
   * properties of the nodes;  this is left for future work.
   */
  removeEvidence = (name: string) => {
    const node = this._nodes.find(x => x.name === name)
    if (node) {
      const evidenceFunc = this._formulas[node.evidenceFunction] as EvidenceFunction
      if (evidenceFunc.levels != null) {
        evidenceFunc.levels = null
        this.clearCachedValues(evidenceFunc.id)
      }
    }
  }

  // Remove all evidence from the cache.   Any nodes that depend on
  // hard evidence will be cleared.
  removeAllEvidence = () =>
    this._nodes.forEach(node => {
      const evidenceFunc = this._formulas[node.evidenceFunction] as EvidenceFunction
      if (evidenceFunc.levels != null) {
        evidenceFunc.levels = null
        this.clearCachedValues(evidenceFunc.id)
      }
    })

  /** Given a single node,  infer the probability of an event from the
   * posterior marginal distribution for that node.
   */
  private inferFromMarginal (nodeId: NodeId, levels: number[]): number {
    const p = evaluate(this._nodes[nodeId].posteriorMarginal, this._nodes, this._formulas, this._potentials)
    return kahanSum(levels.map((level) => p[level]))
  }

  /**
   * Infer the probability of an event where all of the nodes appear in the given clique.
   * @param clique A clique containing all the nodes in the event.   This precondition is not
   *   checked, and must be verified by the caller
   * @param event The event, consisting of a list of variable/level pairs representing the
   *   possible values for the variable. When more than one possible values are provided
   *   for a variable, the cumulative probability over the individual outcomes will be
   *   computed.   When more than one outcome is provided for multiple variables, the
   *   cumulative over all possible combinations of those outcomes will be computed.
   * @returns The probability of the event.
   * NOTE: This function assumes that the event does not contain more than one element for
   *   each variable, that all the variables are in the clique, and that the levels are all
   *   levels of the variable.   These preconditions are not checked, and must be ensured
   *   by the caller.
   */
  private inferFromClique (clique: FastClique, event: { nodeId: number; levels: number[]}[]): number {
    const formulaId = clique.posterior
    const formula = this._formulas[formulaId]
    const nodePositions = event.map(({ nodeId }) => formula.domain.indexOf(nodeId))
    const potential = evaluate(formulaId, this._nodes, this._formulas, this._potentials)
    let total = 0
    potential.forEach((p, i) => {
      const combos = indexToCombination(i, formula.numberOfLevels)
      if (nodePositions.every((nodePosition, j) => event[j].levels.includes(combos[nodePosition]))) total += p
    })
    return total
  }

  /** Given a collection of nodes and levels representing an event, construct the
   * joint probability distribution distribution on the given nodes by creating a
   * new potential function that "adds the fill in edges" between the cliques
   * that contain the nodes.
   * @param - the event for which we wish to infer the probability.
   * NOTE: This algorithm only works when there is no evidence baked into the
   * network.  When this is the case, the evidence must be retracted and then
   * restored afterward.
   */
  private inferFromJointPropagation (event: FastEvent[]): number {
    /**
     * This helper function converts the fast event into a join domain
     * and a collection of values for each of the variables in the join
     * @param event
     */
    const refactorEvent = (event: FastEvent[]): { joinDomain: number[]; values: (number[]|null)[] } => {
      const joinDomain = event.map(({ nodeId }) => nodeId)
      const values = this._nodes.map(node => {
        const e = event.find(entry => entry.nodeId === node.id)
        return e?.levels ?? null
      })
      return { joinDomain, values }
    }

    // get the initial evidence.   If there is evidence it will need to be
    // retracted and restored.
    const initialEvidence = this.getAllEvidence()
    if (Object.keys(initialEvidence).length === 0) {
      // in the case where there is no evidence, we can simply compute it with the
      // special purpose message passing algorithm
      const { joinDomain, values } = refactorEvent(event)
      const { joinProbability } = inferJoinProbability(this._nodes, this._cliques, this._connectedComponents, this._separators, this._formulas, this._potentials, joinDomain, values)
      return joinProbability
    } else {
      // Since evidence has been provided for the network, we are really computing
      // a joint probability distribution conditioned upon some arbitrary collection
      // of parents.   if X is the event, and Y is the evidence, we can use the equality
      // P(X,Y) = P(X|Y)P(Y) to compute that conditional.  This requies us to compute
      // to joins, one over the parents and one over the union of the event and parents.
      // all of this must be done without any evidence baked into the network.

      // We start by caching the initial potentials with the baked in evidence, so that
      // they can be restored at the end.
      const initialPotentials = [...this._potentials]

      // construct a fast event that represents the evidence for the parents.
      const evidenceAsEvent: FastEvent[] = []
      const eventDomain: number[] = event.map(x => x.nodeId)
      this._nodes.forEach(node => {
        const e: EvidenceFunction = this._formulas[node.evidenceFunction] as EvidenceFunction
        if (e.levels && e.levels.length > 0) evidenceAsEvent.push({ nodeId: e.nodeId, levels: e.levels })
      })
      this.removeAllEvidence()

      // Since all the work to ensure that the event does not contain
      // values in conflict with the evidence, we can infer the probability of
      // we can use Bayes theorem to compute the joint joint probability by
      // calling inferJoint twice; once with the join over the event and
      // evidence, and once for the join over just the evidence (parents).
      // From the ratio of these two values we get the requested conditional
      // probability.
      const joinEvent = event.concat(evidenceAsEvent.filter(e => !eventDomain.includes(e.nodeId)))
      const { joinDomain, values: joinValues } = refactorEvent(joinEvent)
      const { joinProbability } = inferJoinProbability(this._nodes, this._cliques, this._connectedComponents, this._separators, this._formulas, this._potentials, joinDomain, joinValues)
      const { joinDomain: parentDomain, values: parentValues } = refactorEvent(evidenceAsEvent)
      const { joinProbability: parentProbability } = inferJoinProbability(this._nodes, this._cliques, this._connectedComponents, this._separators, this._formulas, this._potentials, parentDomain, parentValues)
      const result = joinProbability === 0 ? 0 : joinProbability / parentProbability

      // finally we restore the evidence and potentials to the original values.
      this.setEvidence(initialEvidence)
      this._potentials = initialPotentials
      return result
    }
  }

  /**
   * Infer the probability of an event subject to the currently
   * provided evidence.
   * @param event The event, consisting of a mapping from variable
   *   names to possible values.   When more than one possible value
   *   is provided for a variable, the cumulative probability over the
   *   individual outcomes will be computed.   When more than one
   *   outcome is provided for multiple variables, the cumulative over
   *   all possible combinations of those outcomes will be computed.
   * @returns The probability of the event.   When the event contains
   *   variables that are not part of the network, or when a level for a
   *   variable is provided which is not a level of that variable, or
   *   if the provided levels are contradictory to the provided
   *   evidence for the variable, then this function will return 0.
   *   If the event contains no mappings from variable to level, this
   *   function will return unity.
   * */
  infer = (event: { [name: string]: string[]}) => {
    const names = Object.keys(event)
    // If the empty event has been provided, then the probability is
    // trivially equal to unity.
    if (names.length === 0) return 1

    // If some names have been provided, then we need to find out
    // which nodes they belong to.
    const joinDomain = names.map(name => this._nodes.findIndex(node => node.name === name))

    // If there are some names that were provided that do not belong to
    // any nodes, then the probability of the event is 0.
    if (joinDomain.some(nodeId => nodeId === -1)) return 0

    // If we reached here, all the names are actually variables/nodes in the
    // Bayes network.   We need to check that all the provided levels are
    // valid outcomes for each variable.
    const levels: number[][] = names.map((name, i) => {
      const lvls = event[name] || []
      const node = this._nodes[joinDomain[i]]
      // ensure that the levels are distinct and that they are levels of the
      // variable.
      const uniqueValidLevels = uniq(lvls.map(x => node.levels.indexOf(x)).filter(x => x >= 0))
      // ensure that the levels are not contradictory to the provided
      // evidence
      const evidence = this._formulas[node.evidenceFunction] as EvidenceFunction
      return uniqueValidLevels.filter(x => evidence.levels ? evidence.levels.includes(x) : true)
    })

    // In the case where any of the levels that were provided are not valid
    // outcomes for the corresponding variable, then the probability
    // is exactly zero.
    if (levels.some(lvl => lvl.length === 0)) return 0

    // If we reached here, then all the names and levels are valid.
    // If only one name has been provided, then we are tasked with inferring
    // the marginal probability for the event.
    if (joinDomain.length === 1) return this.inferFromMarginal(joinDomain[0], levels[0])

    // If all the nodes are in the same clique, then we can efficiently
    // infer the probability from the posterior clique potential.
    const clique = pickRootClique(this._cliques, joinDomain, this._formulas)
    const fastEvent = joinDomain.map((nodeId, i) => ({ nodeId, levels: levels[i] }))
    if (joinDomain.every(i => clique.domain.includes(i))) {
      return this.inferFromClique(clique, fastEvent)
    }

    // Otherwise, we are being tasked with finding a joint probability that
    // spans multiple cliques.
    return this.inferFromJointPropagation(fastEvent)
  }

  inferAll = (options?: IInferAllOptions) => {
    const result: INetworkResult = {}
    this._nodes.forEach(fastnode => {
      result[fastnode.name] = {}
      fastnode.levels.forEach(level => {
        const event: { [level: string]: string[]} = {}
        event[fastnode.name] = [level]
        const p = this.infer(event)
        if (options && options.precision != null && options.precision > 0) {
          result[fastnode.name][level] = roundTo(p, options.precision)
        } else {
          result[fastnode.name][level] = p
        }
      })
    })
    return result
  }

  /** Get a random sample from the Bayes Network subject to any
   * of the currently provided evidence.
   * @param size: The sample size to return.
   * Note: This function will throw an error if the sample size
   *  is less than zero, or if the Bayes network posterior
   *  distributions cannot be computed, or if the network is
   *  otherwise ill formed.
   */
  getRandomSample (size: number): Record<string, string>[] {
    // Some sanity checks
    if (size < 0) throw new Error('Cannot generate random sample.   Sample size must be greater than zero.')
    if (size === 0) return []
    // ensure that all the clique potentials have been computed.
    return getRandomSample(this, size)
  }

  // This is a back door for obtaining all the private collections in the inference engine.
  // This function has been provided to aid in writing additonal tests and for persisting
  // inference engines across instances.
  toJSON = () => clone({
    _class: 'LazyCautiousInferenceEngine',
    _cliques: this._cliques,
    _nodes: this._nodes,
    _potentials: this._potentials,
    _formulas: this._formulas,
    _connectedComponents: this._connectedComponents,
    _separators: this._separators,
    _separatorPotentials: this._separatorPotentials,
  })
}
