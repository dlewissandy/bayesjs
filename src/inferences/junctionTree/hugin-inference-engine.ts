import { IInferenceEngine, ISepSet, INode, INetworkResult, ICptWithParents, ICptWithoutParents, INetwork, ICombinations, ICliquePotentials, IClique, IGraph, INodeResult } from '../../types'
import createCliques from './create-cliques'
import createInitialPotentials from './create-initial-potentials'
import propagatePotentials from './propagate-potentials'
import { filterCliquePotentialsByNodeCombinations, filterCliquesByNodeCombinations, getCliqueWithLessNodes, getNodesFromNetwork, getNodeStates, mapPotentialsThen, normalizeCliquePotentials, propIsNotNil } from '../../utils'
import { clone, propEq, sum } from 'ramda'
import { getConnectedComponents } from '../../utils/connected-components'

export class HuginInferenceEngine implements IInferenceEngine {
  private _network: INetwork;
  private _evidence: ICombinations;
  private _potentials: ICliquePotentials = {}
  private _cliques: IClique[];
  private _sepSets: ISepSet[];
  private _junctionTree: IGraph;
  private _marginals: INetworkResult = {}
  private _connectedComponents: string[][];

  constructor (network: INetwork) {
    this._network = network
    this._evidence = {}
    const { cliques, sepSets, junctionTree } = createCliques(network)
    this._connectedComponents = getConnectedComponents(junctionTree)
    this._cliques = cliques
    this._sepSets = sepSets
    this._junctionTree = junctionTree
  }

  private resetCache = () => {
    this._potentials = {}
    this._marginals = {}
  }

  hasVariable = (name: string) => this._network[name] != null;
  getVariables = () => Object.keys(this._network);

  hasLevel = (name: string, level: string) => {
    if (!this._network[name]) return false
    const found = this._network[name].states.filter(x => x === level)
    return found !== []
  }

  getLevels = (name: string) => {
    if (!this._network[name]) return []
    return [...this._network[name].states]
  }

  hasParent = (name: string, parent: string) => {
    if (!this._network[name]) return false
    const found = this._network[name].parents.filter(x => x === parent)
    return found !== []
  }

  getParents = (name: string) => {
    if (!this._network[name]) return []
    return [...this._network[name].parents]
  }

  getDistribution = (name: string) => clone(this._network[name].cpt)

  setDistribution = (name: string, cpt: ICptWithParents | ICptWithoutParents) => {
    const node: INode = this._network[name]
    const expectedLevels = node.states.sort()
    const expectedParents = node.parents.sort()

    const err = (reason: string) => {
      throw new Error(`Cannot set the distribution for ${name}.  ${reason}.`)
    }
    if (!node) err('The variable does not exist in the network')

    const observedLevels = Array.isArray(cpt)
      ? cpt.length > 0
        ? Object.keys(cpt[0].when).sort()
        : []
      : Object.keys(cpt).sort()

    const observedParents = Array.isArray(cpt)
      ? cpt.length > 0
        ? Object.keys(cpt[0].then).sort()
        : []
      : []

    if (observedLevels !== expectedLevels) err('The provided distribution did not have the expected levels.')
    if (observedParents !== expectedParents) err('The provided distribution did not have the expected parents.')

    this.resetCache()
    node.cpt = clone(cpt)
  }

  private getCliquesPotentials: () => ICliquePotentials = () => {
    if (Object.keys(this._potentials).length === 0) {
      const initialPotentials = createInitialPotentials(this._cliques, this._network, this._evidence)
      const propagatedPotentials = propagatePotentials(this._network, this._junctionTree, this._cliques, this._sepSets, initialPotentials, this._connectedComponents.map(x => x[0]))
      this._potentials = normalizeCliquePotentials(propagatedPotentials)
    }
    return this._potentials
  }

  cliques = () => this._cliques;
  sepSets = () => this._sepSets;

  hasEvidenceFor = (name: string) => this._evidence[name] != null;

  setEvidence = (evidence: { [name: string]: string }) => {
    this.removeAllEvidence()
    this.updateEvidence(evidence)
  }

  updateEvidence = (evidence: { [name: string]: string }) => {
    for (const [name, value] of Object.entries(evidence)) {
      const oldValue = this._evidence[name]
      if (oldValue == null || oldValue !== value) {
        this.resetCache()
      }
      this._evidence[name] = value
    }
  }

  removeEvidence = (name: string) => {
    const oldValue = this._evidence[name]
    if (oldValue != null) {
      this.resetCache()
      delete this._evidence[name]
    }
  }

  removeAllEvidence = () => {
    if (Object.keys(this._evidence).length > 0) {
      this.resetCache()
      this._evidence = {}
    }
  }

  infer = (event: ICombinations) => {
    const cliquesPotentials = this.getCliquesPotentials()
    const cliquesNode = filterCliquesByNodeCombinations(this._cliques, event)
    const clique = getCliqueWithLessNodes(cliquesNode)
    const potentials = cliquesPotentials[clique.id]
    const potentialsFiltered = filterCliquePotentialsByNodeCombinations(potentials, event)
    const thens = mapPotentialsThen(potentialsFiltered)

    return sum(thens)
  }

  inferAll = () => {
    const given = this._evidence
    const network = this._network
    if (Object.keys(this._marginals).length === 0) {
      for (const node of getNodesFromNetwork(network)) {
        const marginal: INodeResult = {}
        const nodeId = node.id
        for (const state of getNodeStates(node)) {
          if (propIsNotNil(nodeId, given)) {
            marginal[state] = propEq(nodeId, state, given) ? 1 : 0
          } else {
            marginal[state] = this.infer({ [nodeId]: state })
          }
        }
        this._marginals[node.id] = marginal
      }
    }
    return this._marginals
  }
}
