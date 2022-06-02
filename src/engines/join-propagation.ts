import { FastClique } from './FastClique'
import { Formula, EvidenceFunction, FormulaType, reference, marginalize, mult } from './Formula'
import { uniqBy, uniq, reduce, partition } from 'ramda'
import { FastNode } from './FastNode'
import { pickRootClique, upsertFormula } from './util'

// This helper function is used to create a distinct identifier for messages
// passed between two cliques.
const messageName = (sourceId: number, targetId: number) => `Ξ(${sourceId},${targetId})`

/**
 * Compute the message passed between a source clique and a target clique.
 * The message is repersented as a lazy product of factors, similar to
 * the lazy message propigation strategy of Madsen and Jensen.   However
 * the message passing has been modified so that it does not marginalize
 * away variables in the joinDomain
 * @param sepSet - The separation set between the source and target cliques
 * @param messages - A dictionary of messages that have been passed between adjacent
 *   cliques.   This dictionary will be modified by the message passing algorithm
 * @param upsert- A function for inserting new formulas into the formulas array,
 *   but only if they are distinct.
 * @param formulas - A list of all the formulas for posterior joint distributions for
 *   cliques, posterior marginal distributions for nodes, messages and intermediate
 *   factors.   This array may be appended by the message passing algorithm.
 * @param src The clique from which the message originates
 * @param trg The target clique into which the messages will be collected
 * @param joinDomain - A list of variables for which the joint distribution is to be
 *   computed.
 */
function passMessage (sepSet: number[], messages: Record<string, Formula[]>, upsert: (f: Formula) => Formula, formulas: Formula[], src: FastClique, trg: FastClique, joinDomain: number[]) {
  // Get all the messages that have been received by neighbors of the source
  // clique (other than the target).   These messages need to be passed on
  // to the target.
  const neighborMessages: Formula[] = []
  const neighborEvidence: EvidenceFunction[] = []
  src.neighbors.forEach((neighborId: number) => {
    if (neighborId !== trg.id) { // don't pass a message back to the sender!
      const msgs = messages[messageName(neighborId, src.id)]
      const [es, ms] = partition(m => m.kind === FormulaType.EVIDENCE_FUNCTION, msgs)
      neighborEvidence.push(...es as EvidenceFunction[])
      neighborMessages.push(...ms)
    }
  })
  // Construct the factors for the message that will be passed to the target.
  // Each factor is either the clique potential or one of the messages to
  // the source clique, marginalized to remove any nodes that are not in
  // common between the source and target cliques or in the join domain.
  const cliqueEvidence: EvidenceFunction[] = src.evidence.map(id => formulas[id] as EvidenceFunction)
  const evidence = uniqBy(x => x.id, [...cliqueEvidence, ...neighborEvidence])
  const factors: Formula[] = uniqBy(x => x.id, [reference(src.prior, formulas), ...neighborMessages])
  const keepers = [...new Set([...sepSet, ...joinDomain])]

  // The message that is passed from the source to the target is the
  // product of the factors, marginalized to remove any nodes that are not in
  // the set of keepers.  We can partition the factors into two groups,
  // those that contain in their domains some of the variables that must be removed, and those
  // that do not.   Those that contain the factors to remove must be joined
  // together prior to marginalization, since marginalization does not in general
  // distribute over join.   We initialize the msgs with those factors that do not
  // require marginalization, and then deal with any that do.
  const [dontRequireMarginalization, requireMarginalization] = partition<Formula>((f: Formula) => f.domain.every(x => keepers.includes(x)), factors)
  const msgs = [...dontRequireMarginalization, ...evidence.filter(x => dontRequireMarginalization.some(y => y.domain.includes(x.nodeId)))]
  if (requireMarginalization.length > 0) {
    // There are at least some factors that contain the variables that are
    // being marginalized.
    let marg: Formula
    const es = evidence.filter(x => requireMarginalization.some(y => y.domain.includes(x.nodeId)))
    const ms = [...requireMarginalization, ...es]
    if (ms.length === 1) {
      // If there is only one of them, then the product is trivial.  We can
      // marginalize that one element.
      marg = upsert(marginalize(keepers, ms[0], formulas))
    } else {
      // Otherwise, we need to construct the product.   We use upsert to avoid
      // creating a duplicate formula for both the product and its marginal.
      const prod = upsert(mult(ms))
      marg = upsert(marginalize(keepers, prod, formulas))
    }
    // once complete, we can add the marginal to the message factors being passed
    // between the source and target
    msgs.push(marg)
  }
  // Finally we mutate the messages dictionary to include the newly computed
  // messages.
  messages[messageName(src.id, trg.id)] = msgs
}

/**
 * Pass the messages "upward" through the clique graph to the selected root clique.
 * We use the same lazy message passing architecture used for making the clique
 * graph consistent, however we do not marginalize out variables that occur in the
 * desired joint distribution.
 * @param cliques - All the cliques in the connected component of the clique graph
 *   that contains the variables in the join domain.
 * @param separators - the separator sets (edges) between the cliques
 * @param messages - A dictionary of messages that have been passed between adjacent
 *   cliques.   This dictionary will be modified by the message passing algorithm
 * @param upsert  - A function for inserting new formulas into the formulas array,
 *   but only if they are distinct.
 * @param formulas - A list of all the formulas for posterior joint distributions for
 *   cliques, posterior marginal distributions for nodes, messages and intermediate
 *   factors.   This array may be appended by the message passing algorithm.
 * @param joinDomain - A list of variables for which the joint distribution is to be
 *   computed.
 * @param rootId - The chosen root clique for accumulating the messages.
 * @returns void.   This message passing routine mutates the messages and formulas
 *   and produces no other result.
 */
function collectCliquesEvidence (cliques: FastClique[], separators: number[][], messages: Record<string, Formula[]>, upsert: (f: Formula) => Formula, formulas: Formula[], joinDomain: number[], rootId: number) {
  /**
   * This helper function performs a recursive traversal to collect
   * the messages starting from the given clique identifier
   * @param id The identifier of the clique from which to start the
   *   traversal (or sub-traversal)
   * @param parentId The identifier of the parent node from which
   *   the traversal started.  This is undefined when the current
   *   clique is the root of the traversal.
   */
  const process = (id: number, parentId?: number) => {
    const src = cliques[id]
    const { neighbors } = src
    for (const neighbor of neighbors) {
      // recurse the neighbors, ignoring the parent.   This
      // computes all the messages passed to the neighbors
      if (parentId != null && neighbor === parentId) continue
      process(neighbor, id)
    }

    // If this is the root clique, then we don't need to pass any messages
    // up the chain
    if (parentId == null) return

    // Otherwise, we need to take each message received from the neighbors
    // and pass them on to the parent (possibly marginalizing to remove
    // terms that are neither in the separator set nor the join domain.)
    const trg = cliques[parentId]

    const neighborIndex: number = neighbors.findIndex(id => id === parentId)
    const sepSet = separators[src.separators[neighborIndex]]
    passMessage(sepSet, messages, upsert, formulas, src, trg, joinDomain)
  } //   // start processing from the best root node.
  // Perform the message passing starting from the root clique.
  process(rootId)
  return messages
}

/** Construct the join of an arbitrary collection of variables in a Bayesian Network,
 * conditioned on an optional set of parent variables.   This is performed by a
 * modified symbolic message passing strategy, whereby variables that appear in the
 * join are not marginlized out of the messages passed between cliques.
 * @param nodes: The collection of nodes in the Bayesian network
 * @param cliques: The collection of cliques in the junction tree for the Bayesian
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
 * @returns: The formula for the requested joint distribution and the symbolic
 *   repersentation of all the intermediate calculations that were not part of the
 *   original message passing.   Note that the joint distribution when evalatuated
 *   will be isomorphic to the requested join up to a permutation of the parent variables.
 */
export function propagateJoinMessages (nodes: FastNode[], cliques: FastClique[], connectedComponents: number[][], formulas: Formula[], separators: number[][], headVariables: number[], parentVariables: number[]): { joinFormulaId: number; supplementalFormulas: Formula[] } {
  const distinctHeadVariables = uniq(headVariables)
  const distinctParentVariables = uniq(parentVariables)

  // SANITY CHECKS HERE
  const throwErr = (reason: string) => { throw new Error('Cannot compute the join over the given head and parent variables.  ' + reason) }
  if (headVariables.length === 0) throwErr('No head variables were provided.')
  if (headVariables.length < distinctHeadVariables.length) throwErr('The head variables are not distinct.')
  if (parentVariables.length < distinctParentVariables.length) throwErr('The parent variables are not distinct.')
  if (distinctHeadVariables.some(x => distinctParentVariables.includes(x))) throwErr('The head and parent variables are not disjoint')
  if (distinctHeadVariables.some(x => x < 0 || x >= nodes.length)) throwErr('Some of the head variables do not exist in the network.')
  if (distinctParentVariables.some(x => x < 0 || x >= nodes.length)) throwErr('Some of the parent variables do not exist in the network.')
  const joinDomain = [...headVariables, ...parentVariables]

  // Create a copy of the formulas that we can use for performing the traversal.   This collection of formulas
  // may be updated during message passing with new formulas that were not part of the original
  // junction tree message passing.
  const amendedFormulas = [...formulas]

  // Create a list of connected components that have some of the variables of interest.
  // Each of these connected components will need to be traversed to compute the joint
  // distribution.
  const connectedComponentsThatHaveVariablesOfInterest = connectedComponents.filter(cc =>
    cc.some(x => cliques[x].domain.some(i => joinDomain.includes(i))),
  )

  // Initialize two dictionaries that will be used by the message passing
  // algorithm to find existing formulas and messages, and avoid duplication
  // of effort.
  const messages: Record<string, Formula[]> = {}
  const fmap: Record<string, number> = {}
  formulas.forEach((f, i) => {
    fmap[f.name] = i
  })
  // This helper function will be used during message passing to add new formulas
  // to the dictionary, but only if it doesn't already exist.
  const upsert = upsertFormula(amendedFormulas, fmap)

  // traverse each connected component, collecting all the messages into
  // a root clique.   This traversal will return the formula for the join of
  // all the variables that occur in each clique.
  const cliqueFormulas = connectedComponentsThatHaveVariablesOfInterest.map(cc => {
    // get the cliques and variables that are in this connected component.
    const theseCliques = cc.map(i => cliques[i])
    const theseVariables = joinDomain.filter(i => theseCliques.some(c => c.domain.includes(i)))

    // find a best root clique into which the messages will be collected
    const rootClique = pickRootClique(theseCliques, theseVariables, amendedFormulas)

    // recursively collect the evidence from the neihgbor cliques.   This message passing
    // has been modified from the one used for making the clique graph consistent in that
    // it does not marginalize out the variables in the join domain.  The clique evidence
    // does not need to be distributed out to the other nodes because they are already
    // consistent.
    collectCliquesEvidence(cliques, separators, messages, upsert, amendedFormulas, theseVariables, rootClique.id)
    const messagesReceived: Formula[][] = rootClique.neighbors.map(x => messages[messageName(x, rootClique.id)] || [])
    // After receiving the messages, we need to multiply them together with the root clique's
    // prior distribution.   This may already be a formula, so we use the upsert to avoid
    // creating a duplicate.
    const cliqueFormula = upsert(mult([
      reference(rootClique.prior, amendedFormulas),
      ...reduce((acc: Formula[], xs: Formula[]) => { acc.push(...xs); return acc }, [], messagesReceived),
      ...rootClique.domain.map(id => amendedFormulas[nodes[id].evidenceFunction]),
    ],
    ))
    // The clique formula may need to be marginalized after collecting the evidence to remove
    // variables that do not occur in the join.   Again, we use the upsert to avoid creating
    // a duplicated formula.
    const ccFormula = upsert(marginalize(theseVariables, cliqueFormula, amendedFormulas))
    return ccFormula
  })

  // The joint distributions from each clique must be multiplied together to construct the
  // formula for the overall joint distribution.   Since it is possible that this formula
  // already exists, we use upsert to avoid creating a duplicate.
  const jointFormula = upsert(mult(cliqueFormulas))
  return {
    joinFormulaId: jointFormula.id,
    supplementalFormulas: amendedFormulas.slice(formulas.length),
  }
}
