import { FastPotential, indexToCombination } from './FastPotential'
import { FastClique } from './FastClique'
import { Formula, FormulaType, Product, Marginal, Reference, NodePotential } from './Formula'
import { sum, intersection, product, union } from 'ramda'
import { evaluateMarginalPure, evaluateProductPure, evaluate } from './evaluation'
import { FastNode } from './FastNode'
import { propagateJoinMessages } from './join-propagation'

/** This is a special case algorithm for computing a joint probability when there
 * is no evidence.   This algorithm relies upon a formula evaluation method that
 * removes all the rows of the join that will not participate in the final
 * summation.   If the network has evidence, then this algorithm will fail to
 * produce the correct results because it cannot normalize the distributions.
 * @param nodes: The collection of nodes in the bayesian network
 * @param cliques: The collection of cliques in the junction tree for the bayesian
 *   network.
 * @param formulas: The collection of formulas for computing the posterior
 *   distributions for the cliques, nodes and separators  of the Bayes Network, and
 *   all the intermediate potentials.
 * @param potentialFunction: The collection of potential functions for the cliques
 *   nodes and separators of the Bayes Network and the intermediate potentials.
 * @param join: The indices of the variables that occur as variables of
 *   the joint distribution being constructed.  These indices must be valid references
 *   to variables in the network.
 * @returns: The probability of the join.
 */
export function inferJoinProbability (nodes: FastNode[], cliques: FastClique[], connectedComponents: number[][], separators: number[][], formulas: Formula[], potentials: (FastPotential | null)[], joinDomain: number[], values: (number[] | null)[]): {
  joinProbability: number;
  supplementalFormulas: Formula[];
  supplementalPotentials: (FastPotential | null)[];
} {
  const initialNumberOfFormulas = formulas.length
  const { joinFormulaId, supplementalFormulas } = propagateJoinMessages(nodes, cliques, connectedComponents, formulas, separators, joinDomain, [])
  // We initialize an array of potential functions with the previously computed potentials, and enough new elements to cover
  // the formulas added during message passing.   Then we evaluate the joint formula using
  // that context.
  const amendedPotentials = [...potentials, ...Array(supplementalFormulas.length).fill(null)]
  const amendedFormulas = [...formulas, ...supplementalFormulas]
  const sizes = amendedFormulas.map(formula => formula.domain.reduce((total, v, i) => total * (values[v]?.length ?? formula.numberOfLevels[i]), 1))
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  const joinPotentials = evaluateFormula(joinFormulaId, nodes, amendedFormulas, amendedPotentials, initialNumberOfFormulas, sizes, values)
  const supplementalPotentials = amendedPotentials.slice(potentials.length)

  // Update any potentials that were previously unresolved
  amendedPotentials.slice(0, potentials.length).forEach((ps, i) => {
    if (potentials[i] == null) potentials[i] = ps
  })
  // finally, return the result of the join and the supplementary information
  return {
    joinProbability: sum(joinPotentials),
    supplementalFormulas,
    supplementalPotentials,
  }
}

/**
 * Remove any rows in a potential function that are associated with levels
 * of a variable that will not be summed in the join.   For example, given
 * P(X,Y) = [P(X=T,Y=T)=0.25,P(X=F,Y=T)=0.5,P(X=T,Y=F)=0.15, P(X=F,Y=F) = 0.1]
 * and the event {X: ['T']}, then the resulting vector will be:
 * [P(X=T,Y=T)=0.25,P(X=T,Y=F)=0.15].   Note that this does not correspond
 * to one of the operations in the underlying Join/Marginal algebra, and is
 * introduced soley for efficiency improvements.
 * @param potential - The potential being compacted
 * @param formula - The formula of the potential being compacted
 * @param values - the values of the variables that were provided in the
 *   event for which we are inferring the probability.   The values are
 *   provided as an array with the same  dimension as the nodes of the bayes
 *   network.  Each element is either null, indicating that no values were
 *   provided, or a list of numbers, representing the indicies of the levels
 *   that were provided for that variable.
 * @returns a new potential function which has been compacted to remove the
 *   elements corresponding to values that will not be summed into the joint
 *   probability.
 */
function compactPotentials (potential: FastPotential, formula: Formula, values: (number[] | null)[]) {
  const { domain } = formula
  const oldNumbersOfLevels = formula.numberOfLevels

  // if there is no variable in the domain of this potential that participated in the
  // event, then return the potential function as is.
  if (domain.every(i => values[i] === null)) return potential

  // otherwise, filter out the rows of the distribution for which
  // the levels of one or more of the variables are not in the event.
  const result = potential.filter((p, i) => {
    const combo = indexToCombination(i, oldNumbersOfLevels)
    return combo.every((l, vIdx) => {
      const v = values[domain[vIdx]]
      return v == null || v.includes(l)
    })
  })
  return result
}

/**
 * Compute the number of levels for a formula after it has been compacted.
 * @param formula - the formula being compacted
 * @param nodes - the nodes of the bayes network
 * @param values the values of the variables that were provided in the
 *   event for which we are inferring the probability.   The values are
 *   provided as an array with the same  dimension as the nodes of the bayes
 *   network.  Each element is either null, indicating that no values were
 *   provided, or a list of numbers, representing the indicies of the levels
 *   that were provided for that variable.
 * @returns the number of levels in the formula after compacting.
 */
function compactedNumberOfLevels (formula: Formula, nodes: FastNode[], values: (number[]|null)[]) {
  return formula.domain.map((nodeId) => {
    const v = values[nodeId]
    return v == null ? nodes[nodeId].levels.length : v.length
  })
}

function computeScaleFactor (factorFormulas: Formula[], nodes: FastNode[]): number {
  const filteredFactors = factorFormulas.filter(f => f.kind !== FormulaType.EVIDENCE_FUNCTION)
  const nodePotentials = filteredFactors.filter(x => x.kind === FormulaType.NODE_POTENTIAL) as NodePotential[]

  const divisors = nodePotentials.map(x => product(x.numberOfLevels.slice(1)))
  // Compute the domain after each pairwise multiplication in the product
  const intermediateDomains: number[][] = filteredFactors.reduce((total: number[][], f) => {
    if (total.length === 0) return [f.domain]
    return [...total, union(total[total.length - 1], f.domain)]
  }, [])

  // Compute the intersection of the domains of each factor with the accumulated
  // product.
  const parentDomains = filteredFactors.slice(1).map((f, i) => intersection(intermediateDomains[i], f.domain))
  const multipliers = parentDomains.map(parentDomain => product(parentDomain.map(varIdx => nodes[varIdx].levels.length)))
  return product(multipliers) / product(divisors)
}

/**
 * Given a product formula, evaluate each of the factor formulas and then
 * evaluate their join.   To ensure efficient computation, compact the
 * result to remove any rows associated with levels that do not explicitly
 * occur in the values provided values.
 * @param productFormula The formula for the product to compute.
 * @param nodes The nodes of the Bayes network
 * @param formulas The formulas for the posterior joint distributions over
 *   the cliques of the Bayes network, the posterior marginals over the
 *   variables and all the intermediate formulas as well as any supplemental
 *   formulas added to compute the joint distribution in question.
 * @param potentials The potentials for the posterior joint distributions over
 *   the cliques of the Bayes network, the posterior marginals over the
 *   variables and all the intermediate formulas as well as any supplemental
 *   formulas added to compute the joint distribution in question.
 * @param initialNumberOfFormulas The inital number of formulas, prior to the
 *   addition of the supplemental formulas for computing the join.
 * @param sizes  - the sizes of all the formulas after compacting.
 * @param values the values of the variables that were provided in the
 *   event for which we are inferring the probability.   The values are
 *   provided as an array with the same  dimension as the nodes of the bayes
 *   network.  Each element is either null, indicating that no values were
 *   provided, or a list of numbers, representing the indicies of the levels
 *   that were provided for that variable.
 * @returns the compacted potentials for the requested product.
 */
const evalProduct = (productFormula: Product, nodes: FastNode[], formulas: Formula[], potentials: (FastPotential | null)[], initialNumberOfFormulas: number, sizes: number[], values: (number[]|null)[]): FastPotential => {
  // first we need to evaluate the factors that are being multiplied together.
  const factorFormulas = productFormula.factorIds.map(factorId => formulas[factorId])
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  const factorPotentials: FastPotential[] = productFormula.factorIds.map(factorId => evaluateFormula(factorId, nodes, formulas, potentials, initialNumberOfFormulas, sizes, values))
  // short cuts for nullary and unary products
  if (factorPotentials.length === 0) {
    const result: FastPotential = []
    potentials[productFormula.id] = result // unit potential
    return result
  }

  const scaleFactor = computeScaleFactor(factorFormulas, nodes)

  // If we arrived here, there are at least two factors.  We start by initializing
  // an array for multiplicatively accumulating the potential values
  const factorNumberOfLevels = factorFormulas.map(formula => compactedNumberOfLevels(formula, nodes, values))
  const factorDomains = factorFormulas.map((x: Formula) => x.domain)
  const productDomain = productFormula.domain
  const productNumberOfLevels = compactedNumberOfLevels(productFormula, nodes, values)

  const result = evaluateProductPure(
    factorPotentials,
    factorDomains,
    factorNumberOfLevels,
    productDomain,
    productNumberOfLevels,
    sizes[productFormula.id],
    false,
  ).map(p => p * scaleFactor)

  potentials[productFormula.id] = result
  return result
}

/**
 * Given a marginal formula, evaluate the inner formulas and then
 * evaluate the marginalization.   To ensure efficient computation, compact the
 * result to remove any rows associated with levels that do not explicitly
 * occur in the values provided values.
 * @param marginalFormula The formula for the marginal to compute.
 * @param nodes The nodes of the Bayes network
 * @param formulas The formulas for the posterior joint distributions over
 *   the cliques of the Bayes network, the posterior marginals over the
 *   variables and all the intermediate formulas as well as any supplemental
 *   formulas added to compute the joint distribution in question.
 * @param potentials The potentials for the posterior joint distributions over
 *   the cliques of the Bayes network, the posterior marginals over the
 *   variables and all the intermediate formulas as well as any supplemental
 *   formulas added to compute the joint distribution in question.
 * @param initialNumberOfFormulas The inital number of formulas, prior to the
 *   addition of the supplemental formulas for computing the join.
 * @param sizes  - the sizes of all the formulas after compacting.
 * @param values the values of the variables that were provided in the
 *   event for which we are inferring the probability.   The values are
 *   provided as an array with the same  dimension as the nodes of the bayes
 *   network.  Each element is either null, indicating that no values were
 *   provided, or a list of numbers, representing the indicies of the levels
 *   that were provided for that variable.
 * @returns the compacted potentials for the requested marginal.
 */
const evalMarginal = (marginalFormula: Marginal, nodes: FastNode[], formulas: Formula[], potentials: (FastPotential | null)[], initialNumberOfFormulas: number, sizes: number[], values: (number[] | null)[]): FastPotential => {
  // First we need to evaluate the formula for the potential that is being marginalized.
  // eslint-disable-next-line @typescript-eslint/no-use-before-define
  const innerPotential: FastPotential = evaluateFormula(marginalFormula.potential, nodes, formulas, potentials, initialNumberOfFormulas, sizes, values)
  const innerFormula = formulas[marginalFormula.potential]
  // Marginalization will remove zero or more variables (nodes) from the distribution.
  // We need to know which nodes are retained after the marginalization.
  const { domain: innerDomain } = innerFormula
  const innerNumberOfLevels = compactedNumberOfLevels(innerFormula, nodes, values)
  const { domain: marginalDomain } = marginalFormula
  const marginalSize = sizes[marginalFormula.id]
  const marginalNumberOfLevels = compactedNumberOfLevels(marginalFormula, nodes, values)

  const result = evaluateMarginalPure(
    innerPotential, innerDomain, innerNumberOfLevels, marginalDomain, marginalNumberOfLevels, marginalSize, false,
  )

  potentials[marginalFormula.id] = result
  return result
}

/**
 * Recursively evaluate a formula for a potential.
 * @param marginalFormula - the symbolic representation of the marginal being evaulated
 * @param nodes - The collection of nodes in the inference engine.  This is used to
 *   locate locate nodes in the domain of the marginal and comprehend their properties
 * @param formulas - A list containing the symbolic representation of the potential functions
 *   of each clique, potential and term that occurs in the Bayes network.   This is used for
 *   evaluating the potential that is being marginalized
 * @param potentials - The collection of potentials corresponding to possibly cached results
 *   of evaluting the given list of formulas.  The list of potentials and list of formulas
 *   share the same ordering scheme such that the first potential corresponds to the cached
 *   result of evaluating the first potential, and so on.
 * @param initialNumberOfFormulas - the number of formulas, not including supplementary
 *   formulas.   This term is used to ensure that formulas used for making the network
 *   consistent are not compacted.
 * @param sizes - the sizes of all the formulas after compacting.
 * @param values - the values of the variables that were provided in the
 *   event for which we are inferring the probability.   The values are
 *   provided as an array with the same  dimension as the nodes of the bayes
 *   network.  Each element is either null, indicating that no values were
 *   provided, or a list of numbers, representing the indicies of the levels
 *   that were provided for that variable.
 * NOTE: If this function computes a new value for the given formula or any of its terms,
 *   it will mutate the cache of potentials to update the cached values.
 */
export const evaluateFormula = (formulaId: number, nodes: FastNode[], formulas: Formula[], potentials: (FastPotential | null)[], initialNumberOfFormulas: number, sizes: number[], values: (number[] | null)[]): FastPotential => {
  const cachedValue = potentials[formulaId]
  if (cachedValue) {
    // A cached value already exists.  Return it.
    if (cachedValue.length === sizes[formulaId]) return cachedValue
    return compactPotentials(cachedValue, formulas[formulaId], values)
  }
  if (formulaId < initialNumberOfFormulas) {
    // If it hasn't previously been computed, but is part of the base set of
    // formulas, then use the evaluate function to computeso that the
    // uncompacted potential function will be stored for future use, but then
    // return the compacted version for use here.
    const potential = evaluate(formulaId, nodes, formulas, potentials)
    return compactPotentials(potential, formulas[formulaId], values)
  } else {
    const formula = formulas[formulaId]
    switch (formula.kind) {
      case FormulaType.PRODUCT: {
        // recursively evaulate the product and all of its factors.
        const productFormula = formula as Product
        return evalProduct(productFormula, nodes, formulas, potentials, initialNumberOfFormulas, sizes, values)
      }
      case FormulaType.MARGINAL: {
        // Recursively evaluate the marginalization of a potential function.
        const marginalFormula = formula as Marginal
        return evalMarginal(marginalFormula, nodes, formulas, potentials, initialNumberOfFormulas, sizes, values)
      }
      case FormulaType.REFERENCE: {
        // If the formula is a reference, evaluate the referenced potential
        const ref = formula as Reference
        const result = evaluateFormula(ref.formulaId, nodes, formulas, potentials, initialNumberOfFormulas, sizes, values)
        return result
      }
      case FormulaType.EVIDENCE_FUNCTION:
      case FormulaType.UNIT:
      case FormulaType.NODE_POTENTIAL: throw new Error('Cannot infer the probability of the given event.  Unexpected formula type.')
    }
  }
}
