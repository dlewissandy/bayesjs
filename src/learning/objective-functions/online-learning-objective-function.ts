import { InferenceEngine, FastPotential } from '../..'
import { parentIndex, kahanSum } from '../../engines/util'
import { GroupedEvidence, sampleBasedAverages } from '../Observation'
import { approximateHessian, norm2, descentDirection, LagrangianMultipliers, directionalDerivative } from '../vector-utils'
import { chiSqrDistance, logLikelihood } from './util'
import { ObjectiveFunction } from './ObjectiveFunction'

/** Compute the exact value of the EM-online objective function and its tower of
 * derivatives for the given trial set of parameters and priors.  Speficially,
 * given the prior set of parameters X0 and the current set of parameters Xc,
 * compute:
 *
 * F(Xc) = learningRate * LogLikelihood(Xc) - ChiSqrDistance(Xc,X0)
 *
 * @param engine: an inference engine containing a bayes network with the current
 *   best set of parameters.
 * @param groups: The paired observations for which the new posterior distribution
 *   is being fit
 * @param priors: The set of prior conditional probability table entries for the variables.
 *   The objective function tries to find a solution that is not too far from these parameters.
 * @param learningRate: Higher values for this parameter cause the solution to favor the
 *   new observations, while lower values favor the priors.
 */
export function objectiveFunction (groups: GroupedEvidence[], priors: FastPotential[], learningRate: number, engine: InferenceEngine): ObjectiveFunction {
  return function (xs: FastPotential[]) {
    // Exctract the information about the structure of the bayes network
    // from the engine.
    const n = engine.getVariables().length
    engine.removeAllEvidence()
    const variableNames = engine.getVariables()
    const currentParams: FastPotential[] = xs.slice(0, n)
    const numbersOfHeadLevels: number[] = variableNames.map(name => engine.getLevels(name).length)

    Object.assign(engine, { _potentials: currentParams })
    // Perform the probabilistic inferences using the engine.
    const ll = logLikelihood(engine, groups)
    const sampleAvgs = sampleBasedAverages(engine, groups)
    Object.assign(engine, { _potentials: priors })
    const distance = kahanSum(engine.getVariables().map((_, i) => chiSqrDistance(priors[i], currentParams[i], sampleAvgs[i].parents, numbersOfHeadLevels[i])))

    // compute the various components of the tower of derivatives.
    const value = -(learningRate * ll - (1 - learningRate) * distance)

    // The gradient computed here is the "unconstrained" gradient.   It
    // does not ensure that the result of taking a step will fall on the
    // constraint surface ( e.g. blocks of every CPT add to unity ).
    const gradient = sampleAvgs.map(({ joint, parents }, i) => joint.map((p, jk) => {
      const result = learningRate * p / currentParams[i][jk] -
        (1 - learningRate) * parents[parentIndex(jk, numbersOfHeadLevels[i])] * (currentParams[i][jk] - priors[i][jk]) / priors[i][jk]
      return -result
    },
    ))

    // Because the Hessian is a diagonal matrix, we only store the
    // on-diagonal elements.   This reduces storage and also increases
    // computational efficiency.
    const hessian = sampleAvgs.map(({ joint, parents }, i) => joint.map((p, jk) => {
      const result =
        -learningRate * p / Math.pow(currentParams[i][jk], 2) -
        (1 - learningRate) * parents[parentIndex(jk, numbersOfHeadLevels[i])] / priors[i][jk]
      return -result
    },
    ))

    // Adjust the Hessian so that it is safely negative definite.  If the
    // Hessian is already safely negative definite, this function will
    // return the original matrix.
    const { hessian: H, isApproximated } = approximateHessian(hessian)

    // Compute the Lagrangian multipliers.    These are used to adjust
    // the gradient to ensure that it falls on the constraint surface.
    const gammas = LagrangianMultipliers(gradient, H, numbersOfHeadLevels)

    // Compute the gradient for constrained optimization of the objective
    // function.   This is achieved by subtracting the Lagrangian multipliers
    // from each element of the unconstrained gradient.
    const adjustedGradient = gradient.map((gs, i) => gs.map((g, jk) => g + gammas[i][parentIndex(jk, numbersOfHeadLevels[i])]))

    // Compute the quasi-newton ascent direction.   This is determined
    // by approximating the objective function as a second order Taylor
    // series approximation in the vicinity of the current value, and
    // solving for the conditions which make the gradient equal to the
    // zero vector.
    const direction = descentDirection(adjustedGradient, H)
    const magnitude = norm2(direction)
    const result = {
      xs: currentParams.slice(0, n),
      value,
      gradient: adjustedGradient,
      hessian: H,
      hessianIsApproximate: isApproximated,
      descentDirection: direction.map(ps => ps.map(p => p / magnitude)),
      descentDirectionMagnitude: 1,
      directionalDerivative: directionalDerivative(direction, gradient),
    }
    return result
  }
}
