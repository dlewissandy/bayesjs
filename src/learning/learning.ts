import { InferenceEngine } from '..'
import { PairedObservation, groupDataByObservedValues } from './Observation'
import { localDistributionPosteriorPotentials } from './objective-functions/util'
import { objectiveFunction } from './objective-functions/online-learning-objective-function'
import { minimize, OptimizationResult } from './optimize'
import { restoreEngine } from '../engines/util'

/** Given a collection of paired observations and a Bayesian network with possibly informed
 * prior distributions, update the posterior distributions such that it balances the
 * objectives of fitting the observed data and respecting the prior values.
* @param engine: The inference engine for the Bayesian network.   The network must have at least
 *   one variable, and every variable must have at least one level.  This precondition is checked
 *   by the function, which will throw an error if the condition is not met.
 * @param data: A collection of paired observations of the variables in the distribution.  Any
 *   observation for a variable in the network must be for one of the levels of that variable.
 *   This precondition is checked by the function, which will throw an error if the condition is
 *   not met.   Observations for variables that do not occur in the network are ignored.
 * @param learningRate:  This optional argument controls the initial step
 *   size that the line search method will use.  Conceptually, it represents the weighting that is
 *   applied to the prior and data average values when computing a new estimate for the cpt values.
 *   When this value is 1, then both are weighted equally.
 * @param maxIterations: This optional argument controls the maximum
 *   number of steps that the algorithm is allowed to take before giving up.  Changing
 *   this value may be necessary when the provided data set is very small or very sparse.
 * @param tolerance: This optional argument specifies the target tolerance which
 *   must be satisfied in order for the algorithm to terminate.
 *
 * @returns An object containing the status of the optimization.  The "converged" field of this
 *   object can be checked by the calling function to tell if the algorithm reached the optimal
 *   assignment within the given tolerance and number of iterations.
 *
 * NOTE: Although convergence is guaranteed, the resulting assignments may not be globally
 *   optimal.   It may be necessary to use additional heuristics (e.g. tabu search) to
 *   find globally optimal assignments.
 */
export function learnParameters (engine: InferenceEngine, data: PairedObservation[], learningRate = 0.1, maxIterations = 100, tolerance = 1E-4): OptimizationResult {
  // Perform some sanity checks on the data and parameters before we start
  // mutating the inference engine.
  const throwErr = (reason: string) => { throw new Error(`Cannot update Bayes network. ${reason}`) }
  if (data.length === 0) throwErr('no paired observations were provided')
  if (data.some((x: Record<string, string>) => Object.keys(x).length === 0)) throwErr('Dataset contains vacuous observations')
  if (learningRate <= 0 || learningRate >= 1) throwErr('The learning rate must be between 0 and 1')
  if (maxIterations < 0) throwErr('The maximum iterations must be positive')
  if (tolerance < Number.EPSILON || tolerance > 1) throwErr('The tolerance must be between 0 and 1.')

  const variableNames = engine.getVariables()

  // Cache the initial state of the inference engine.   We cache the initial
  // evidence so that it can be restored at the end of the learning episode.
  // We cache the local distributions so that we can roll them back in the
  // event of a failure to converge.
  const initialEvidence = engine.getAllEvidence()
  const initialParameters = engine.toJSON()._potentials
  engine.removeAllEvidence()
  const initialPriors = variableNames.map(name => localDistributionPosteriorPotentials(name, engine))
  const groupedData = groupDataByObservedValues(data, engine)
  const objectiveFn = objectiveFunction(groupedData, initialPriors, learningRate, engine)
  const current = objectiveFn(initialPriors)

  const result = minimize(objectiveFn, current, maxIterations, 1, tolerance)
  if (result.parameters.some(ps => ps.some(p => p < -1E-4))) {
    result.converged = false
    result.message = 'FAILURE: The distributions contained negative probabilities.  Try decreasing the tolerance or learning rate.'
  }
  if (result.converged) {
    // if the training converged, then set the parameters to those that were trained
    restoreEngine(engine, result.parameters.map(ps => ps.map(p => p < 0 ? 0 : p)), initialEvidence)
  } else {
    // otherwise restore the engine to its original state
    restoreEngine(engine, initialParameters, initialEvidence)
  }
  return result
}
