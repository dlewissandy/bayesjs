import { FastPotential } from '..'
import { ObjectiveFunction } from './objective-functions'
import { TowerOfDerivatives } from './TowerOfDerivatives'
import { weightedSum, dot, norm2 } from './vector-utils'

export type OptimizationResult = {
  steps: number;
  converged: boolean;
  value: number;
  directionalDerivative: number;
  parameters: FastPotential[];
  message: string;
}

/** The constants for the WOlfe conditions.   These values are bsed on the
 * recommendations by Nocedal and Wright for quasi-Newton descent directions.
 */
const C1 = 1E-4
const C2 = 0.9

/**
 * Evaluate the Armijo condition on the value approximation.   This condition
 * will be true if and only if the value of the objective function evaluated
 * at (xs + lambda) is less than or equal to the linear approximation of the
 * value evaluated at a small step away from the current value along the
 * descent direction.
 * @param y The value of the objective function evaluated at x+lambda*pk
 * @param y0 The value of the objective function evaluated at x
 * @param slope0 The directional derivative evaluated at x
 * @param lambda The trial step size
 */
function ArmijoCondition (y: number, y0: number, slope0: number, lambda: number) {
  return y <= (y0 + C1 * slope0 * lambda)
}

/**
 * Evaluate the Strong Wolfe condition on curvature.  This is true if and
 * only if the curvature (directional derivative) evaluated at (xs + lambda * pk)
 * is less than a fraction of the original curvature.
 * @param slope The directional derivative evaluated at xs + lambda * pk
 * @param slope0 The directional derivative evaluated at xs
 */
function StrongWolfeCondition (slope: number, slope0: number) {
  return Math.abs(slope) <= C2 * Math.abs(slope0)
}

/**
 * Attempt to find a step size such that both the Wolfe curvature conditon
 * and Armijo conditions are satisified
 * @param fn - The objective function
 * @param current - the current value of the parameters, xs
 * @param descentDirection - The descent direction at xs
 * @param y0 - the value of the objective function at xs0
 * @param slope0 - the directional derivative at xs0
 * @param lo - the lower bound on the step size
 * @param hi - the upper bond on the step size
 * @param yLo - the value of the objective function evaluated at the lower bound of step size.
 * @returns the found step size and the value of the objective function evaluated at that location.
 */
function adaptStepSize (fn: ObjectiveFunction, current: TowerOfDerivatives, descentDirection: FastPotential[], y0: number, slope0:number, lo: number, hi: number, yLo: number): { tower: TowerOfDerivatives; stepSize: number} {
  const MAXITERATIONS = 16
  for (let i = 0; i < MAXITERATIONS; ++i) {
    const lambda = (lo + hi) / 2
    const nextXs = weightedSum(current.xs, lambda, descentDirection)
    const next = fn(nextXs)
    const y = next.value
    const slope = dot(next.gradient, descentDirection)

    if (!ArmijoCondition(y, y0, slope0, lambda) || (y >= yLo)) {
      // if the value of the objective function evaluated at the next estimate
      // for the parameters, fn(nextXs) is not less than a linear approximation
      // of the value at a very small step in the descent direction away from the
      // current location OR the current value is worse than the value estimated
      // at the smallest evaluated step size, then reduce the search region by
      // setting upper bound equal to the current trial step size.
      hi = lambda
    } else {
      // The Armijo condition is satisfied
      if (StrongWolfeCondition(slope, slope0)) {
        // If the Strong Wolfe condition on the curvature is satisfied, then we have
        // found a good step size.
        return { stepSize: lambda, tower: next }
      }

      if (slope * (hi - lo) >= 0) {
        // If for some reason the upper bound is greater than the lower bound, or if the
        // directional derivative is positive, then swap the bounds.
        hi = lo
      }

      // increase the lower bound, and continue searching on the smaller interval.
      lo = lambda
      yLo = y
    }
  }
  return { stepSize: 0, tower: current }
}

export function lineSearch (f: ObjectiveFunction, descentDirection: FastPotential[], current: TowerOfDerivatives, lambda = 1.0, MAXITERATIONS = 100) {
  const y0 = current.value
  const slope0 = dot(current.gradient, descentDirection)
  let y = y0
  let yOld = y0
  let slope = slope0
  let lambda0 = 0
  let next = current

  for (let iteration = 0; iteration < MAXITERATIONS; ++iteration) {
    next = f(weightedSum(current.xs, lambda, descentDirection))
    y = next.value
    slope = dot(next.gradient, descentDirection)
    if (!ArmijoCondition(y, y0, slope0, lambda) || ((iteration > 0) && (y >= yOld))) {
      return adaptStepSize(f, current, descentDirection, y0, slope0, lambda0, lambda, yOld)
    }
    if (StrongWolfeCondition(slope, slope0)) {
      return { stepSize: lambda, tower: next }
    }

    if (slope >= 0) {
      return adaptStepSize(f, current, descentDirection, y0, slope0, lambda, lambda0, y)
    }

    yOld = y
    lambda0 = lambda
    lambda *= 2
  }

  return { stepSize: lambda, tower: next }
}

/**
 * Minimize the given multidimensional, nonlinear objective function by iterative line searching.  At each iteration
 * compute an approximate descent direction and find a step size that satisfies the strong Wolfe conditions.
 * @param f - The multidimensional nonlinear objective function to be minimized
 * @param current - An initial guess for the arguments for the objective function
 * @param maxIterations - The maximum number of iterations to try
 * @param initialStepSize - The initial step size.  This may be decreased by the line search algorithm to acheive the
 *   desired tolerance, or increased to improve the rate of convergence
 * @param tolerance - the minimum change between to iterations.
 * @returns
 */
export function minimize (f: ObjectiveFunction, current: TowerOfDerivatives, maxIterations = 100, initialStepSize = 1.0, tolerance = 1e-6): OptimizationResult {
  let tower: TowerOfDerivatives = current
  let stepSize: number = initialStepSize
  let i = 0
  let converged = false
  let msg = 'FAILURE: Tolerance could not be reached in the specified number of steps.  Try descreasing the tolerance or increasing the number of iterations'

  for (i = 0; i < maxIterations; ++i) {
    const pk = tower.descentDirection
    ;({ stepSize, tower } = lineSearch(f, pk, tower, stepSize))
    const gradSize = norm2(tower.gradient)
    if (stepSize < 1E-16) {
      msg = 'SUCCESS: Could not find a better approximation than the current value'
      converged = true
      break
    } else if (gradSize < tolerance) {
      msg = 'SUCCESS: The magnitude of the gradient is smaller than the specified tolerance'
      converged = true
      break
    }
  }

  return {
    value: tower.value,
    parameters: tower.xs,
    directionalDerivative:
    tower.directionalDerivative,
    converged,
    steps: i + 1,
    message: msg,
  }
}
