import { TowerOfDerivatives } from './TowerOfDerivatives'
import { StepResult, StepStatus } from './StepResult'
import { ObjectiveFunction } from './objective-functions/ObjectiveFunction'
import { SQRTEPS, CUBEROOTEPS, directionalDerivative } from './vector-utils'
import { FastPotential } from '..'

const ALPHA = CUBEROOTEPS
const BETA = 0.9

type LineSearchCoordinate = {
  stepSize: number;
  tower: TowerOfDerivatives;
}

function relativeLength (p: FastPotential[], current: TowerOfDerivatives): number {
  return current.xs.reduce((accI, xx, i) =>
    Math.max(accI, xx.reduce((accJK, x, jk) => Math.max(accJK,
      Math.abs(p[i][jk]) / Math.max(Math.abs(x), SQRTEPS),
    ), 0),
    ), 0)
}

/** Given the current best maximizer for the objective function, and a trial
 * maximizer for the objective function which does not satisfy the
 * Armijo Goldstien criteria, attempt to find a better step size
 * and trial maximizer by approximating the objective function as
 * cubic function of the step size.
 *
 * @param lambda: The best step size found so far
 * @param initslope: The directional derivative at the current x coordiante
 * @param current: The tower of derivatives at the current location
 * @param trial: The tower of derivatives at the best approximator found so far.
 * @param previous: The previously bet approximator found so far.
 *
 */
function cubicInterpolation (lambda: number, initslope: number, current: TowerOfDerivatives, trial: TowerOfDerivatives, previous: LineSearchCoordinate): number {
  // Destructure the inputs
  const { tower, stepSize: lambdaPrev } = previous
  const { value: fprev } = tower
  const { value: fplus } = trial
  const { value: fc } = current

  const deltaLambda = lambda - lambdaPrev
  const lambdaSqr = lambda * lambda
  const lambdaPrevSqr = lambdaPrev * lambdaPrev
  const y = [fplus - fc - lambda * initslope, fprev - fc - lambdaPrev * initslope]
  const a = (y[0] / lambdaSqr - y[1] / lambdaPrevSqr) / deltaLambda
  const b = (-lambdaPrev * y[0] / lambdaSqr + lambda * y[1] / lambdaPrevSqr) / deltaLambda
  const disc = b * b - 3 * a * initslope
  return (a === 0)
    // the cubic has repeated roots
    ? -initslope / (2 * b)
    // the cubic does not have repeated roots
    : (-b + Math.sqrt(disc)) / (3 * a)
}

/** A test for the Arjio-Goldstein value change condition.   This condition ensures that the
  step size is not too large by requiring that the next apprixmation for the minimizer
  causes the value of the objective function to decrease by at least a small amount
*/
function isAlphaConditionSatisfied (fplus: number, fc: number, initslope: number, lambda: number): boolean {
  return fplus <= fc + ALPHA * lambda * initslope
}

// A test for the Arjio-Goldstein gradient change condition.  This condition ensures that
// the step size is not too small by requiring that the directional derivative measured
// at the next approximation for the minimizer causes the magnitude of the slope to
// decrease by some small amount.
function isBetaConditionSatisfied (newslope: number, initslope: number): boolean {
  // directional derivatives should be non-positive because the hessian is positive definite.
  return newslope >= BETA * initslope
}

function successful (lambda: number, tower: TowerOfDerivatives): StepResult {
  return { stepSize: lambda, tower, status: StepStatus.STEP_TAKEN_TOWARD_MAXIMIZER }
}

function failure (lambda: number, tower: TowerOfDerivatives, status: StepStatus): StepResult {
  return { stepSize: lambda, tower, status: status }
}

/** Given a current approximation of the minimizer for the objective function, attempt to
 * find a step size to take in the direction of the descent vector such that the
 * objective function increases and both the Arjio and Goldstein conditions are satisfied.
 * This algorithm is based on
 *
 * Dennis, J.E. and Schnabel, R.B. (1983) Numerical Methods for Unconstrained Optimization and Nonlinear Equations. Prentice-Hall, Englewoods Cliffs.
 *
 * with the following changes:
 * 1) it uses hermite cubic approximation rather than the cubic or quadratic approximation
 *    in the original algorithm.
 * 2) it contains additional optimizations which are possible because the Hessian matrix
 *    is diagonal.
 *
 * @param engine - the inference engine containing the Bayesian network for which the
 *   parameters are being learned.
 * @param current - the tower of derivatives ar the current set of parameters.
 * @param maxStepSize - the maximum allowable step size
 * @param tolerance - the desired tolerance on the final parameters.
 * @param objectiveFn - the objective function being minimized.
 * @param afterStep - an action to perform after a step is performed (optional)
 * @param afterSuccess - an action to perform after the minimizer is found (optional)
 * @param afterFailure - an action to perform when no minimizer is found (optional)
 */
export function lineSearch (current: TowerOfDerivatives, maxStepSize: number, tolerance: number, objectiveFn: ObjectiveFunction): StepResult {
  //= ============= CONSTANTS ====================================
  const MAXSTEPSIZE = maxStepSize
  const MAXITERATIONS = 100

  //= =============== PRECONDITIONING ==========================
  // Clamp the descent direction's magnitude so that it does not exceed the
  // maximum step size.
  const [p, newtLen] = current.descentDirectionMagnitude <= MAXSTEPSIZE
    ? [current.descentDirection, current.descentDirectionMagnitude]
    : [current.descentDirection.map(xs => xs.map(x => MAXSTEPSIZE * x / current.descentDirectionMagnitude)), MAXSTEPSIZE]

  const MAXLAMBDA = (newtLen > MAXSTEPSIZE) ? 1 : MAXSTEPSIZE / newtLen
  const MINLAMBDA = tolerance / relativeLength(p, current)

  let lambda = 1 // Step 9
  const initslope = directionalDerivative(p, current.gradient) // Step 6
  const fc = current.value

  //= =============== UTILITY FUNCTIONS ==========================

  // Compute the next approximation for the maximizer of the objective function by taking
  // a step along the (quasi-)Newton descent direction.
  const nextEstimate = (lambda: number) => {
    const xPlus = current.xs.map((qs, i) => qs.map((q, jk) => q + lambda * p[i][jk]))
    return objectiveFn(xPlus)
  }

  //= =============== BACKTRACKING SEARCH ==========================
  // We begin the search with a full step in the quasi-newton direction.
  let iteration = 0

  let trial: TowerOfDerivatives = current
  let newslope = 0
  let previous: LineSearchCoordinate | undefined
  do {
    trial = nextEstimate(lambda) // step 10.1, 10.2,
    console.log(trial)
    if (isAlphaConditionSatisfied(trial.value, fc, initslope, lambda)) {
      console.log('Alpha is satisfied')
      // *********************
      // STEP 10.3a
      // *********************
      newslope = directionalDerivative(p, trial.gradient)
      if (isBetaConditionSatisfied(newslope, initslope)) {
        console.log('EXIT A')
        return successful(lambda, trial)
      }
      if (lambda === 1 && newtLen < MAXSTEPSIZE) {
        // If this is the first iteration and the step size can be increased, then
        // attempt to do so.
        do {
          previous = { tower: trial, stepSize: lambda }
          lambda = Math.min(2 * lambda, MAXLAMBDA)
          trial = nextEstimate(lambda)
          if (isAlphaConditionSatisfied(trial.value, fc, initslope, lambda)) { newslope = directionalDerivative(p, trial.gradient) }
        } while (isAlphaConditionSatisfied(trial.value, fc, initslope, lambda) && !isBetaConditionSatisfied(newslope, initslope) && lambda < MAXLAMBDA)
        // Note: This may result in either both conditions being true, just alpha being true (when max step size is reached),
        // or neither condition being satisfied.
      }
      if (previous != null && ((lambda < 1) || ((lambda > 1) && !(isAlphaConditionSatisfied(trial.value, fc, initslope, lambda))))) {
        // if the step size has previously been modified, see if there is a better step size between the current
        // and trial step sizes.  This will have the effect of reducing the step size, which may cause the alpha
        // condition to become satisfied, or the beta condition to become unsatisfied.
        const trialCoord = { tower: trial, stepSize: lambda }
        let [lo, hi]: LineSearchCoordinate[] = lambda > previous.stepSize
          ? [previous, trialCoord] : [trialCoord, previous]

        let diff: number = Math.abs(hi.stepSize - lo.stepSize)
        while (diff >= MINLAMBDA && !isBetaConditionSatisfied(newslope, initslope)) {
          const lambdaIncr = Math.max(
            -(newslope * diff * diff) / (2 * (hi.tower.value - (lo.tower.value + newslope * diff))),
            0.2 * diff)
          lambda = lo.stepSize + lambdaIncr
          trial = nextEstimate(lambda)
          if (!isAlphaConditionSatisfied(trial.value, fc, initslope, lambda)) {
            diff = lambdaIncr
            hi = { tower: trial, stepSize: lambda }
          } else {
            newslope = directionalDerivative(p, trial.gradient)
            if (!isBetaConditionSatisfied(newslope, initslope)) {
              lo = { stepSize: lambda, tower: trial }
              diff = diff - lambdaIncr
            }
          }
        }
        if (isBetaConditionSatisfied(newslope, initslope)) {
          console.log('EXIT B')
          return successful(lambda, nextEstimate(lambda))
        } else {
          console.log('EXIT C')
          return successful(lo.stepSize, lo.tower)
        }
      }
    } else if (lambda < MINLAMBDA) {
      console.log('EXIT D')
      console.log(lambda)
      return failure(0, current, StepStatus.STEPSIZE_TOO_SMALL)
    } else {
      console.log('Alpha not satisfied but could be smaller')
      // STEP 10.3c
      let temp = lambda
      if (lambda === 1) {
        // STEP 10.3c.1T
        temp = -initslope / (2 * (trial.value - fc - initslope))
      } else {
        // STEP 10.3c.1E
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        temp = cubicInterpolation(lambda, initslope, current, trial, previous!)
        temp = Math.min(0.5 * lambda, temp)
        console.log(temp)
      }
      previous = { tower: trial, stepSize: lambda }
      lambda = Math.max(0.1 * lambda, temp)
    }
    iteration++
  } while (iteration < MAXITERATIONS)
  console.log('EXIT E')
  return failure(lambda, trial, StepStatus.BACKTRACKING_STEPS_EXCEEDED)
}
