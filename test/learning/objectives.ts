import { FastPotential } from '../../src'
import { ObjectiveFunction } from '../../src/learning/objective-functions'
import { directionalDerivative, approximateHessian, descentDirection, norm2 } from '../../src/learning/vector-utils'

function mkObjective (f: (x: FastPotential[]) => number, gradF: (x: FastPotential[], y: number) => FastPotential[], hessianF: (x: FastPotential[], y: number, grad: FastPotential[]) => FastPotential[]): ObjectiveFunction {
  return (xs: FastPotential[]) => {
    const value = f(xs)
    // The gradient computed here is the "unconstrained" gradient.   It
    // does not ensure that the result of taking a step will fall on the
    // constraint surface ( e.g. blocks of every CPT add to unity ).
    const gradient = gradF(xs, value)

    // Because the Hessian is a diagonal matrix, we only store the
    // on-diagonal elements.   This reduces storage and also increases
    // computational efficiency.
    const hessian = hessianF(xs, value, gradient)
    // The hessian may be ill conditioned or non-negative definite.   When this
    // is the case, we compute a small value to subtract from each of the elements of
    // the hessian to make it safely negative definite.
    const { hessian: H, isApproximated } = approximateHessian(hessian)
    const direction = descentDirection(gradient, H)
    const magnitude = norm2(direction)
    const result = {
      xs,
      value,
      gradient,
      hessian: H,
      hessianIsApproximate: isApproximated,
      descentDirection: direction.map(ps => ps.map(p => p / magnitude)),
      descentDirectionMagnitude: magnitude,
      directionalDerivative: directionalDerivative(direction, gradient),
    }

    return result
  }
}

export const rootsObjectiveFn = mkObjective(
  (xs: FastPotential[]) => (xs[0][0] - 1) * (xs[0][0] + 1),
  (xs: FastPotential[]) => [[2 * xs[0][0]]],
  () => [[2]],
)

export const cubicObjectiveFn = mkObjective(
  (xs: FastPotential[]) => (xs[0][0] - 1) * (xs[0][0] + 1) * xs[0][0],
  (xs: FastPotential[]) => [[3 * xs[0][0] * xs[0][0] - 1]],
  (xs: FastPotential[]) => [[6 * xs[0][0]]],
)

export const cosineObjectiveFn = mkObjective(
  (xs: FastPotential[]) => -Math.cos(xs[0][0] * 2 * Math.PI),
  (xs: FastPotential[]) => [[Math.sin(xs[0][0] * 2 * Math.PI)]],
  (xs: FastPotential[]) => [[Math.cos(xs[0][0] * 2 * Math.PI)]],
)
