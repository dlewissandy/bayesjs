import { FastPotential } from '..'
import { kahanSum } from '../engines/util'

export const CUBEROOTEPS = Math.pow(Number.EPSILON, 1 / 3)
export const SQRTEPS = Math.sqrt(Number.EPSILON)

/** Compute the l2 norm (magnitude) of a parameter vector
 * @param potentials: The elements of the ascent direction, arranged
 *    as a 2 dimensional array where the first index is the variable
 *    and the second index encodes the levels of the variable and
 *    its parents.
 */
export function norm2 (potentials: FastPotential[]): number {
  return Math.sqrt(kahanSum(potentials.map(ps => kahanSum(ps.map(p => p * p)))))
}

export function dot (a: FastPotential[], b: FastPotential[]) {
  return kahanSum(a.map((ps, i) => kahanSum(ps.map((p, jk) => p * b[i][jk]))))
}

export function scale (value: FastPotential[], c: number): FastPotential[] {
  return value.map(ps => ps.map(p => p * c))
}

export function weightedSum (v1: FastPotential[], w2: number, v2: FastPotential[]) {
  return v1.map((ps, i) => ps.map((p, jk) => p + w2 * v2[i][jk]))
}

/** Compute the condition number for a Hessian matrix.   Larger
* values for the condition number indicate that the matrix
* can be inverted safely, while smaller numbers indicate
* that matrix inversion may be subject to numerical errors.
* @param Hessian - the Hessian matrix for the objective function.
*   since the Hessian is a diagonal matrix, we store only
*   the on-diagonal elements, using the same indexing scheme
*   as for fast potentials.
* @return the condition number and the maximum and minimum
*   on diagonal elements of the Hessian.
*/
export function conditionNumber (hessian: FastPotential[]): { number: number; max: number; min: number } {
  let mx = -Infinity
  let mn = Infinity
  hessian.forEach(hs => hs.forEach(h => {
    mx = Math.max(mx, h)
    mn = Math.min(mn, h)
  }))
  return {
    number: Math.min(Math.abs(mn), Math.abs(mx)) / Math.max(Math.abs(mx), Math.abs(mn)),
    max: mx,
    min: mn,
  }
}

export function directionalDerivative (direction: FastPotential[], gradient: FastPotential[]): number {
  return kahanSum(gradient.map((gs, i) =>
    kahanSum(gs.map((g, jk) => g * direction[i][jk])),
  ))
}

/** Given a Hessian matrix, determine if it is ill conditioned.   If it is
 * return the quasi-Newton approximation of the Hessian, otherwise return
 * the original Hessian. As a side effect, this functional also returns
 * the condition number and mu, the quasi-Newton parameter.
 * @param hessian - The elements of the diagonal of the Hessian matrix,
 *   indexed by variable, and then level index.   We are justified in this
 *   representation because the Hessian matrix will always be diagonal
 *   for this objective function.
 */
export function approximateHessian (hessian: FastPotential[]): { hessian: FastPotential[]; isApproximated: boolean; mu: number} {
  let { max: maxdiag, min: mindiag } = conditionNumber(hessian)
  const maxPosDiag = Math.max(maxdiag, 0)
  let mu = 0
  if (mindiag <= maxPosDiag * CUBEROOTEPS) {
    mu = 2 * (maxPosDiag - mindiag) * CUBEROOTEPS - mindiag
    maxdiag = maxdiag + mu
  }
  const maxoff = 0
  if (maxoff * (1 + 2 * CUBEROOTEPS) > maxdiag) {
    mu = mu + (maxoff - maxdiag) + 2 * CUBEROOTEPS * maxoff
    maxdiag = maxoff * (1 + 2 * CUBEROOTEPS)
  }
  if (maxdiag < CUBEROOTEPS) {
    // H == [0..]
    mu = 1
    maxdiag = 1
  }
  const H = (mu > 0)
    ? hessian.map(hs => hs.map(h => h + mu))
    : hessian

  return {
    hessian: H,
    isApproximated: mu > 0,
    mu,
  }
}

/** Compute the descent direction for a given tower of derivatives of
 * the objective function.   If the Hessian is safely negative
 * definite, then return the Newton direction, otherwise return
 * the quasi-Newton direction.   As a side effect, return the
 * condition number, and the quasi-Newton parameter, mu.
 * @param tower: The tower of derivatives for the objective function
 * @param numbersOfHeadLevels: The number of levels for each
 *   variable in the Bayes network.
 * */
export function descentDirection (gradient: FastPotential[], hessian: FastPotential[]): FastPotential[] {
  const hessianInv = hessian.map(hs => hs.map(h => 1 / h))

  // Compute the ascent direction.
  const direction = hessianInv.map((hs, i) => hs.map((h, jk) =>
    -h * gradient[i][jk]),
  )

  return direction
}

/** Compute the lagrangian multipliers for a given tower of derivatives of
 * the objective function.   If the Hessian is safely negative
 * definite, then return the Newton direction, otherwise return
 * the quasi-Newton direction.
 * @param gradient: the unconditioned gradient.
 * @param hessian: the unconditioned hessian
 * @param numberOfHeadLevels: the number of levels of each head variable in each distirbution in the Bayes network
 * */
export function LagrangianMultipliers (gradient: FastPotential[], hessian: FastPotential[], numbersOfHeadLevels: number[]): FastPotential[] {
  const gammas: number[][] = []
  const hessInv = hessian.map(ps => ps.map(p => 1 / p))

  // Compute the Lagragian multipliers.  These parameters ensure that
  // the sum of the CPT entries over each block of a CPT sum to unity.
  gradient.forEach((grad, variable) => {
    const gs: number[] = []
    const direction: FastPotential[] = descentDirection(gradient, hessian)
    const n = numbersOfHeadLevels[variable]
    for (let block = 0; block < grad.length; block += n) {
      const hslice = hessInv[variable].slice(block, block + n)
      const dslice = direction[variable].slice(block, block + n)
      gs.push(kahanSum(dslice) / kahanSum(hslice))
    }
    gammas.push(gs)
  })
  return gammas
}
