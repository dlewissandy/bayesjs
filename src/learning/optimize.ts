import { FastPotential, LearningResult } from '..'
import { kahanSum } from '../engines/util'
import { ObjectiveFunction } from './objective-functions'
import { TowerOfDerivatives } from './TowerOfDerivatives'

export function dot (a: FastPotential[], b: FastPotential[]) {
  return kahanSum(a.map((ps, i) => kahanSum(ps.map((p, jk) => p * b[i][jk]))))
}

export function norm2 (a: FastPotential[]) {
  return Math.sqrt(dot(a, a))
}

export function scale (value: FastPotential[], c: number): FastPotential[] {
  return value.map(ps => ps.map(p => p * c))
}

export function weightedSum (w1: number, v1: FastPotential[], w2: number, v2: FastPotential[]) {
  return v1.map((ps, i) => ps.map((p, jk) => w1 * p + w2 * v2[i][jk]))
}

export function lineSearch (f: ObjectiveFunction, pk: FastPotential[], current: TowerOfDerivatives, a = 1.0, c1 = 1E-6, c2 = 1e-1) {
  const phi0 = current.value
  const phiPrime0 = dot(current.gradient, pk)
  let phi = phi0
  let phiOld = phi0
  let phiPrime = phiPrime0
  let a0 = 0
  let next = current

  function zoom (aLo: number, aHi: number, phiLo: number): { tower: TowerOfDerivatives; stepSize: number} {
    for (let iteration = 0; iteration < 16; ++iteration) {
      a = (aLo + aHi) / 2
      next = f(weightedSum(1.0, current.xs, a, pk))
      phi = next.value
      phiPrime = dot(next.gradient, pk)

      if ((phi > (phi0 + c1 * a * phiPrime0)) ||
              (phi >= phiLo)) {
        aHi = a
      } else {
        if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
          return { stepSize: a, tower: next }
        }

        if (phiPrime * (aHi - aLo) >= 0) {
          aHi = aLo
        }

        aLo = a
        phiLo = phi
      }
    }

    return { stepSize: 0, tower: current }
  }

  for (let iteration = 0; iteration < 10; ++iteration) {
    next = f(weightedSum(1.0, current.xs, a, pk))
    phi = next.value
    phiPrime = dot(next.gradient, pk)
    if ((phi > (phi0 + c1 * a * phiPrime0)) ||
          (iteration && (phi >= phiOld))) {
      return zoom(a0, a, phiOld)
    }

    if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
      return { stepSize: a, tower: next }
    }

    if (phiPrime >= 0) {
      return zoom(a, a0, phi)
    }

    phiOld = phi
    a0 = a
    a *= 2
  }

  return { stepSize: a, tower: next }
}

export function gradientDescentLineSearch (f: ObjectiveFunction, current: TowerOfDerivatives, maxIterations = 100, initialStepSize = 1.0): LearningResult {
  let tower = current
  let stepSize = initialStepSize
  let i = 0

  for (i = 0; i < maxIterations; ++i) {
    const pk = scale(tower.gradient, -1)
    const result = lineSearch(f, pk, tower, stepSize)
    // eslint-disable-next-line prefer-destructuring
    stepSize = result.stepSize
    // eslint-disable-next-line prefer-destructuring
    tower = result.tower
    const gradSize = norm2(tower.gradient)

    if ((stepSize === 0) || (gradSize < 1e-5)) break
  }

  return { value: tower.value, parameters: tower.xs, directionalDerivative: tower.directionalDerivative, converged: i < maxIterations, steps: i, message: '' }
}
