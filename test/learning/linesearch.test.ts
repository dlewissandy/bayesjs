import { lineSearch } from '../../src/learning/line-search'
import { ObjectiveFunction } from '../../src/learning/objective-functions'
import { directionalDerivative } from '../../src/learning/vector-utils'
import { cosineObjectiveFn, rootsObjectiveFn, cubicObjectiveFn } from './objectives'

/** This test verifies that the the line search method finds a step size that
 * satisfies the Armijo Goldstien criteria.
 */
const runTest = (x0: number, objectiveFn: ObjectiveFunction, maxstep = 1, tolerance = 1E-4) => {
  const tower0 = objectiveFn([[x0]])
  const { tower } = lineSearch(tower0, maxstep, tolerance, objectiveFn)
  expect(tower0.value).toBeGreaterThanOrEqual(tower.value)
  const stat = directionalDerivative(tower0.descentDirection, tower.gradient)
  expect(tower0.directionalDerivative * 0.9).toBeLessThan(stat)
}

describe('lineSearch', () => {
  const MAXSTEPSIZE = 100
  describe('for Quadratic', () => {
    it('Goes in the correct direction when starting from right', () => {
      runTest(1, rootsObjectiveFn, MAXSTEPSIZE)
    })
    it('Goes in the correct direction when starting from left', () => {
      runTest(-1, rootsObjectiveFn)
    })
    it('finds the correct step when starting near minimizer', () => {
      runTest(0.001, rootsObjectiveFn, MAXSTEPSIZE)
    })
  })
  describe('for Cubic', () => {
    it('Goes in the correct direction when starting from right', () => {
      runTest(1, cubicObjectiveFn, MAXSTEPSIZE)
    })
    it('Goes in the correct direction when starting from left', () => {
      runTest(0, cubicObjectiveFn)
    })
    it('finds the correct step when starting near minimizer', () => {
      runTest(Math.sqrt(1 / 3) + 0.01, cubicObjectiveFn, MAXSTEPSIZE)
    })
    it('finds the correct step when starting near maximizer', () => {
      runTest(0.01 - Math.sqrt(1 / 3), cubicObjectiveFn, MAXSTEPSIZE)
    })
  })
  describe('for sinusoid', () => {
    const MAXSTEPSIZE = 10
    it('Goes in the correct direction when starting from the left', () => runTest(0.25, cosineObjectiveFn, MAXSTEPSIZE))
    it('Goes in the correct direction when starting from the right', () => runTest(-0.25, cosineObjectiveFn, MAXSTEPSIZE))
    it('finds the correct step when starting near minimizer', () => runTest(0.001, cosineObjectiveFn, MAXSTEPSIZE))
    it('finds the correct step when starting near maximizer', () => runTest(0.501, cosineObjectiveFn, MAXSTEPSIZE))
  })
})
