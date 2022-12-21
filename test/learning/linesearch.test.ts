import { lineSearch } from '../../src/learning/optimize'
import { ObjectiveFunction } from '../../src/learning/objective-functions'
import { directionalDerivative } from '../../src/learning/vector-utils'
import { cosineObjectiveFn, rootsObjectiveFn, cubicObjectiveFn } from './objectives'

/** This test verifies that the the line search method finds a step size that
 * satisfies the Armijo Goldstien criteria.
 */
const runTest = (x0: number, objectiveFn: ObjectiveFunction) => {
  const tower0 = objectiveFn([[x0]])
  const { tower } = lineSearch(objectiveFn, tower0.descentDirection, tower0)
  expect(tower0.value).toBeGreaterThanOrEqual(tower.value)
  const stat = directionalDerivative(tower0.descentDirection, tower.gradient)
  expect(tower0.directionalDerivative).toBeLessThan(stat * 1e-6)
}

describe('lineSearch', () => {
  describe('for Quadratic', () => {
    it('Goes in the correct direction when starting from right', () => {
      runTest(1, rootsObjectiveFn)
    })
    it('Goes in the correct direction when starting from left', () => {
      runTest(-1, rootsObjectiveFn)
    })
    it('finds the correct step when starting near minimizer', () => {
      runTest(0.001, rootsObjectiveFn)
    })
  })
  describe('for Cubic', () => {
    it('Goes in the correct direction when starting from right', () => {
      runTest(1, cubicObjectiveFn)
    })
    it('Goes in the correct direction when starting from left', () => {
      runTest(0, cubicObjectiveFn)
    })
    it('finds the correct step when starting near minimizer', () => {
      runTest(Math.sqrt(1 / 3) + 0.01, cubicObjectiveFn)
    })
    it('finds the correct step when starting near maximizer', () => {
      runTest(0.01 - Math.sqrt(1 / 3), cubicObjectiveFn)
    })
  })
  describe('for sinusoid', () => {
    it('Goes in the correct direction when starting from the left', () => runTest(0.25, cosineObjectiveFn))
    it('Goes in the correct direction when starting from the right', () => runTest(-0.25, cosineObjectiveFn))
    it('finds the correct step when starting near minimizer', () => runTest(0.001, cosineObjectiveFn))
    it('finds the correct step when starting near maximizer', () => runTest(0.501, cosineObjectiveFn))
  })
})
