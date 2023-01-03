
import { ObjectiveFunction } from '../../src/learning/objective-functions'
import { cosineObjectiveFn, rootsObjectiveFn, cubicObjectiveFn } from './objectives'
import { minimize } from '../../src/learning/optimize'

const runTest = (x0: number, converged: boolean, xfinal: number, yfinal: number, objectiveFn: ObjectiveFunction) => {
  const tolerance = 2E-3
  const maxIterations = 100
  const xs0 = [[x0]]
  const initialTower = objectiveFn(xs0)
  const result = minimize(objectiveFn, initialTower, maxIterations)
  expect(result.converged).toEqual(true)
  // verify that the required precision has been satisfied
  expect(result.value).toBeGreaterThan(yfinal - 0.001)
  expect(result.value).toBeLessThan(yfinal + 0.001)
  expect(result.parameters[0][0]).toBeGreaterThan(xfinal - tolerance)
  expect(result.parameters[0][0]).toBeLessThan(xfinal + tolerance)
}

describe('argmax', () => {
  describe('for Quadratic', () => {
    it('finds the optimum from the right', () => runTest(1, true, 0, -1, rootsObjectiveFn))
    it('finds the optimum from the left', () => runTest(-1, true, 0, -1, rootsObjectiveFn))
    it('finds the optimum from the far right', () => runTest(10, true, 0, -1, rootsObjectiveFn))
    it('finds the optimum from the far left', () => runTest(-10, true, 0, -1, rootsObjectiveFn))
    it('finds the optimum when near optimimum', () => runTest(0.001, true, 0, -1, rootsObjectiveFn))
  })
  describe('for Cubic', () => {
    const expectedY = -2 * Math.sqrt(3) / 9
    const expectedX = 1 / Math.sqrt(3)
    it('finds the optimum from the right', () => runTest(1, true, expectedX, expectedY, cubicObjectiveFn))
    it('finds the optimum from the left', () => runTest(-expectedX + 0.01, true, expectedX, expectedY, cubicObjectiveFn))
    it('finds the optimum when near optimimum', () => runTest(expectedX + 0.001, true, expectedX, expectedY, cubicObjectiveFn))
    it('at inflection point', () => runTest(0, true, expectedX, expectedY, cubicObjectiveFn))
  })
  describe('for Sinusoid', () => {
    const expectedY = -1
    const expectedX = 0
    it('finds the optimum from the right', () => runTest(0.25, true, expectedX, expectedY, cosineObjectiveFn))
    it('finds the optimum from the left', () => runTest(-0.25, true, expectedX, expectedY, cosineObjectiveFn))
    it('finds the optimum from the far right', () => runTest(0.49, true, expectedX, expectedY, cosineObjectiveFn))
    it('finds the optimum from the far left', () => runTest(-0.49, true, expectedX, expectedY, cosineObjectiveFn))
    it('finds the optimum when near optimimum', () => runTest(0.001, true, expectedX, expectedY, cosineObjectiveFn))
    it('at inflection point', () => runTest(0.125, true, expectedX, expectedY, cosineObjectiveFn))
  })
})
