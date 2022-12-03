
import { Distribution, FastPotential, ICptWithoutParents, ICptWithParents, InferenceEngine } from '../../src'
import { learnParameters } from '../../src/learning/learning'
import { PairedObservation } from '../../src/learning/Observation'

export function runTest (network: { [x: string]: { levels: string[]; parents: string[]; potentialFunction?: FastPotential | undefined; distribution?: Distribution | undefined; cpt?: ICptWithParents | ICptWithoutParents | undefined } | { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents | undefined } }, sampleSize = 100000, tol = 0.001, learningRate = 1, maxIterations = 100) {
  const dataset: PairedObservation[] = new InferenceEngine(network).getRandomSample(sampleSize)
  describe('learnParameters', () => {
    it('converges whis random dataset', () => {
      const engine = new InferenceEngine(network)
      const result = learnParameters(engine, dataset, learningRate, maxIterations, tol)
      expect(result.steps).toBeGreaterThan(0)
      expect(result.converged).toEqual(true)
    })
    it('converges with a singleton dataset', () => {
      const engine = new InferenceEngine(network)
      const result = learnParameters(engine, [dataset[0]], learningRate, maxIterations, tol)
      expect(result.steps).toBeGreaterThan(0)
      expect(result.converged).toEqual(true)
    })
    it('converges with singleton dataset', () => {
      const engine = new InferenceEngine(network)
      const result = learnParameters(engine, Array(sampleSize).fill(dataset[0]), learningRate, maxIterations, tol)
      expect(result.steps).toBeGreaterThan(1)
      expect(result.converged).toEqual(true)
    })
    it('throws with empty dataset', () => {
      const engine = new InferenceEngine(network)
      expect(() => learnParameters(engine, [], learningRate, maxIterations, tol)).toThrowError()
    })
    it('throws with vacuous observations', () => {
      const engine = new InferenceEngine(network)
      expect(() => learnParameters(engine, [{}], learningRate, maxIterations, tol)).toThrowError()
    })
  })
}
