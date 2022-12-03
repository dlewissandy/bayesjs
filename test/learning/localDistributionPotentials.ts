
import { network } from '../../models/alarm'
import { InferenceEngine, FastPotential } from '../../src'
import { restoreEngine } from '../../src/engines/util'
import { localDistributionPotentials } from '../../src/learning/objective-functions/util'

function roundTo (x: number, precision: number) {
  return Number.parseFloat(x.toPrecision(precision))
}

function sumOfBlocks (potentials: FastPotential[], numbersOfHeadLevels: number[]): number[] {
  const result: number[] = []
  for (let i = 0; i < potentials.length; i++) {
    for (let jk = 0; jk < potentials[i].length; jk += numbersOfHeadLevels[i]) {
      const block = potentials[i].slice(jk, jk + numbersOfHeadLevels[i])
      result.push(roundTo(block.reduce((total, x) => x + total, 0), 4))
    }
  }
  return result
}

describe('localDistributionPotentials', () => {
  const engine = new InferenceEngine(network)
  describe('without conditioning to non-zero', () => {
    const locals = engine.getVariables().map(name => localDistributionPotentials(name, engine))
    const numbersOfHeadLevels = engine.getVariables().map(name => engine.getLevels(name).length)
    // this code removes any clique potentials that were computed by localDistributionPotentials, but
    // not by the inferAll().
    engine.setEvidence({ ALARM: ['F'] })
    engine.removeAllEvidence()
    it('has blocks that sum to unity', () => {
      const observed = sumOfBlocks(locals, numbersOfHeadLevels)
      const expected = observed.map(() => 1)
      expect(expected).toEqual(observed)
    })
    it('Respect the probabilistic inferences of the original potentials', () => {
      engine.inferAll()
      const numberOfVariables = engine.getVariables().length
      const expected = engine.toJSON()._potentials.slice(numberOfVariables).map(ps => ps?.map(p => roundTo(p, 6)))
      restoreEngine(engine, locals, {})
      engine.inferAll()
      engine.getVariables().map(name => localDistributionPotentials(name, engine))
      // this code removes any clique potentials that were computed by localDistributionPotentials, but
      // not by the inferAll().
      engine.setEvidence({ ALARM: ['F'] })
      engine.removeAllEvidence()
      engine.inferAll()
      const observed = engine.toJSON()._potentials.slice(numberOfVariables).map(ps => ps?.map(p => roundTo(p, 6)))
      expect(expected).toEqual(observed)
    })
  })
  describe('with conditioning to non-zero', () => {
    const locals = engine.getVariables().map(name => localDistributionPotentials(name, engine))
    const numbersOfHeadLevels = engine.getVariables().map(name => engine.getLevels(name).length)
    // this code removes any clique potentials that were computed by localDistributionPotentials, but
    // not by the inferAll().
    engine.setEvidence({ ALARM: ['F'] })
    engine.removeAllEvidence()
    it('has blocks that sum to unity', () => {
      const observed = sumOfBlocks(locals, numbersOfHeadLevels)
      const expected = observed.map(() => 1)
      expect(expected).toEqual(observed)
    })
    it('Respect the probabilistic inferences of the original potentials', () => {
      engine.inferAll()
      const numberOfVariables = engine.getVariables().length
      const expected = engine.toJSON()._potentials.slice(numberOfVariables).map(ps => ps?.map(p => roundTo(p, 6)))
      restoreEngine(engine, locals, {})
      engine.inferAll()
      engine.getVariables().map(name => localDistributionPotentials(name, engine))
      // this code removes any clique potentials that were computed by localDistributionPotentials, but
      // not by the inferAll().
      engine.setEvidence({ ALARM: ['F'] })
      engine.removeAllEvidence()
      engine.inferAll()
      const observed = engine.toJSON()._potentials.slice(numberOfVariables).map(ps => ps?.map(p => roundTo(p, 6)))
      expect(expected).toEqual(observed)
    })
  })
})
