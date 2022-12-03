
import { network } from '../../models/alarm'
import { InferenceEngine } from '../../src'
import { restoreEngine } from '../../src/engines/util'

describe('restoreEngine', () => {
  it('clears the cached potentials', () => {
    const engine = new InferenceEngine(network)
    // compute all the posterior joint distributions for the cliques of the junction tree
    engine.inferAll()
    const numberOfVariables = engine.getVariables().length
    const priors = engine.toJSON()._potentials.slice(0, numberOfVariables)
    restoreEngine(engine, priors, {})
    const cache = engine.toJSON()._potentials.slice(numberOfVariables)
    expect(cache.every(x => x == null)).toEqual(true)
  })
  it('sets the local distributions', () => {
    const engine = new InferenceEngine(network)
    const numberOfVariables = engine.getVariables().length
    const priors = engine.toJSON()._potentials.slice(0, numberOfVariables) as number[][]
    const randomPriors = priors.map(ps => ps.map(() => Math.random()))
    restoreEngine(engine, randomPriors, {})
    const updatedPriors = engine.toJSON()._potentials.slice(0, numberOfVariables) as number[][]
    expect(updatedPriors).toEqual(randomPriors)
  })
  it('sets evidence', () => {
    const engine = new InferenceEngine(network)
    engine.setEvidence({ ALARM: ['T'] })
    const numberOfVariables = engine.getVariables().length
    const priors = engine.toJSON()._potentials.slice(0, numberOfVariables) as number[][]
    restoreEngine(engine, priors, { EARTHQUAKE: ['T'] })
    // retracts alarm evidence
    expect(engine.hasEvidenceFor('ALARM')).toEqual(false)
    expect(engine.getEvidence('EARTHQUAKE')).toEqual(['T'])
    // resets earthquake evidence
    restoreEngine(engine, priors, { EARTHQUAKE: ['F'], ALARM: ['T'] })
    expect(engine.getEvidence('EARTHQUAKE')).toEqual(['F'])
    expect(engine.getEvidence('ALARM')).toEqual(['T'])
    restoreEngine(engine, priors, {})
    expect(engine.getAllEvidence()).toEqual({})
  })
})
