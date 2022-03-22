
import { network } from '../../models/alarm'
import { InferenceEngine, FastPotential } from '../../src'
import { groupDataByObservedValues } from '../../src/learning/Observation'
import { indexToCombination } from '../../src/engines'
import { product } from 'ramda'

describe('getRandomSample', () => {
  it('returns an empty sample when requested', () => {
    const engine = new InferenceEngine(network)
    const result: Record<string, string>[] = engine.getRandomSample(0)
    expect(result).toEqual([])
  })
  it('throws when a negative sample size is requested', () => {
    const engine = new InferenceEngine(network)
    expect(() => engine.getRandomSample(-1)).toThrow()
  })
  it('sample contains the correct number of observations', () => {
    const engine = new InferenceEngine(network)
    const n = Math.floor(Math.random() * 1000)
    const result = engine.getRandomSample(n)
    expect(result.length).toEqual(n)
  })
  it('each observation has a level for each variable', () => {
    const engine = new InferenceEngine(network)
    const variables = engine.getVariables()
    const n = 1
    const result = engine.getRandomSample(n)
    const expected = result.map(x => variables.every(name => x[name] != null))
    expect(expected).toEqual(Array(n).fill(true))
  })
  it('is consistent with any provided evidence', () => {
    const engine = new InferenceEngine(network)
    engine.setEvidence({ ALARM: ['T'] })
    const n = 100
    const result = engine.getRandomSample(n)
    const str = 'ALARM'
    expect(result.every(x => x[str] === 'T')).toEqual(true)
  })
  it('approximates the joint distribution', () => {
    const engine = new InferenceEngine(network)
    const n = 10000
    const result = engine.getRandomSample(n)
    const groupedData = groupDataByObservedValues(result, engine)
    const variableNames = engine.getVariables()
    const dist = engine.getJointDistribution(variableNames, [])
    const variableLevels = variableNames.map(name => engine.getLevels(name))
    const numbersOfHeadLevels = variableNames.map(name => engine.getLevels(name).length)
    const observed: FastPotential = Array(product(numbersOfHeadLevels)).fill(null).map((_, idx) => {
      const combo = indexToCombination(idx, numbersOfHeadLevels)
      const group = groupedData.filter(x => combo.every((lvl, varIdx) => x.evidence[variableNames[varIdx]][0] === variableLevels[varIdx][lvl]))
      return (group.length === 0) ? 0 : group[0].frequency
    })
    const expected = dist.getPotentials().map(p => Number.isNaN(p) ? 0 : p)
    const residuals = observed.map((x, i) => Math.abs(x - expected[i]))
    expect(residuals.every(p => p < 0.01)).toEqual(true)
  })
  it('approximates the joint distribution with evidence', () => {
    const engine = new InferenceEngine(network)
    engine.setEvidence({ ALARM: ['T'] })
    const n = 10000
    const result = engine.getRandomSample(n)
    const groupedData = groupDataByObservedValues(result, engine)
    const variableNames = engine.getVariables()
    const dist = engine.getJointDistribution(variableNames, [])
    const variableLevels = variableNames.map(name => engine.getLevels(name))
    const numbersOfHeadLevels = variableNames.map(name => engine.getLevels(name).length)
    const observed: FastPotential = Array(product(numbersOfHeadLevels)).fill(null).map((_, idx) => {
      const combo = indexToCombination(idx, numbersOfHeadLevels)
      const group = groupedData.filter(x => combo.every((lvl, varIdx) => x.evidence[variableNames[varIdx]][0] === variableLevels[varIdx][lvl]))
      return (group.length === 0) ? 0 : group[0].frequency
    })
    const expected = dist.getPotentials().map(p => Number.isNaN(p) ? 0 : p)
    const residuals = observed.map((x, i) => Math.abs(x - expected[i]))
    expect(residuals.every(p => p < 0.01)).toEqual(true)
  })
})
