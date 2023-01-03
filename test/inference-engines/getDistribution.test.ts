import * as expect from 'expect'

import { network } from '../../models/huge-network'
import { InferenceEngine, ICptWithParents, ICptWithoutParents } from '../../src/index'
import { fromCPT } from '../../src/engines'

describe('getPriorDistribution', () => {
  it('round trip is faithful', () => {
    const engine = new InferenceEngine(network)
    const name = 'node3'
    const observed = engine.getPriorDistribution(name).toJSON()
    const cpt = network[name]?.cpt as ICptWithParents | ICptWithoutParents
    const expected = fromCPT('node3', [], [['T', 'F']], cpt).toJSON()
    expect(observed.numberOfHeadVariables).toEqual(expected.numberOfHeadVariables)
    expect(observed.variableNames).toEqual(expected.variableNames)
    expect(observed.variableLevels).toEqual(expected.variableLevels)
    expect(observed.potentialFunction.map(x => x.toExponential(5))).toEqual(expected.potentialFunction.map(x => x.toExponential(5)))
  })
  it('getDistribution is the prior local distribution', () => {
    const engine = new InferenceEngine(network)
    const name = 'node3'
    engine.setEvidence({ node3: ['T'] })
    const observed = engine.getPriorDistribution(name).toJSON()
    const cpt = network[name]?.cpt as ICptWithParents | ICptWithoutParents
    const expected = fromCPT('node3', [], [['T', 'F']], cpt).toJSON()
    expect(observed.numberOfHeadVariables).toEqual(expected.numberOfHeadVariables)
    expect(observed.variableNames).toEqual(expected.variableNames)
    expect(observed.variableLevels).toEqual(expected.variableLevels)
    expect(observed.potentialFunction.map(x => x.toExponential(5))).toEqual(expected.potentialFunction.map(x => x.toExponential(5)))
  })
})

describe('getPosteriorDistribution', () => {
  it('is same as prior when there is no evidence', () => {
    const engine = new InferenceEngine(network)
    const name = 'node3'
    const expected = engine.getPriorDistribution(name).toJSON()
    const observed = engine.getPosteriorDistribution(name).toJSON()
    expect(observed.numberOfHeadVariables).toEqual(expected.numberOfHeadVariables)
    expect(observed.variableNames).toEqual(expected.variableNames)
    expect(observed.variableLevels).toEqual(expected.variableLevels)
    expect(observed.potentialFunction.map(x => x.toExponential(5))).toEqual(expected.potentialFunction.map(x => x.toExponential(5)))
  })
  it('getDistribution is the posterior local distribution', () => {
    const engine = new InferenceEngine(network)
    const name = 'node3'
    engine.setEvidence({ node3: ['T'] })
    const observed = engine.getPosteriorDistribution(name).toJSON().potentialFunction
    const expected = [1, 0]
    expect(observed.map(x => x.toExponential(5))).toEqual(expected.map(x => x.toExponential(5)))
  })
})
