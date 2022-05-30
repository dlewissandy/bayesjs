import { InferenceEngine } from '../../src'
import { network } from '../../models/fourNode'

describe('infer', () => {
  const init = () => new InferenceEngine(network)
  it('returns 0 when event has variables that don\'t exist', () => {
    const net = init()
    const observed = net.infer({ Q: ['T'] })
    expect(observed).toEqual(0)
  })
  it('returns 0 when event has levels that don\'t exist', () => {
    const net = init()
    const observed = net.infer({ B: ['Q'] })
    expect(observed).toEqual(0)
  })
  it('returns 1 when event has no variables', () => {
    const net = init()
    const observed = net.infer({})
    expect(observed).toEqual(1)
  })
  it('restores the original evidence', () => {
    const net = init()
    const expected = { F: ['-0.5'], I: ['0'] }
    const event = { B: ['-0.5'], S: ['0'] }
    net.setEvidence(expected)
    net.infer(event)
    const observed = net.getAllEvidence()
    expect(observed).toEqual(expected)
  })
  it('does not clear prior potentials', () => {
    const net = init()
    const event = { B: ['-0.5'], S: ['0'] }
    const initialPotentials = net.toJSON()._potentials
    net.infer(event)
    const finalPotentials = net.toJSON()._potentials
    expect(initialPotentials.filter(x => x != null)).toEqual(finalPotentials.filter((_, i) => initialPotentials[i] != null))
  })
  it('inferring twice does not evaluate new potentials', () => {
    const net = init()
    const event = { B: ['-0.5'], S: ['0'] }
    net.infer(event)
    const initialPotentials = net.toJSON()._potentials
    net.infer(event)
    const finalPotentials = net.toJSON()._potentials
    expect(finalPotentials).toEqual(initialPotentials)
  })
})
