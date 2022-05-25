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
  it('restores the inference engine to its initial state', () => {
    let net = init()
    net.inferAll()
    net.setEvidence({ F: ['-0.5'], I: ['0'] })
    const expected = net.toJSON()
    net = init()
    net.inferAll()
    net.setEvidence({ F: ['-0.5'], I: ['0'] })
    net.infer({ B: ['T'], S: ['A'] })
    const observed = net.toJSON()
    expect(observed).toEqual(expected)
  })
})
