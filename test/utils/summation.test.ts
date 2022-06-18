import { kahanSum } from '../../src/engines/util'
import { sum } from 'ramda'

describe('kahanSummation', () => {
  it('returns a more precise result', () => {
    const xs = [1, Number.EPSILON, -Number.EPSILON, 1, Number.EPSILON, -Number.EPSILON]
    const observed = kahanSum(xs)
    expect(observed).toEqual(2)
    expect(xs.reduce((a, b) => a + b, 0)).not.toEqual(2)
    expect(sum(xs)).not.toEqual(2)
  })
})
