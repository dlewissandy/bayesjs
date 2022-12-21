import { runTest } from './runTests'
import { network as coinNet } from '../../models/coinflip'
import { network as coinsNet } from '../../models/independentVariables'
import { network as coinWinNet } from '../../models/dependentVariable'
import { InferenceEngine, learnParameters } from '../../src'

let dataset = Array(5000).fill({ COIN: 'HEADS' })
// Train coin flip network with large dataset of observations of a single outcome
describe('coinflip network', () => {
  runTest(coinNet, 100, 1e-6, 1, 100, dataset)
},
)

// Train coin and dice network with large dataset of observations of a single outcome for die
// and no observations for coin
describe('coin and dice network', () => {
  runTest(coinsNet, 100, 1E-4, 1, 100, dataset)
  it('learns the correct values', () => {
    const coinEngine = new InferenceEngine(coinsNet)
    const coinsEngine = new InferenceEngine(coinsNet)
    const result1 = learnParameters(coinEngine, dataset, 1, 100, 1e-6)
    const result2 = learnParameters(coinsEngine, dataset, 1, 100, 1e-6)
    expect(result2.parameters[0]).toEqual(result1.parameters[0])
    expect(result2.parameters[1]).toEqual([0.5, 0.5])
  })
},
)

dataset = Array(5000).fill({ WIN: 'TRUE' })
describe('dependent coin toss', () => {
  runTest(coinWinNet, 100, 1e-6, 1, 100, dataset)
},
)
