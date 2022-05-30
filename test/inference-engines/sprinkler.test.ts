import { network } from '../../models/rain-sprinkler-grasswet'
import { runTests } from './helpers'

// These gold standard values were computed indepdendently in R.
const GOLD_STANDARD: [string[], number][] = [
  [['T', 'T', 'T'], 1.98000000000000E-03],
  [['F', 'T', 'T'], 2.88000000000000E-01],
  [['T', 'F', 'T'], 1.58400000000000E-01],
  [['F', 'F', 'T'], 0.00000000000000E+00],
  [['T', 'T', 'F'], 2.00000000000000E-05],
  [['F', 'T', 'F'], 3.20000000000000E-02],
  [['T', 'F', 'F'], 3.96000000000000E-02],
  [['F', 'F', 'F'], 4.80000000000000E-01],
]

const names: string[] = ['RAIN', 'SPRINKLER', 'GRASS_WET']
const testValues: string[][][] = [
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
]

describe('inference on rain/sprinkler/grass wet network', () => runTests(network, names, testValues, GOLD_STANDARD))
