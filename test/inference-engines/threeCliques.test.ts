import { network } from '../../models/three-cliques'
import { runTests } from './helpers'

// These gold standard values were computed indepdendently in R.
const GOLD_STANDARD: [string[], number][] = [
  [['T', 'T', 'T', 'T', 'T'], 0.00625],
  [['F', 'T', 'T', 'T', 'T'], 0.00625],
  [['T', 'F', 'T', 'T', 'T'], 0.0125],
  [['F', 'F', 'T', 'T', 'T'], 0.0125],
  [['T', 'T', 'F', 'T', 'T'], 0.05625],
  [['F', 'T', 'F', 'T', 'T'], 0.05625],
  [['T', 'F', 'F', 'T', 'T'], 0.05],
  [['F', 'F', 'F', 'T', 'T'], 0.05],
  [['T', 'T', 'T', 'F', 'T'], 0.00625],
  [['F', 'T', 'T', 'F', 'T'], 0.00625],
  [['T', 'F', 'T', 'F', 'T'], 0.0125],
  [['F', 'F', 'T', 'F', 'T'], 0.0125],
  [['T', 'T', 'F', 'F', 'T'], 0.05625],
  [['F', 'T', 'F', 'F', 'T'], 0.05625],
  [['T', 'F', 'F', 'F', 'T'], 0.05],
  [['F', 'F', 'F', 'F', 'T'], 0.05],
  [['T', 'T', 'T', 'T', 'F'], 0.00625],
  [['F', 'T', 'T', 'T', 'F'], 0.00625],
  [['T', 'F', 'T', 'T', 'F'], 0.0125],
  [['F', 'F', 'T', 'T', 'F'], 0.0125],
  [['T', 'T', 'F', 'T', 'F'], 0.05625],
  [['F', 'T', 'F', 'T', 'F'], 0.05625],
  [['T', 'F', 'F', 'T', 'F'], 0.05],
  [['F', 'F', 'F', 'T', 'F'], 0.05],
  [['T', 'T', 'T', 'F', 'F'], 0.00625],
  [['F', 'T', 'T', 'F', 'F'], 0.00625],
  [['T', 'F', 'T', 'F', 'F'], 0.0125],
  [['F', 'F', 'T', 'F', 'F'], 0.0125],
  [['T', 'T', 'F', 'F', 'F'], 0.05625],
  [['F', 'T', 'F', 'F', 'F'], 0.05625],
  [['T', 'F', 'F', 'F', 'F'], 0.05],
  [['F', 'F', 'F', 'F', 'F'], 0.05],
]

const names: string[] = ['A', 'B', 'C', 'D', 'E']
const testValues: string[][][] = [
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F'], ['T', 'F']],
]

describe('inference on three clique network', () => runTests(network, names, testValues, GOLD_STANDARD))
