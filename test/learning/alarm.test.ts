import { runTest } from './runTests'
import { network } from '../../models/alarm'

runTest(network, 100, 0.01, 1, 1)
