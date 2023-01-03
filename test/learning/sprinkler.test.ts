import { runTest } from './runTests'
import { network } from '../../models/rain-sprinkler-grasswet'
import { InferenceEngine } from '../../src'

const samplesize = 1000
const maxsteps = 1000
const learningrate = 0.1
const tol = 1e-5
const dataset = new InferenceEngine(network).getRandomSample(samplesize)
runTest(network, samplesize, tol, learningrate, maxsteps, dataset)
