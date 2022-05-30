import { sum, product, partition } from 'ramda'
import { InferenceEngine, FastPotential, Distribution, ICptWithParents, ICptWithoutParents } from '../../src'

type NetworkInput = { [name: string]: { levels: string[]; parents: string[]; potentialFunction?: FastPotential | undefined; distribution?: Distribution | undefined; cpt?: ICptWithParents | ICptWithoutParents | undefined } }

/**
 * Round a percentage to the desired precision.
 * @param x The number to be rounded
 */
export const roundResult = (x: number) => {
  if (x === 0) return x
  const scale = Math.pow(10, 8 + Math.floor(-Math.log10(x)))
  return Math.round(x * scale) / scale
}

/**
 * The test case type represents a test case case
 * to be run.  The event and evidence fields provide the
 * arguments to the probabilistic inference to compute
 * and the expected field contains the expected value.
 */
export type TestCase = {
  event: Record<string, string[]>;
  evidence: Record<string, string[]>;
  expected: number;
}

/**
 * This algorithm provides a substantially different means of computing
 * a probabilistic inference by filterning and summing the rows in a
 * gold standard dataset.
 * @param names A list of the names of the varialbes in the dataset
 * @param GOLD_STANDARD An independently verified dataset containing
 *   the joint probability of every possible combination of variable's
 *   values.
 * @returns a function that when given the event will give you the
 *   probability by summing over the rows in the dataset.
 */
export const aggregateGoldStandardResults = (names: string[], GOLD_STANDARD: [string[], number][]) => (event: Record<string, string[]>): number => {
  // filter the dataset to find all the rows that are not in conflict with
  // the evidence.
  const rows = GOLD_STANDARD.filter(kv => names.every((name, i) => {
    const vs = event[name]
    return vs == null || vs.includes(kv[0][i])
  }))
  // sum over the rows to compute the probability.
  return sum(rows.map(kv => kv[1]))
}

/**
 * This helper function constructs all the distinct possible events that can
 * be constructed by choosing one of the provided test values for each
 * variable.   The probability of each event is computed using the gold
 * standard dataset.
 * @param names the names of the variables in the dataset
 * @param testValues A list of possible values for each of the variables
 * @param goldStandard An independently verified dataset containing
 *   the joint probability of every possible combination of variable's
 *   values.
 * @returns a collection of test cases that can be passed to the test
 *   runner.
 */
function makeTestCases (names: string[], testValues: string[][][], goldStandard: [string[], number][]): TestCase[] {
  const testCases: { event: Record<string, string[]>; evidence: Record<string, string[]>; expected: number}[] = []
  const numberOfTestCases = product(testValues.map(xs => xs.length * 3))
  const aggregate = aggregateGoldStandardResults(names, goldStandard)

  for (let i = 0; i < numberOfTestCases; i++) {
    let varIdx = 0
    let x = i
    const event: Record<string, string[]> = {}
    const evidenceAndEvent: Record<string, string[]> = {}
    const evidence: Record<string, string[]> = {}
    // for each variable pick a value and a location in either the
    // evidence or event.
    while (varIdx < testValues.length) {
      const location = x % 3
      x = Math.floor(x / 3)
      const value = x % testValues[varIdx].length
      x = Math.floor(x / testValues[varIdx].length)
      // The variable occurs in the event
      if (location === 1) event[names[varIdx]] = testValues[varIdx][value]
      // The variable occurs in the evidence
      if (location === 2) {
        evidence[names[varIdx]] = testValues[varIdx][value]
      }
      if (location !== 0) {
        evidenceAndEvent[names[varIdx]] = testValues[varIdx][value]
      }
      varIdx++
    }
    // If the event is non-trivial, then compute the probability
    if (Object.keys(event).length > 0) {
      if (Object.keys(evidence).length === 0) {
        // The evidence is empty, so the probability is just the joint probability
        const eventProbability = aggregate(event)
        testCases.push({ event, evidence, expected: eventProbability })
      } else {
        // The event is not empty, so the probability is the joint probability
        // over the union of the evidence and event divided by the probability
        // of the event.
        const evidenceAndEventProbability = aggregate(evidenceAndEvent)
        const evidenceProbability = aggregate(evidence)
        const expected = evidenceAndEventProbability === 0 ? 0 : evidenceAndEventProbability / evidenceProbability
        testCases.push({ event, evidence, expected })
      }
    }
  }
  return testCases
}

/**
 * Run a single test case.
 * @param engine The inference engine to use for inferring the probability
 * @param testCase The test case containing the event, evidence and expected value
 */
function runTestCase (engine: InferenceEngine, testCase: TestCase) {
  if (Object.keys(testCase.evidence).length > 0) engine.setEvidence(testCase.evidence)
  const observed = engine.infer(testCase.event)
  const { expected } = testCase
  const difference = Math.abs(expected - observed)
  const relativeDifference = difference === 0 ? 0 : 2 * difference / (expected + observed)
  const tol = 1e-3
  if (difference !== 0 && relativeDifference > tol) {
    // if the test is going to fail, then output some debugging information.
    console.log(`EVENT: ${JSON.stringify(testCase.event, null, 2)}\nEVIDENCE:${JSON.stringify(testCase.evidence, null, 2)}\nEXPECTED: ${expected}\nOBSERVED: ${observed}`)
  }
  // Fail when the difference between the observed and expected is off by more than 0.1%
  expect(relativeDifference).toBeLessThan(tol)
}

/**
 * Run all the test cases in a specific category
 * @param engine The inference engine to use for inferring the probability
 * @param testCase The test cases containing the events, evidence and expected values
 */
function runTestCases (network: NetworkInput, testCases: TestCase[]) {
  const engine = new InferenceEngine(network)
  for (const testCase of testCases) {
    runTestCase(engine, testCase)
  }
}

/**
 * Run all the tests for all categories of probabilistic queries for the given
 * dataset.
 * @param network The Bayes network to use for making the probabilisic inferences.
 * @param testCases The test cases containing the events, evidence and expected values
 */
export function runTest (network: NetworkInput, testCases: TestCase[]) {
  const hasEvidence = (testCase: TestCase) => Object.values(testCase.evidence).length > 0
  const hasSoftEvidence = (testCase: TestCase) => Object.values(testCase.evidence).some(vs => vs.length > 1)
  const [testCasesWithEvidence, testCasesWithoutEvidence] = partition(hasEvidence, testCases)
  const [testCasesWithSoftEvidence, testCasesWithHardEvidence] = partition(hasSoftEvidence, testCasesWithEvidence)
  it('without evidence', () => { runTestCases(network, testCasesWithoutEvidence) })
  it('with hard evidence', () => { runTestCases(network, testCasesWithHardEvidence) })
  if (testCasesWithSoftEvidence.length > 0) it('with soft evidence', () => { runTestCases(network, testCasesWithSoftEvidence) })
}

/**
 * Construct all the test cases for a given dataset and then run all the tests.
 * This will ensure that the inferred values agree with those predicted from
 * the GOLD_STANDARD dataset.
 * @param network The Bayes network to use for the inferences
 * @param names The names of the variables
 * @param testValues A list of possible values for each of the variables
 * @param GOLD_STANDARD An independently verified dataset containing
 *   the joint probability of every possible combination of variable's
 *   values.
 */
export function runTests (network: NetworkInput, names: string[], testValues: string[][][], GOLD_STANDARD: [string[], number][]) {
  const testCases = makeTestCases(names, testValues, GOLD_STANDARD)
  const isMarginal = (testCase: TestCase) => Object.keys(testCase.event).length === 1
  const isPointEvent = (testCase: TestCase) => Object.values(testCase.event).every(xs => xs.length === 1)

  const [marginalTestCases, joinTestCases] = partition(isMarginal, testCases)
  const [pointMarginalTestCases, cumulativeMarginalTestCases] = partition(isPointEvent, marginalTestCases)
  const [pointJoinTestCases, cumulativeJoinTestCases] = partition(isPointEvent, joinTestCases)

  describe('infers correct marginal probability', () => runTest(network, pointMarginalTestCases))
  if (cumulativeMarginalTestCases.length > 0) describe('infers correct cumulative marginal probability', () => runTest(network, cumulativeMarginalTestCases))
  describe('infers correct joint probability', () => runTest(network, pointJoinTestCases))
  if (cumulativeJoinTestCases.length > 0) describe('infers correct cumulative joint probability', () => runTest(network, cumulativeJoinTestCases))
}
