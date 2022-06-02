import { network } from '../../models/fourNode'
import { InferenceEngine } from '../../src'
import { roundResult } from './helpers'

describe('engine infer and distribution infer', () => {
  const runTest = (event: Record<string, string[]>, evidence?: Record<string, string[]>) => {
    const engine = new InferenceEngine(network)
    if (evidence) engine.setEvidence(evidence)
    const expected = roundResult(engine.getJointDistribution(Object.keys(event), Object.keys(evidence ?? {})).infer(event, evidence))
    const observed = roundResult(engine.infer(event))
    expect(observed).toEqual(expected)
  }

  describe('have the same result for marginal probability', () => {
    it('with no evidence', () => {
      runTest({ B: ['T'] })
      runTest({ I: ['-1'] })
      runTest({ F: ['0'] })
      runTest({ S: ['A'] })
    })
    it('with hard evidence', () => {
      runTest({ B: ['T'] }, { I: ['1'] })
      runTest({ B: ['T'] }, { F: ['0'] })
      runTest({ B: ['T'] }, { S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['1'] })
      runTest({ B: ['T'] }, { F: ['0'], S: ['A'] })
      runTest({ B: ['T'] }, { I: ['0'], S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['0'], S: ['A'] })

      runTest({ F: ['0'] }, { I: ['1'] })
      runTest({ F: ['0'] }, { B: ['T'] })
      runTest({ F: ['0'] }, { S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['1'] })
      runTest({ F: ['0'] }, { B: ['T'], S: ['A'] })
      runTest({ F: ['0'] }, { I: ['0'], S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['0'], S: ['A'] })

      runTest({ I: ['0'] }, { F: ['0'] })
      runTest({ I: ['0'] }, { B: ['T'] })
      runTest({ I: ['0'] }, { S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0'] })
      runTest({ I: ['0'] }, { B: ['T'], S: ['A'] })
      runTest({ I: ['0'] }, { F: ['0'], S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0'], S: ['A'] })

      runTest({ S: ['A'] }, { F: ['0'] })
      runTest({ S: ['A'] }, { B: ['T'] })
      runTest({ S: ['A'] }, { I: ['1'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0'] })
      runTest({ S: ['A'] }, { B: ['T'], I: ['1'] })
      runTest({ S: ['A'] }, { F: ['0'], I: ['1'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0'], I: ['1'] })
    })
    it('with soft evidence', () => {
      runTest({ B: ['T'] }, { I: ['1', '0'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'] }, { S: ['A', 'C'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], I: ['1'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['1', '0'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], I: ['0'], S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['0'], S: ['A', 'B'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'] }, { F: ['0'], I: ['0', '1'], S: ['A', 'C'] })
      runTest({ B: ['T'] }, { F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ F: ['0'] }, { I: ['1', '0'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'] }, { S: ['A', 'C'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], I: ['1'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['1', '0'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T'], S: ['A', 'C'] })
      runTest({ F: ['0'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], I: ['0'], S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['0'], S: ['A', 'B'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], I: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0'] }, { B: ['T'], I: ['0', '1'], S: ['A', 'C'] })
      runTest({ F: ['0'] }, { B: ['T', 'F'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ I: ['0'] }, { F: ['0', '0.5'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'] })
      runTest({ I: ['0'] }, { S: ['A', 'C'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], F: ['0.5'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0', '0.5'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T'], S: ['A', 'C'] })
      runTest({ I: ['0'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], F: ['0'], S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0'], S: ['A', 'B'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], F: ['0'], S: ['A', 'C'] })
      runTest({ I: ['0'] }, { B: ['T'], F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ I: ['0'] }, { B: ['T', 'F'], F: ['0', '0.5'], S: ['A', 'C'] })

      runTest({ S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'] })
      runTest({ S: ['A'] }, { I: ['0', '1'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], F: ['0.5'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0', '0.5'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], I: ['0'] })
      runTest({ S: ['A'] }, { B: ['T'], I: ['0', '1'] })
      runTest({ S: ['A'] }, { F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A'] }, { F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], F: ['0'], I: ['0'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A'] }, { B: ['T'], F: ['0', '0.5'], I: ['0', '1'] })
      runTest({ S: ['A'] }, { B: ['T', 'F'], F: ['0', '0.5'], I: ['0', '1'] })
    })
  })
  describe('have the same result for cumulative marginal probability', () => {
    it('with no evidence', () => {
      runTest({ B: ['T', 'F'] })
      runTest({ I: ['-1', '0'] })
      runTest({ F: ['0', '0.5'] })
      runTest({ S: ['A', 'C'] })
    })
    it('with hard evidence', () => {
      runTest({ B: ['T', 'F'] }, { I: ['1'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['1'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { I: ['0'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['0'], S: ['A'] })

      runTest({ F: ['0', '0.5'] }, { I: ['1'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'] }, { S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['1'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { I: ['0'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['0'], S: ['A'] })

      runTest({ I: ['0', '1'] }, { F: ['0'] })
      runTest({ I: ['0', '1'] }, { B: ['T'] })
      runTest({ I: ['0', '1'] }, { S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { F: ['0'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0'], S: ['A'] })

      runTest({ S: ['A', 'B'] }, { F: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'] })
      runTest({ S: ['A', 'B'] }, { I: ['1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], I: ['1'] })
      runTest({ S: ['A', 'B'] }, { F: ['0'], I: ['1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0'], I: ['1'] })
    })
    it('with soft evidence', () => {
      runTest({ B: ['T', 'F'] }, { I: ['1', '0'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'] }, { S: ['A', 'C'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], I: ['1'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['1', '0'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], I: ['0'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['0'], S: ['A', 'B'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'] }, { F: ['0'], I: ['0', '1'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'] }, { F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ F: ['0', '0.5'] }, { I: ['1', '0'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'] }, { S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], I: ['1'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['1', '0'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], I: ['0'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['0'], S: ['A', 'B'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], I: ['0', '1'], S: ['A'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], I: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T'], I: ['0', '1'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'] }, { B: ['T', 'F'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ I: ['0', '1'] }, { F: ['0', '0.5'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'] })
      runTest({ I: ['0', '1'] }, { S: ['A', 'C'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], F: ['0.5'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0', '0.5'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], S: ['A', 'C'] })
      runTest({ I: ['0', '1'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], F: ['0'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0'], S: ['A', 'B'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], F: ['0'], S: ['A', 'C'] })
      runTest({ I: ['0', '1'] }, { B: ['T'], F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ I: ['0', '1'] }, { B: ['T', 'F'], F: ['0', '0.5'], S: ['A', 'C'] })

      runTest({ S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], F: ['0.5'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0', '0.5'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], I: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A', 'B'] }, { F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], F: ['0'], I: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], F: ['0', '0.5'], I: ['0'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], F: ['0'], I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T'], F: ['0', '0.5'], I: ['0', '1'] })
      runTest({ S: ['A', 'B'] }, { B: ['T', 'F'], F: ['0', '0.5'], I: ['0', '1'] })
    })
  })
  describe('have the same result for joint probability', () => {
    it('with no evidence', () => {
      runTest({ B: ['T'], F: ['0'] })
      runTest({ B: ['T'], I: ['-1'] })
      runTest({ B: ['T'], S: ['A'] })
      runTest({ F: ['0'], I: ['-1'] })
      runTest({ F: ['0'], S: ['A'] })
      runTest({ I: ['-1'], S: ['A'] })

      runTest({ B: ['T'], F: ['0'], I: ['1'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A'] })
      runTest({ B: ['T'], F: ['0'], S: ['A'] })
      runTest({ F: ['0'], I: ['0'], S: ['A'] })

      runTest({ B: ['T'], F: ['0'], I: ['0'], S: ['A'] })
    })
    it('with hard evidence', () => {
      runTest({ B: ['T'], F: ['0'] }, { I: ['0'] })
      runTest({ B: ['T'], F: ['0'] }, { S: ['A'] })
      runTest({ B: ['T'], F: ['0'] }, { I: ['0'], S: ['A'] })

      runTest({ B: ['T'], I: ['-1'] }, { F: ['0'] })
      runTest({ B: ['T'], I: ['-1'] }, { S: ['A'] })
      runTest({ B: ['T'], I: ['-1'] }, { F: ['0'], S: ['A'] })

      runTest({ B: ['T'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T'], S: ['A'] }, { I: ['0'] })
      runTest({ B: ['T'], S: ['A'] }, { F: ['0'], I: ['0'] })

      runTest({ F: ['0'], I: ['-1'] }, { B: ['T'] })
      runTest({ F: ['0'], I: ['-1'] }, { S: ['A'] })
      runTest({ F: ['0'], I: ['-1'] }, { B: ['T'], S: ['A'] })

      runTest({ F: ['0'], S: ['A'] }, { I: ['0'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T'], I: ['0'] })

      runTest({ I: ['-1'], S: ['A'] }, { F: ['0'] })
      runTest({ I: ['-1'], S: ['A'] }, { B: ['T'] })
      runTest({ I: ['-1'], S: ['A'] }, { F: ['0'], B: ['T'] })

      runTest({ B: ['T'], F: ['0'], I: ['1'] }, { S: ['A'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T'], F: ['0'], S: ['A'] }, { I: ['0'] })
      runTest({ F: ['0'], I: ['0'], S: ['A'] }, { B: ['T'] })
    })
    it('with soft evidence', () => {
      runTest({ B: ['T'], F: ['0'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], F: ['0'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'], F: ['0'] }, { I: ['0'], S: ['A', 'C'] })

      runTest({ B: ['T'], I: ['-1'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1'] }, { S: ['A', 'C'] })
      runTest({ B: ['T'], I: ['-1'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'], I: ['-1'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ B: ['T'], I: ['-1'] }, { F: ['0', '0.5'], S: ['A', 'C'] })

      runTest({ B: ['T'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], S: ['A'] }, { F: ['0', '0.5'], I: ['0'] })
      runTest({ B: ['T'], S: ['A'] }, { F: ['0'], I: ['0', '1'] })
      runTest({ B: ['T'], S: ['A'] }, { F: ['0', '0.5'], I: ['0', '1'] })

      runTest({ F: ['0'], I: ['-1'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], I: ['-1'] }, { S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1'] }, { B: ['T'], S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0'], I: ['-1'] }, { B: ['T', 'F'], S: ['A', 'B'] })

      runTest({ F: ['0'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T'], I: ['0', '1'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T', 'F'], I: ['0'] })
      runTest({ F: ['0'], S: ['A'] }, { B: ['T', 'F'], I: ['0', '1'] })

      runTest({ I: ['-1'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ I: ['-1'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ I: ['-1'], S: ['A'] }, { F: ['0'], B: ['T', 'F'] })
      runTest({ I: ['-1'], S: ['A'] }, { F: ['0', '0.5'], B: ['T'] })
      runTest({ I: ['-1'], S: ['A'] }, { F: ['0', '0.5'], B: ['T', 'F'] })

      runTest({ B: ['T'], F: ['0'], I: ['1'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], F: ['0'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ F: ['0'], I: ['0'], S: ['A'] }, { B: ['T', 'F'] })
    })
  })
  describe('have the same result for cumulative joint probability', () => {
    it('with no evidence', () => {
      runTest({ B: ['T', 'F'], F: ['0'] })
      runTest({ B: ['T', 'F'], I: ['-1'] })
      runTest({ B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['-1'] })
      runTest({ F: ['0', '0.5'], S: ['A'] })
      runTest({ I: ['-1', '0'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1', '0'] })
      runTest({ B: ['T'], S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1', '0'] })
      runTest({ F: ['0'], S: ['A', 'B'] })
      runTest({ I: ['-1'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], F: ['0'], I: ['1'] })
      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0'], S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A'] })

      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A'] })

      runTest({ B: ['T', 'F'], F: ['0'], I: ['1', '0'] })
      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A', 'B'] })

      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A'] })

      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1', '0'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A', 'B'] })

      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1', '0'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'B'] })

      runTest({ B: ['T'], F: ['0'], I: ['1', '0'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0'], S: ['A', 'C'] })
      runTest({ F: ['0'], I: ['0'], S: ['A', 'B'] })

      runTest({ B: ['T', 'F'], F: ['0'], I: ['0'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['0'], S: ['A'] })
      runTest({ B: ['T'], F: ['0'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'], F: ['0'], I: ['0'], S: ['A', 'B'] })

      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['0'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0'], I: ['0'], S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'], F: ['0'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'], F: ['0'], I: ['0', '1'], S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'C'] })
    })
    it('with hard evidence', () => {
      runTest({ B: ['T'], F: ['0', '0.5'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], F: ['0'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { I: ['0'] })

      runTest({ B: ['T', 'F'], F: ['0'] }, { S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { S: ['A'] })

      runTest({ B: ['T', 'F'], F: ['0'] }, { I: ['0'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { I: ['0'], S: ['A'] })
      runTest({ B: ['T', 'T'], F: ['0', '0.5'] }, { I: ['0'], S: ['A'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { S: ['A'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { S: ['A'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0'], S: ['A'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0'], S: ['A'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0'], S: ['A'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { I: ['0'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { I: ['0'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0'], I: ['0'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0'], I: ['0'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0'], I: ['0'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { S: ['A'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { S: ['A'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T'], S: ['A'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T'], S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T'], S: ['A'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { I: ['0'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { I: ['0'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T'], I: ['0'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T'], I: ['0'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T'], I: ['0'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { B: ['T'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { B: ['T'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { B: ['T'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0'], B: ['T'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0'], B: ['T'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0'], B: ['T'] })

      runTest({ B: ['T', 'F'], F: ['0'], I: ['1'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0'], I: ['1', '0'] }, { S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1', '0'] }, { S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1', '0'] }, { S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1'] }, { S: ['A'] })
      runTest({ B: ['T'], F: ['0'], I: ['1', '0'] }, { S: ['A'] })

      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A', 'B'] }, { F: ['0'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A'] }, { F: ['0'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A', 'B'] }, { F: ['0'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0'] })

      runTest({ B: ['T', 'F'], F: ['0'], S: ['A'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], F: ['0'], S: ['A', 'B'] }, { I: ['0'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A'] }, { I: ['0'] })
      runTest({ B: ['T'], F: ['0'], S: ['A', 'B'] }, { I: ['0'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0'] })

      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A', 'B'] }, { B: ['T'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A'] }, { B: ['T'] })
      runTest({ F: ['0'], I: ['0'], S: ['A', 'B'] }, { B: ['T'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A', 'B'] }, { B: ['T'] })
    })
    it('with soft evidence', () => {
      runTest({ B: ['T', 'F'], F: ['0'] }, { I: ['0', '1'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { I: ['0', '1'] })

      runTest({ B: ['T', 'F'], F: ['0'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { S: ['A', 'B'] })

      runTest({ B: ['T', 'F'], F: ['0'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { I: ['0', '1'], S: ['A'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { I: ['0', '1'], S: ['A'] })

      runTest({ B: ['T', 'F'], F: ['0'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'], F: ['0', '0.5'] }, { I: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'] }, { I: ['0'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0', '0.5'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { S: ['A', 'C'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { S: ['A', 'C'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0', '0.5'], S: ['A'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0', '0.5'], S: ['A'] })

      runTest({ B: ['T', 'F'], I: ['-1'] }, { F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ B: ['T'], I: ['-1', '0'] }, { F: ['0', '0.5'], S: ['A', 'C'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'] }, { F: ['0', '0.5'], S: ['A', 'C'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0', '0.5'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { I: ['0', '1'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0', '0.5'], I: ['0'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0', '0.5'], I: ['0'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0', '0.5'], I: ['0'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0'], I: ['0', '1'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0'], I: ['0', '1'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0'], I: ['0', '1'] })

      runTest({ B: ['T', 'F'], S: ['A'] }, { F: ['0', '0.5'], I: ['0', '1'] })
      runTest({ B: ['T'], S: ['A', 'B'] }, { F: ['0', '0.5'], I: ['0', '1'] })
      runTest({ B: ['T', 'F'], S: ['A', 'B'] }, { F: ['0', '0.5'], I: ['0', '1'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T', 'F'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { S: ['A', 'B'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { S: ['A', 'B'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T'], S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T'], S: ['A', 'B'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T'], S: ['A', 'B'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T', 'F'], S: ['A'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T', 'F'], S: ['A'] })

      runTest({ F: ['0', '0.5'], I: ['-1'] }, { B: ['T', 'F'], S: ['A', 'B'] })
      runTest({ F: ['0'], I: ['-1', '0'] }, { B: ['T', 'F'], S: ['A', 'B'] })
      runTest({ F: ['0', '0.5'], I: ['-1', '0'] }, { B: ['T', 'F'], S: ['A', 'B'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0', '1'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T', 'F'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T'], I: ['0', '1'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T'], I: ['0', '1'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T'], I: ['0', '1'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T', 'F'], I: ['0'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T', 'F'], I: ['0'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T', 'F'], I: ['0'] })

      runTest({ F: ['0', '0.5'], S: ['A'] }, { B: ['T', 'F'], I: ['0', '1'] })
      runTest({ F: ['0'], S: ['A', 'B'] }, { B: ['T', 'F'], I: ['0', '1'] })
      runTest({ F: ['0', '0.5'], S: ['A', 'B'] }, { B: ['T', 'F'], I: ['0', '1'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0', '0.5'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { B: ['T', 'F'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0'], B: ['T', 'F'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0'], B: ['T', 'F'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0'], B: ['T', 'F'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0', '0.5'], B: ['T'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0', '0.5'], B: ['T'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0', '0.5'], B: ['T'] })

      runTest({ I: ['-1', '0'], S: ['A'] }, { F: ['0', '0.5'], B: ['T', 'F'] })
      runTest({ I: ['-1'], S: ['A', 'B'] }, { F: ['0', '0.5'], B: ['T', 'F'] })
      runTest({ I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0', '0.5'], B: ['T', 'F'] })

      runTest({ B: ['T', 'F'], F: ['0'], I: ['1'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1'] }, { S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0'], I: ['1', '0'] }, { S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0'], I: ['1', '0'] }, { S: ['A', 'B'] })
      runTest({ B: ['T'], F: ['0', '0.5'], I: ['1', '0'] }, { S: ['A', 'B'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], I: ['1', '0'] }, { S: ['A', 'B'] })

      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1'], S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'], I: ['-1'], S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T'], I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0', '0.5'] })
      runTest({ B: ['T', 'F'], I: ['-1', '0'], S: ['A', 'B'] }, { F: ['0', '0.5'] })

      runTest({ B: ['T', 'F'], F: ['0'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], F: ['0'], S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ B: ['T', 'F'], F: ['0'], S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ B: ['T'], F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0', '1'] })
      runTest({ B: ['T', 'F'], F: ['0', '0.5'], S: ['A', 'B'] }, { I: ['0', '1'] })

      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], I: ['0'], S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'], I: ['0'], S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ F: ['0'], I: ['0', '1'], S: ['A', 'B'] }, { B: ['T', 'F'] })
      runTest({ F: ['0', '0.5'], I: ['0', '1'], S: ['A', 'B'] }, { B: ['T', 'F'] })
    })
  })
})
