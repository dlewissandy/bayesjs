import { INode, ICptWithParents, ICptWithoutParents } from '../src/types'

const SALES_STRENGTH: INode =
  {
    id: 'SALES_STRENGTH',
    parents: ['SURVEY_RESULT', 'PRODUCE'],
    states: ['STRONG', 'WEAK', 'N/A'],
    cpt: [
      { when: { SURVEY_RESULT: 'POSITIVE', PRODUCE: 'true' }, then: { STRONG: 0.7739999999999999, WEAK: 0.22599999999999998, 'N/A': 0.0 } },
      { when: { SURVEY_RESULT: 'NEGATIVE', PRODUCE: 'true' }, then: { STRONG: 0.08699999999999998, WEAK: 0.9129999999999999, 'N/A': 0.0 } },
      { when: { SURVEY_RESULT: 'N/A', PRODUCE: 'true' }, then: { STRONG: 0.30000000000000004, WEAK: 0.7000000000000001, 'N/A': 0.0 } },
      { when: { SURVEY_RESULT: 'POSITIVE', PRODUCE: 'false' }, then: { STRONG: 0, WEAK: 0, 'N/A': 1 } },
      { when: { SURVEY_RESULT: 'NEGATIVE', PRODUCE: 'false' }, then: { STRONG: 0, WEAK: 0, 'N/A': 1 } },
      { when: { SURVEY_RESULT: 'N/A', PRODUCE: 'false' }, then: { STRONG: 0, WEAK: 0, 'N/A': 1 } },
    ],
  }

const SURVEY_RESULT: INode = {
  id: 'SURVEY_RESULT',
  parents: ['SURVEY_MARKET'],
  states: ['POSITIVE', 'NEGATIVE', 'N/A'],
  cpt: [
    { when: { SURVEY_MARKET: 'true' }, then: { POSITIVE: 0.31, NEGATIVE: 0.69, 'N/A': 0 } },
    { when: { SURVEY_MARKET: 'false' }, then: { POSITIVE: 0, NEGATIVE: 0, 'N/A': 1 } },
  ],
}

const PRODUCE: INode =
    {
      id: 'PRODUCE',
      parents: ['SURVEY_RESULT'],
      states: ['true', 'false'],
      cpt: [
        { when: { SURVEY_RESULT: 'POSITIVE' }, then: { true: 0.5, false: 0.5 } },
        { when: { SURVEY_RESULT: 'NEGATIVE' }, then: { true: 0.5, false: 0.5 } },
        { when: { SURVEY_RESULT: 'N/A' }, then: { true: 0.5, false: 0.5 } },
      ],
    }

const SURVEY_MARKET: INode = {
  id: 'SURVEY_MARKET',
  parents: [],
  states: ['true', 'false'],
  cpt: { true: 0.5, false: 0.5 },
}

export const allNodes = [SALES_STRENGTH, SURVEY_RESULT, SURVEY_MARKET, PRODUCE]
export const network: { [name: string]: { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents}} = {}
allNodes.forEach((node: INode) => {
  network[node.id] = {
    levels: node.states,
    parents: node.parents,
    cpt: node.cpt,
  }
})
