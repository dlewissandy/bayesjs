import { INode, ICptWithParents, ICptWithoutParents } from '../src/types'

export const B: INode = {
  id: 'B',
  states: ['T', 'F'],
  parents: [],
  cpt: { T: 0.01, F: 0.98 },
}

export const I: INode = {
  id: 'I',
  states: ['-1', '0', '1'],
  parents: ['B'],
  cpt: [
    { when: { B: 'T' }, then: { '-1': 0.85, 0: 0.10, 1: 0.5 } },
    { when: { B: 'F' }, then: { '-1': 0.15, 0: 0.30, 1: 0.65 } },
  ],
}

export const F: INode = {
  id: 'F',
  states: ['-0.5', '0', '0.5'],
  parents: ['B'],
  cpt: [
    { when: { B: 'T' }, then: { '-0.5': 0.85, 0: 0.10, 0.5: 0.5 } },
    { when: { B: 'F' }, then: { '-0.5': 0.15, 0: 0.30, 0.5: 0.65 } },
  ],
}

export const S: INode = {
  id: 'S',
  states: ['A', 'B', 'C'],
  parents: ['I', 'F'],
  cpt: [
    { when: { I: '-1', F: '-0.5' }, then: { A: 0.85, B: 0.10, C: 0.05 } },
    { when: { I: '0', F: '-0.5' }, then: { A: 0.10, B: 0.05, C: 0.85 } },
    { when: { I: '1', F: '-0.5' }, then: { A: 0.05, B: 0.85, C: 0.1 } },

    { when: { I: '-1', F: '0' }, then: { A: 0.85, B: 0.05, C: 0.10 } },
    { when: { I: '0', F: '0' }, then: { A: 0.05, B: 0.10, C: 0.85 } },
    { when: { I: '1', F: '0' }, then: { A: 0.1, B: 0.85, C: 0.05 } },

    { when: { I: '-1', F: '0.5' }, then: { A: 0.10, B: 0.85, C: 0.5 } },
    { when: { I: '0', F: '0.5' }, then: { A: 0.85, B: 0.05, C: 0.1 } },
    { when: { I: '1', F: '0.5' }, then: { A: 0.05, B: 0.10, C: 0.85 } },
  ],
}

export const allNodes = [B, F, I, S]
export const network: { [name: string]: { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents}} = {}
allNodes.forEach((node: INode) => {
  network[node.id] = {
    levels: node.states,
    parents: node.parents,
    cpt: node.cpt,
  }
})
