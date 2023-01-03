import { INode, ICptWithParents, ICptWithoutParents } from '../src/types'
import { COIN } from './coinflip'

const WIN: INode = {
  id: 'WIN',
  parents: ['COIN'],
  states: ['TRUE', 'FALSE'],
  cpt: [
    { when: { COIN: 'HEADS' }, then: { TRUE: 1, FALSE: 0 } },
    { when: { COIN: 'TAILS' }, then: { TRUE: 0, FALSE: 1 } },
  ],
}

export const allNodes = [COIN, WIN]
export const network: { [name: string]: { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents}} = {}
allNodes.forEach((node: INode) => {
  network[node.id] = {
    levels: node.states,
    parents: node.parents,
    cpt: node.cpt,
  }
})
