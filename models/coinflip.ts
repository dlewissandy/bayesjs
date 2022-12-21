import { INode, ICptWithParents, ICptWithoutParents } from '../src/types'

export const COIN: INode = {
  id: 'COIN',
  parents: [],
  states: ['HEADS', 'TAILS'],
  cpt: { HEADS: 0.5, TAILS: 0.5 },

}

export const allNodes = [COIN]
export const network: { [name: string]: { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents}} = {}
allNodes.forEach((node: INode) => {
  network[node.id] = {
    levels: node.states,
    parents: node.parents,
    cpt: node.cpt,
  }
})
