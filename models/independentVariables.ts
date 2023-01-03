import { INode, ICptWithParents, ICptWithoutParents } from '../src/types'
import { COIN } from './coinflip'

export const allNodes = [{ ...COIN, id: 'COIN' }, { ...COIN, id: 'COIN2' }]
export const network: { [name: string]: { levels: string[]; parents: string[]; cpt?: ICptWithParents | ICptWithoutParents}} = {}
allNodes.forEach((node: INode) => {
  network[node.id] = {
    levels: node.states,
    parents: node.parents,
    cpt: node.cpt,
  }
})
