import { FastPotential } from '../..'
import { TowerOfDerivatives } from '../TowerOfDerivatives'

export type ObjectiveFunction = (xs: FastPotential[]) => TowerOfDerivatives
