import { network } from '../../models/asia'
import { runTests } from './helpers'

// These gold standard values were computed indepdendently in R.
const GOLD_STANDARD: [string[], number][] = [
  [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T'], 1.323000000E-05],
  [['F', 'T', 'T', 'T', 'T', 'T', 'T', 'T'], 2.619540000E-04],
  [['T', 'F', 'T', 'T', 'T', 'T', 'T', 'T'], 2.513700000E-04],
  [['F', 'F', 'T', 'T', 'T', 'T', 'T', 'T'], 2.593344600E-02],
  [['T', 'T', 'F', 'T', 'T', 'T', 'T', 'T'], 6.615000000E-07],
  [['F', 'T', 'F', 'T', 'T', 'T', 'T', 'T'], 1.309770000E-05],
  [['T', 'F', 'F', 'T', 'T', 'T', 'T', 'T'], 1.256850000E-05],
  [['F', 'F', 'F', 'T', 'T', 'T', 'T', 'T'], 1.296672300E-03],
  [['T', 'T', 'T', 'F', 'T', 'T', 'T', 'T'], 1.190700000E-04],
  [['F', 'T', 'T', 'F', 'T', 'T', 'T', 'T'], 2.357586000E-03],
  [['T', 'F', 'T', 'F', 'T', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'T', 'T', 'T'], 6.548850000E-05],
  [['F', 'T', 'F', 'F', 'T', 'T', 'T', 'T'], 1.296672300E-03],
  [['T', 'F', 'F', 'F', 'T', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'T', 'T', 'T'], 1.026000000E-04],
  [['F', 'F', 'T', 'F', 'F', 'T', 'T', 'T'], 1.058508000E-02],
  [['T', 'T', 'F', 'F', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'T', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'T', 'T', 'T'], 5.643000000E-05],
  [['F', 'F', 'F', 'F', 'F', 'T', 'T', 'T'], 5.821794000E-03],
  [['T', 'T', 'T', 'T', 'T', 'F', 'T', 'T'], 2.700000000E-07],
  [['F', 'T', 'T', 'T', 'T', 'F', 'T', 'T'], 5.346000000E-06],
  [['T', 'F', 'T', 'T', 'T', 'F', 'T', 'T'], 5.130000000E-06],
  [['F', 'F', 'T', 'T', 'T', 'F', 'T', 'T'], 5.292540000E-04],
  [['T', 'T', 'F', 'T', 'T', 'F', 'T', 'T'], 1.350000000E-08],
  [['F', 'T', 'F', 'T', 'T', 'F', 'T', 'T'], 2.673000000E-07],
  [['T', 'F', 'F', 'T', 'T', 'F', 'T', 'T'], 2.565000000E-07],
  [['F', 'F', 'F', 'T', 'T', 'F', 'T', 'T'], 2.646270000E-05],
  [['T', 'T', 'T', 'F', 'T', 'F', 'T', 'T'], 2.430000000E-06],
  [['F', 'T', 'T', 'F', 'T', 'F', 'T', 'T'], 4.811400000E-05],
  [['T', 'F', 'T', 'F', 'T', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'F', 'T', 'T'], 1.336500000E-06],
  [['F', 'T', 'F', 'F', 'T', 'F', 'T', 'T'], 2.646270000E-05],
  [['T', 'F', 'F', 'F', 'T', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'F', 'T', 'T'], 1.949400000E-03],
  [['F', 'F', 'T', 'F', 'F', 'F', 'T', 'T'], 2.011165200E-01],
  [['T', 'T', 'F', 'F', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'F', 'T', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'F', 'T', 'T'], 1.072170000E-03],
  [['F', 'F', 'F', 'F', 'F', 'F', 'T', 'T'], 1.106140860E-01],
  [['T', 'T', 'T', 'T', 'T', 'T', 'F', 'T'], 6.860000000E-06],
  [['F', 'T', 'T', 'T', 'T', 'T', 'F', 'T'], 1.358280000E-04],
  [['T', 'F', 'T', 'T', 'T', 'T', 'F', 'T'], 1.303400000E-04],
  [['F', 'F', 'T', 'T', 'T', 'T', 'F', 'T'], 1.344697200E-02],
  [['T', 'T', 'F', 'T', 'T', 'T', 'F', 'T'], 1.200500000E-06],
  [['F', 'T', 'F', 'T', 'T', 'T', 'F', 'T'], 2.376990000E-05],
  [['T', 'F', 'F', 'T', 'T', 'T', 'F', 'T'], 2.280950000E-05],
  [['F', 'F', 'F', 'T', 'T', 'T', 'F', 'T'], 2.353220100E-03],
  [['T', 'T', 'T', 'F', 'T', 'T', 'F', 'T'], 6.174000000E-05],
  [['F', 'T', 'T', 'F', 'T', 'T', 'F', 'T'], 1.222452000E-03],
  [['T', 'F', 'T', 'F', 'T', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'T', 'F', 'T'], 1.188495000E-04],
  [['F', 'T', 'F', 'F', 'T', 'T', 'F', 'T'], 2.353220100E-03],
  [['T', 'F', 'F', 'F', 'T', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'T', 'F', 'T'], 8.550000000E-06],
  [['F', 'F', 'T', 'F', 'F', 'T', 'F', 'T'], 8.820900000E-04],
  [['T', 'T', 'F', 'F', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'T', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'T', 'F', 'T'], 1.645875000E-05],
  [['F', 'F', 'F', 'F', 'F', 'T', 'F', 'T'], 1.698023250E-03],
  [['T', 'T', 'T', 'T', 'T', 'F', 'F', 'T'], 1.400000000E-07],
  [['F', 'T', 'T', 'T', 'T', 'F', 'F', 'T'], 2.772000000E-06],
  [['T', 'F', 'T', 'T', 'T', 'F', 'F', 'T'], 2.660000000E-06],
  [['F', 'F', 'T', 'T', 'T', 'F', 'F', 'T'], 2.744280000E-04],
  [['T', 'T', 'F', 'T', 'T', 'F', 'F', 'T'], 2.450000000E-08],
  [['F', 'T', 'F', 'T', 'T', 'F', 'F', 'T'], 4.851000000E-07],
  [['T', 'F', 'F', 'T', 'T', 'F', 'F', 'T'], 4.655000000E-07],
  [['F', 'F', 'F', 'T', 'T', 'F', 'F', 'T'], 4.802490000E-05],
  [['T', 'T', 'T', 'F', 'T', 'F', 'F', 'T'], 1.260000000E-06],
  [['F', 'T', 'T', 'F', 'T', 'F', 'F', 'T'], 2.494800000E-05],
  [['T', 'F', 'T', 'F', 'T', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'F', 'F', 'T'], 2.425500000E-06],
  [['F', 'T', 'F', 'F', 'T', 'F', 'F', 'T'], 4.802490000E-05],
  [['T', 'F', 'F', 'F', 'T', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'F', 'F', 'T'], 1.624500000E-04],
  [['F', 'F', 'T', 'F', 'F', 'F', 'F', 'T'], 1.675971000E-02],
  [['T', 'T', 'F', 'F', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'F', 'F', 'T'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'F', 'F', 'T'], 3.127162500E-04],
  [['F', 'F', 'F', 'F', 'F', 'F', 'F', 'T'], 3.226244175E-02],
  [['T', 'T', 'T', 'T', 'T', 'T', 'T', 'F'], 1.470000000E-06],
  [['F', 'T', 'T', 'T', 'T', 'T', 'T', 'F'], 2.910600000E-05],
  [['T', 'F', 'T', 'T', 'T', 'T', 'T', 'F'], 2.793000000E-05],
  [['F', 'F', 'T', 'T', 'T', 'T', 'T', 'F'], 2.881494000E-03],
  [['T', 'T', 'F', 'T', 'T', 'T', 'T', 'F'], 7.350000000E-08],
  [['F', 'T', 'F', 'T', 'T', 'T', 'T', 'F'], 1.455300000E-06],
  [['T', 'F', 'F', 'T', 'T', 'T', 'T', 'F'], 1.396500000E-06],
  [['F', 'F', 'F', 'T', 'T', 'T', 'T', 'F'], 1.440747000E-04],
  [['T', 'T', 'T', 'F', 'T', 'T', 'T', 'F'], 1.323000000E-05],
  [['F', 'T', 'T', 'F', 'T', 'T', 'T', 'F'], 2.619540000E-04],
  [['T', 'F', 'T', 'F', 'T', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'T', 'T', 'F'], 7.276500000E-06],
  [['F', 'T', 'F', 'F', 'T', 'T', 'T', 'F'], 1.440747000E-04],
  [['T', 'F', 'F', 'F', 'T', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'T', 'T', 'F'], 2.565000000E-05],
  [['F', 'F', 'T', 'F', 'F', 'T', 'T', 'F'], 2.646270000E-03],
  [['T', 'T', 'F', 'F', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'T', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'T', 'T', 'F'], 1.410750000E-05],
  [['F', 'F', 'F', 'F', 'F', 'T', 'T', 'F'], 1.455448500E-03],
  [['T', 'T', 'T', 'T', 'T', 'F', 'T', 'F'], 3.000000000E-08],
  [['F', 'T', 'T', 'T', 'T', 'F', 'T', 'F'], 5.940000000E-07],
  [['T', 'F', 'T', 'T', 'T', 'F', 'T', 'F'], 5.700000000E-07],
  [['F', 'F', 'T', 'T', 'T', 'F', 'T', 'F'], 5.880600000E-05],
  [['T', 'T', 'F', 'T', 'T', 'F', 'T', 'F'], 1.500000000E-09],
  [['F', 'T', 'F', 'T', 'T', 'F', 'T', 'F'], 2.970000000E-08],
  [['T', 'F', 'F', 'T', 'T', 'F', 'T', 'F'], 2.850000000E-08],
  [['F', 'F', 'F', 'T', 'T', 'F', 'T', 'F'], 2.940300000E-06],
  [['T', 'T', 'T', 'F', 'T', 'F', 'T', 'F'], 2.700000000E-07],
  [['F', 'T', 'T', 'F', 'T', 'F', 'T', 'F'], 5.346000000E-06],
  [['T', 'F', 'T', 'F', 'T', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'F', 'T', 'F'], 1.485000000E-07],
  [['F', 'T', 'F', 'F', 'T', 'F', 'T', 'F'], 2.940300000E-06],
  [['T', 'F', 'F', 'F', 'T', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'F', 'T', 'F'], 4.873500000E-04],
  [['F', 'F', 'T', 'F', 'F', 'F', 'T', 'F'], 5.027913000E-02],
  [['T', 'T', 'F', 'F', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'F', 'T', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'F', 'T', 'F'], 2.680425000E-04],
  [['F', 'F', 'F', 'F', 'F', 'F', 'T', 'F'], 2.765352150E-02],
  [['T', 'T', 'T', 'T', 'T', 'T', 'F', 'F'], 2.940000000E-06],
  [['F', 'T', 'T', 'T', 'T', 'T', 'F', 'F'], 5.821200000E-05],
  [['T', 'F', 'T', 'T', 'T', 'T', 'F', 'F'], 5.586000000E-05],
  [['F', 'F', 'T', 'T', 'T', 'T', 'F', 'F'], 5.762988000E-03],
  [['T', 'T', 'F', 'T', 'T', 'T', 'F', 'F'], 5.145000000E-07],
  [['F', 'T', 'F', 'T', 'T', 'T', 'F', 'F'], 1.018710000E-05],
  [['T', 'F', 'F', 'T', 'T', 'T', 'F', 'F'], 9.775500000E-06],
  [['F', 'F', 'F', 'T', 'T', 'T', 'F', 'F'], 1.008522900E-03],
  [['T', 'T', 'T', 'F', 'T', 'T', 'F', 'F'], 2.646000000E-05],
  [['F', 'T', 'T', 'F', 'T', 'T', 'F', 'F'], 5.239080000E-04],
  [['T', 'F', 'T', 'F', 'T', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'T', 'F', 'F'], 5.093550000E-05],
  [['F', 'T', 'F', 'F', 'T', 'T', 'F', 'F'], 1.008522900E-03],
  [['T', 'F', 'F', 'F', 'T', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'T', 'F', 'F'], 7.695000000E-05],
  [['F', 'F', 'T', 'F', 'F', 'T', 'F', 'F'], 7.938810000E-03],
  [['T', 'T', 'F', 'F', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'T', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'T', 'F', 'F'], 1.481287500E-04],
  [['F', 'F', 'F', 'F', 'F', 'T', 'F', 'F'], 1.528220925E-02],
  [['T', 'T', 'T', 'T', 'T', 'F', 'F', 'F'], 6.000000000E-08],
  [['F', 'T', 'T', 'T', 'T', 'F', 'F', 'F'], 1.188000000E-06],
  [['T', 'F', 'T', 'T', 'T', 'F', 'F', 'F'], 1.140000000E-06],
  [['F', 'F', 'T', 'T', 'T', 'F', 'F', 'F'], 1.176120000E-04],
  [['T', 'T', 'F', 'T', 'T', 'F', 'F', 'F'], 1.050000000E-08],
  [['F', 'T', 'F', 'T', 'T', 'F', 'F', 'F'], 2.079000000E-07],
  [['T', 'F', 'F', 'T', 'T', 'F', 'F', 'F'], 1.995000000E-07],
  [['F', 'F', 'F', 'T', 'T', 'F', 'F', 'F'], 2.058210000E-05],
  [['T', 'T', 'T', 'F', 'T', 'F', 'F', 'F'], 5.400000000E-07],
  [['F', 'T', 'T', 'F', 'T', 'F', 'F', 'F'], 1.069200000E-05],
  [['T', 'F', 'T', 'F', 'T', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'F', 'T', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'F', 'T', 'F', 'F', 'F'], 1.039500000E-06],
  [['F', 'T', 'F', 'F', 'T', 'F', 'F', 'F'], 2.058210000E-05],
  [['T', 'F', 'F', 'F', 'T', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'F', 'T', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'T', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'F', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'F', 'F', 'T', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'T', 'T', 'F', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'T', 'F', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'T', 'F', 'F', 'F', 'F', 'F'], 1.462050000E-03],
  [['F', 'F', 'T', 'F', 'F', 'F', 'F', 'F'], 1.508373900E-01],
  [['T', 'T', 'F', 'F', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['F', 'T', 'F', 'F', 'F', 'F', 'F', 'F'], 0.000000000E+00],
  [['T', 'F', 'F', 'F', 'F', 'F', 'F', 'F'], 2.814446250E-03],
  [['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'], 2.903619758E-01],
]

const names: string[] = ['VisitToAsia', 'Tuberculosis', 'Smoker', 'LungCancer', 'TbOrCa', 'AbnormalXRay', 'Bronchitis', 'Dyspnea']
const testValues: string[][][] = [
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F']],
  [['T'], ['F'], ['T', 'F']],
  [['T'], ['F']],
  [['T'], ['F']],
]

describe('inference on asia network', () => runTests(network, names, testValues, GOLD_STANDARD))
