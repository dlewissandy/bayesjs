export type TowerOfDerivatives = {
  /** The set of parameters at which the objective function was evaluated  */
  xs: number[][];
  /**  The value of the objective function at the given set of parameters */
  value: number;
  /** The gradient of the objective function at the given coordinate.   This is stored
   * using the same indexing scheme as the parameters. */
  gradient: number[][];
  /**  The hessian matrix of the objective function, evaluated at the given parameter
   * values.   Since this is a diagonal matrix, we store it using the same indexing
   * scheme as the parameters */
  hessian: number[][];
  /** true if the hessian was not safely positive definite.   An approximator is returned. */
  hessianIsApproximate: boolean;
  /** The Newton or quasi-Newton descent direction from the current set of parameters. */
  descentDirection: number[][];
  /** The magnitude (l2 norm) of the descent direction */
  descentDirectionMagnitude: number;
  /** The directional derivative at the current set of coordinates.   For an descent
   *direction, the directional derivative should be positive. */
  directionalDerivative: number;
}
