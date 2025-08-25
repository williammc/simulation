
Focus on simulation and simplify codebase

- Implement GTSAM based estimators: GtsamEkfEstimator and GtsamSWBAEstimator
- Integrate to tests and 'run.sh e2e/e2e-simple'
- Remove custom estimators: EKF, SWBA, SRIF
    - Remove related tests or 'run.sh e2e/e2e-simple' support

Reference code: tests/gtsam-comparison