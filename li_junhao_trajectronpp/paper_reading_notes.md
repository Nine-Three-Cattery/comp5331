# Trajectron++ paper reading notes

Salzmann, Ivanovic, Chakravarty and Pavone, *Trajectron++: Dynamically-Feasible
Trajectory Forecasting With Heterogeneous Data*, ECCV 2020.
[Proceedings paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf)

- Section 3 defines forecasting from current and past agent states. Any future
  information supplied for the optional ego-agent setting is a distinct
  conditioning input. The Group 7 pedestrian benchmark omits that setting.
- Section 4 and Figure 2 combine recurrent history and edge encoders with a
  discrete CVAE and recurrent probabilistic decoder. The posterior may see
  training futures. The forecasting prior uses context available at the anchor.
- Section 4.3 integrates controls into trajectories through an agent dynamics
  model. For pedestrians, the first-order integrator advances position using
  sampled velocity. This represents a kinematic update, with no social
  collision constraint in our implementation.
- The ETH/UCY experiment motivates the pedestrian-only configuration. Use the
  Group 7 split, causal features and sample budget for project comparisons.
  Published scores describe the paper's own protocol.

The repository was consulted to check pedestrian dimensions, radius, latent
category count, and the interpretation of the velocity integrator. The
implementation decisions and deviations are tracked in the design and source
inventory files.
