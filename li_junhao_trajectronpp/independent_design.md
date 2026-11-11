# Independent trajectory-only pedestrian design

Read the pinned coordinate data into time-segmented neutral rows. Use the
locked manifest to construct an eight-step target and neighbor history.
Compute velocity and acceleration with backward differences within each
window. Initialize the first velocity and first two accelerations to zero.
Target perturbations precede differencing. The graph uses only neighbors with
eight points and masks observations individually beyond three meters.

A target LSTM and a single pedestrian edge-type LSTM encode the histories.
Neighbor states are summed within the edge type. The type attention operates
over one element when an edge is present. An LSTM posterior encoder sees the
future position sequence only during training. A categorical prior over 25
latent classes supplies test samples. An autoregressive GRU parameterizes one
correlated bivariate Gaussian velocity at each step, conditional on history
and the sampled latent class. The decoder uses velocity NLL with KL
regularization. Integrate sampled velocity with dt=0.4 and invert the
target-relative coordinates.

| Design choice | Deviation or consequence |
|---|---|
| Trajectories and one pedestrian type | The paper supports heterogeneous types, maps and ego-plan conditioning. The group protocol uses only trajectories. |
| Complete neighbor history and hard radius mask | This simplifies the reference's variable histories and edge addition/removal filters. |
| Per-window backward differences | Eliminates anchor look-ahead through central differences. Input normalization is fixed, with target-relative position and velocity divided by two. |
| Velocity NLL, posterior sampling and fixed KL schedule | The independent loss and sampler implement the chosen pedestrian approximation. The official objective and distribution integration are broader. |
| Manifest, target-only perturbation and 20 samples | Group-specific protocol. Metrics are computed in original metric coordinates and are tied to this protocol. |
| Training-only diagnostic thresholds | Speed, acceleration and jerk use train 99th percentiles. The proximity diagnostic uses the train fifth percentile of contemporaneous pair distances. |

Inference accepts history features, graph masks and a coordinate origin.
Future labels and absolute observations travel independently to the exporter
and evaluator. Stage signatures bind each run to its code, inputs and
checkpoint candidates. The forecasting module can run with only the local
independent package and the frozen data representation.
