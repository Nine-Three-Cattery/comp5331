# DSPIGCN presentation notes

## Frame 1 · Two views of the same motion

We observe the positions of a pedestrian and the people walking nearby. Zhengyang's implementation represents each step in two ways. One stream measures the displacement in meters. The other expresses the same movement as velocity in meters per second. A fourth-degree polynomial links the predicted positions and velocities during training. Its derivative gives the velocity expected from that curve. The loss penalizes disagreement with the network's predicted means. To forecast, the model samples possible future positions and assigns unused samples to a sequence of twelve steps. It returns twenty paths for the pedestrian. Neighbors need only be visible in the eight observed frames.

## Frame 2 · Twenty sampled paths

The left panel measures how far a typical sampled path lies from the true walk. On clean histories the average distance was 0.76 meters. If we know the true future and choose the best of the twenty paths, the error is 0.50 meters. These numbers describe the same forecasts. The right panel uses the Energy Score, which accounts for both the error and the spread of the sampled paths. Adding twenty centimeters of position noise to the target's history gave a mean error of 2.72 meters, compared with 0.91 meters when the last two observations were filled from the previous motion. We gave each held-out scene equal weight.

## Frame 3 · The same pedestrian under three conditions

All three panels track the same ETH pedestrian. We fixed the example from the shared window list before inspecting the forecasts. Green shows the true future and the lighter lines show individual predictions. On this particular window, mean error was 0.99 meters with the original history, 0.97 with noisy positions and 3.70 with the final two observations missing. These are three views of one person. The preceding chart summarizes all 34,161 test windows. Preparing, forecasting and returning predictions for the 128 registered timing targets took a median of 142.7 milliseconds on the measured device. We can now compare the six models on the same windows.
