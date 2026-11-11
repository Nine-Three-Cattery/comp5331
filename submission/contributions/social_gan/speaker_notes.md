# Social-GAN speaker notes

## Frame 1 · Social-GAN pools observed neighbors before decoding

Imagine two people walking toward each other. Either person might slow
down or move aside, so one predicted line can leave out a reasonable
outcome. Social-GAN gives us a set of possible paths from the same
history.

The model reads eight positions per person. It encodes the target and
its visible neighbors, then pools the neighbors' motion and their
relative positions once, at the last observed point. Each Gaussian draw
gives the decoder a different starting state. The decoder predicts
twelve steps as corrections to a simple constant-velocity estimate.

During training, a separate LSTM learns to distinguish recorded
trajectories from generated ones. We also draw twenty possible futures
and train on the one closest to the recorded future. The next slide
shows what happens when we score different numbers of draws from the
trained model.

## Frame 2 · More draws improve the best match

Both lines come from the same checkpoints and the same saved samples.
We score the first draw, then the first five, then ten, then all twenty.
The green line asks how close the best draw came after we saw the real
path. With twenty draws, that error is 0.32 meters. The brown line asks
what error to expect if we pick a draw at random. That is 0.84 meters.
A close match can be present even when other candidates are farther away.

We repeated the comparison across five held-out scenes and three
training seeds. We also added noise to the target's past while keeping
its neighbors and the future fixed. With twenty centimeters of position
noise, mean error rose by 0.47 meters compared with the clean runs.

## Frame 3 · Twenty samples can still miss a trajectory

The two panels on the left follow the same pedestrian in ETH. The dark
line is the history the model received, and the orange line is the
recorded future. The thin green lines are the sampled futures. The
dashed green line is the closest one. The noisy input happens to produce
a closer best match here, even though mean error rose across the benchmark.

The Univ panel shows the clean example with the largest best-of-twenty
error for seed zero. The pedestrian almost reverses direction. Even
the closest draw misses by 4.69 meters on average over the forecast.
The runtime underneath includes history and neighbor preprocessing, then
twenty paths each for 128 targets on the local CPU. I'll hand over to
Zongmin for Social-STGCNN and its
spatiotemporal graph.
