# Trajectron++ speaker notes

## Frame 1 · Forecast paths from history

The four steps are graph history, a categorical latent, a velocity
decoder and integration. The graph gives the target access to nearby
pedestrian histories. A sampled latent changes the velocity sequence, so
the decoder can generate several futures. The original derivative
pipeline could use positions after the anchor. We recompute velocity and
acceleration within the visible window. When we change future positions
for either the target or its neighbor, the same-seed predictions stay
identical. The revised protocol covers 15 runs and all 90 bundles.

## Frame 2 · Coverage and typical error

Follow the green curve on the left. If we score one path, ADE is 0.857
meters. If we allow the evaluator to choose the closest of 20 samples from
the same stream, it becomes 0.330 meters. That is a 62 percent reduction
in best-sample error. The orange curve barely moves, at about 0.855
meters. Best-of-20 therefore tells us how well the forecast covers the
realized future. Mean error describes the quality of a typical sample.
The report also breaks this down by scene. Clean minADE is 0.600 meters
on ETH and 0.222 on Zara2.

## Frame 3 · Noise and short gaps

These are changes against clean for the same windows. Open points show
individual scenes, and diamonds give the equal-weight average. The
0.20-meter noise condition adds 2.178 meters of meanADE. Dropping the
last two observations and filling causally adds 0.092 meters. Even
five centimeters of noise hurts more than missing two on every holdout.
One possible explanation is that differencing noisy positions magnifies
the velocity and acceleration disturbance. This is a hypothesis.
The measured conclusion applies to target-only perturbations with
neighbor histories still available. When we compare models, keep the
sample budgets, corruption rules and timing boundaries visible.
