# Trajectron++ speaker notes

## Frame 1 · Forecast paths from history

My Trajectron++ implementation starts with a graph of the target and nearby
pedestrians. A categorical latent conditions the velocity decoder, and
integrating the decoded velocities yields the forecast paths. We compute
velocity and acceleration within the observed window. Changing future
positions for either the target or a neighbor leaves the same-seed
predictions identical. We selected checkpoints on validation and audited
15 trained runs and 90 test bundles. That gives us one set of forecasts to
score at different sample budgets.

## Frame 2 · Sample budget and scene differences

Start on the left. We score the first one, five, ten and twenty draws from
the same clean forecast stream. Minimum ADE falls from 0.857 to 0.484,
0.394 and finally 0.330 meters, a 62 percent reduction. Mean ADE stays
near 0.855 meters. Minimum ADE uses the realized future to pick the best
draw, while mean ADE describes a typical draw.

Now fix both models at twenty draws. Each row on the right is Trajectron++
minus Social-GAN on the same clean test windows, averaged across the three
seeds. Negative means Trajectron++ was lower. It had lower minimum ADE on
two of five scenes, lower minimum FDE on three and lower Energy Score on
four. Social-GAN was lower on all three metrics in Hotel. The group
averages capture the overall pattern, and the table shows how it varies
by scene. Next, we hold the model fixed and change what it observes.

## Frame 3 · Noise and short gaps

These are changes against clean for the same windows. Open circles show
individual scenes, and diamonds give the equal-weight average. Adding
0.20 meters of noise increases mean ADE by 2.178 meters. Dropping and
causally filling the final two observations increases it by 0.092 meters.
Even five centimeters of noise hurts more than the two-point gap in every
scene. Differencing noisy positions could amplify the disturbance in
velocity and acceleration. This experiment changes only the target
history, with neighbor histories still available.

I’ll hand back to the group for the remaining model evidence. For the
group comparison, keep the draw budget in view for best-path scores and
read each corruption result against that model’s clean forecast.
