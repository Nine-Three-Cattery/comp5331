# Trajectron++ speaker notes

Three-minute rehearsal. Timings include brief pauses to point at the slide.
Read the table entries in the script and let the audience scan the rest.

## Frame 1 · 0:00–0:48 · Forecast paths from history

I implemented the pedestrian-only version of Trajectron++. It takes the
target's observed path and nearby pedestrian histories. The model encodes
that graph, samples a latent category, and predicts velocities. We
integrate those velocities to get future paths.

We compute velocity and acceleration within the observed window. When we
changed future positions for the target or a neighbor, the forecasts
stayed identical with the same seed. We selected checkpoints on validation
and audited ninety test bundles from fifteen training runs. Now let's vary
the draw budget.

## Frame 2 · 0:48–2:04 · Sample budget and scene differences

*0:48–1:20. Point to the first and last rows of the left table.*

The left table scores the first one, five, ten or twenty paths from the
same clean forecasts. Minimum ADE drops from 0.857 to 0.330 meters, about
sixty-two percent. That score picks the closest path after seeing the
actual future. Mean ADE averages the errors across draws and stays near
0.855 meters.

*1:20–2:04. Point to the Hotel row and the minFDE column on the right.*

Both models get twenty draws on the same clean windows. Each entry
averages three seeds and subtracts Social-GAN from Trajectron++. Negative
values favor Trajectron++. Hotel's plus 0.087 in the middle column means
our minimum endpoint error was about nine centimeters higher there.
Across five scenes, Trajectron++ had lower minimum ADE on two, lower
minimum FDE on three, and lower Energy Score on four. Next, let's see
what happens when the input changes.

## Frame 3 · 2:04–3:00 · Noise and short gaps

*Point to the diamonds for five-centimeter noise and the two-point gap.*

This plot shows the change in mean ADE relative to clean input. The
circles are individual scenes, and the diamonds are the five-scene
averages. Five centimeters of input noise adds about sixty-three
centimeters of error on average. When we fill the final two missing
positions from the visible history, the increase is about nine centimeters.
The noise penalty is larger in every scene. We changed the target history
and kept the neighbors visible.

*Pause before handing back to the group.*

As we compare the group's models, start with the sample-budget column.
With one forecast, minimum and mean ADE coincide. For the twenty-draw
models, read the best-path scores alongside mean ADE and Energy Score.
