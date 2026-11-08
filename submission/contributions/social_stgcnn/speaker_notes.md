# Social-STGCNN · Zongmin Zhang

## Frame 1 · The decoder keeps all pedestrians in the scene

Each arrow shows how far someone moved in one observed interval. Suppose the target and the person on the right both step east by a quarter of a meter. Their displacements match, so their direct edge has zero weight in the authors' released graph. The third person takes the same step west. The two displacement vectors are now half a meter apart, giving a raw edge weight of two per meter. After graph filtering, the temporal decoder still has a feature vector for everyone. It predicts distributions for twelve future steps, and only then do I select our target. I checked this calculation against the authors' code, including a recorded ETH scene with three pedestrians.

## Frame 2 · Best-sample ADE falls from 0.971 to 0.583 m

The right-hand chart scores the same forecasts four times. I start with the first path, then include the first five, the first ten, and finally all twenty. For each path, I measure its distance from the true position at each of the twelve future steps and take the average. The green circles pick the path with the smallest error. That error falls from about ninety-seven centimeters with one draw to fifty-eight with twenty. The orange squares give every draw equal weight and stay near ninety-seven centimeters throughout. We can pick the best path only once we know where the person went. On the left, even with twenty draws, best-sample error ranges from forty-one centimeters on Zara two to ninety-two on ETH.

## Frame 3 · All twenty forecasts overshot the target in one ETH window

Look at the black dot in the right panel. By this last observation, the pedestrian had moved six and a half meters to the right. Every green forecast ends more than eight and a half meters farther right. The dashed line stays close to the dot. After twelve steps, the person was only a little over a meter from that last observation. Even the closest forecast endpoint missed by eight point four meters. I picked this window because it had the highest best-of-twenty average error across all five scenes at seed zero. The other two examples were fixed before scoring. In the middle University window, the closest endpoint was within half a meter. Junhao will now show how Trajectron++ forecasts these same pedestrian windows.
