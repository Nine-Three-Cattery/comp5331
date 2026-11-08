# Social-STGCNN · Zongmin Zhang

## Frame 1 · The decoder reads one target from the crowd graph

The rust dot is the person whose path we want to predict. A neighbor one meter away gets twice the raw weight of one two meters away. We then normalize using the degree at both ends of each edge. Everyone visible throughout the eight frames enters the graph. In my version, the decoder sees one weighted sum of positions for the target at each frame. Imagine two neighbors walking toward a stationary person from opposite sides. If they stay equally far away, their contributions cancel. The same thing happens if they walk apart. The authors' decoder keeps a feature vector for every pedestrian. With the smaller decoder, I draw twenty paths from the target's last position. Preparing 128 targets from stored coordinates took about forty-six milliseconds on our two-thread CPU.

## Frame 2 · Twenty draws reduce best-sample ADE by 0.32 m

The circles on the left show what happens if we can use the true future to pick the best path. The squares average the errors of all twenty paths. Look at ETH and Zara1. Even the best path is about a meter off on ETH, compared with less than half a meter on Zara1. On the right I kept the predictions fixed and changed how many draws were scored. With one draw, average displacement error was 0.94 meters. Picking the best of twenty brought it down to 0.61, while the average draw stayed near 0.94. In a separate test, adding twenty centimeters of Gaussian noise to the observed positions raised mean error by 0.57 meters. Filling two missing observations raised it by about ten centimeters.

## Frame 3 · All twenty paths missed this reversal

The dark line is the observed track. The dashed line is where the pedestrian actually went. The first two examples were chosen before scoring. In the middle panel, the person moved less than a meter from the last observation to the endpoint. Most forecasts went much farther. For the right panel I searched the seed-zero run for the largest error left after taking the best of twenty. The person reversed direction. There were twenty-three visible neighbors, yet all twenty paths continued toward the right. Even the closest endpoint was more than nine meters away. This example shows why I kept the paths as well as the summary scores. Next, Junhao will show how the recurrent Trajectron++ model behaves on the same windows.
