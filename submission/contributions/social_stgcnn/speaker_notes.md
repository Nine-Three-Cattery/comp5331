# Social-STGCNN · Zongmin Zhang

## Frame 1 · Visible pedestrians form a graph at each observed step

The rust dot is the person whose path we want to predict. Suppose one neighbor is a meter away and another is two meters away. The nearer one gets twice the raw edge weight. We then normalize using the degree at both ends of each edge. I build a graph like this for each of the eight observed frames. Every pedestrian visible throughout those frames has a place in the graph. More than twenty-three thousand training windows in the ETH holdout had at least eight neighbors. Preparing the registered 128 targets from stored coordinates took about forty-six milliseconds on our two-thread CPU. The network learns how to turn these eight graph summaries into twelve future displacements. For each target I draw twenty paths, starting all of them at the last input position.

## Frame 2 · Twenty draws reduce best-sample ADE by 0.32 m

The circles on the left show what happens if we can use the true future to pick the best path. The squares average the errors of all twenty paths. Look at ETH and Zara1. Even the best path is about a meter off on ETH, compared with less than half a meter on Zara1. On the right I kept the predictions fixed and changed how many draws were scored. With one draw, average displacement error was 0.94 meters. Picking the best of twenty brought it down to 0.61, while the average draw stayed near 0.94. In a separate test, adding twenty centimeters of Gaussian noise to the observed positions raised mean error by 0.57 meters. Filling two missing observations raised it by about ten centimeters.

## Frame 3 · All twenty paths missed this reversal

The dark line is the observed track. The dashed line is where the pedestrian actually went. The first two examples were chosen before scoring. In the middle panel, the person moved less than a meter from the last observation to the endpoint. Most forecasts went much farther. For the right panel I searched the seed-zero run for the largest error left after taking the best of twenty. The person reversed direction. There were twenty-three visible neighbors, yet all twenty paths continued toward the right. Even the closest endpoint was more than nine meters away. This example shows why I kept the paths as well as the summary scores. Next, Junhao will show how the recurrent Trajectron++ model behaves on the same windows.
