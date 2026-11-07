# Social-GAN design

Gupta et al. describe a stochastic recurrent encoder and decoder, social
pooling, a recurrent discriminator and a best-of-many variety loss. This
implementation adapts those components to Group 7's position-only
`causal_v2` benchmark.

## Input and pooling

The target receives eight metric positions. Its history is centered at
the final observed or causally filled point. Each neighboring track must
contain all eight observations through the same anchor. Pooling uses up
to twelve neighbors within 6 m of the target input endpoint, ordered by
distance then pedestrian ID. It depends on neither neighbor futures nor
eligibility as an evaluation target. Neighbor histories stay clean in
all six conditions. A corrupt target can change the relative geometry
and the neighbor list.

The shared 16-unit LSTM encodes frame displacements after a 16-dimensional
embedding. For each neighbor, an MLP takes its hidden state and its
endpoint offset from the target. Componentwise max pooling reduces the
messages to one vector, with zeros when the target has no neighbors.
A context MLP combines that vector with the target's hidden state.

## Generator and discriminator

The 32-unit LSTM decoder starts from the encoded context concatenated
with an eight-dimensional standard Gaussian draw. At each step it
predicts a displacement correction to the target's average displacement
over its last four input intervals. It feeds the resulting displacement
back into the decoder. A cumulative sum and the target's metric anchor
restore world coordinates. There are no fitted normalization statistics
or scene-dependent coordinate transforms.

A separate LSTM scores the target trajectory through all 20 positions.
Each training batch updates the discriminator once with true futures and
detached generated futures, then updates the generator once. The
generator loss adds a non-saturating adversarial term (weight 0.05) to
the RMS displacement of the closest of twenty candidates. The winning
draw is chosen separately per target using the entire forecast. The
adversarial term uses the first draw without selecting for error.

Training draws are independent by target. Validation and test share
sample indices and Gaussian draws across simultaneous targets in the
same scene and source file. All six test conditions reuse those draws.
The decoder forecasts each target separately from its pooled history.
Aligned samples support the shared proximity diagnostic.

## Validation and departures

Adam uses a learning rate of 0.001, batch size 128 and generator gradient
clipping at norm 5. The formal budget is 20 epochs, stopping after five
epochs without an improvement. The clean validation score orders
checkpoints by Energy Score, mean ADE and then earlier epoch. Test inputs
are constructed after selection. The formal matrix includes all training,
validation and test windows from the pinned manifests.

The paper describes scene-level pooling, a 64-item batch and a 200-epoch
budget. This implementation restricts the visible neighborhood, uses a
constant-velocity residual, applies a target-level discriminator and
uses the smaller budget above. Both training variety draws and test draws
are 20. The group data, checkpoint rule and aggregation differ from
those used for the published headline scores.
