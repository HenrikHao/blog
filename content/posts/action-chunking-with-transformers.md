---
title: "Action Chunking with Transformers, piece by piece"
slug: action-chunking-with-transformers
date: 2026-10-13T12:00:00+11:00
draft: false
summary: "ACT is the usual starting point for modern imitation learning. This post walks through action chunking, the architecture, the style variable, training and inference, how attention works inside it, and temporal ensembling."
description: "How ACT works: action chunks, the CVAE style variable, training and inference, what queries, keys and values are, what self-attention and cross-attention do, and temporal ensembling."
tags:
  - robotics
  - imitation-learning
  - transformers
series:
  - Robot Learning
math: true
viz: true
---

I have always wanted to write about technology, and I have always been too lazy
to actually do it :) Lately I have noticed that as work piles up, my memory is
not as reliable as it used to be. Some time after reading a paper or using a
technique, I can no longer recall the details. So I have decided to start
writing them down!

This is the first post in the [Robot Learning](/series/robot-learning/) series,
which works up from imitation learning policies to vision-language-action
models (VLAs).

Action Chunking with Transformers (ACT) is where I want to start, for two
reasons. It is the policy that shipped with ALOHA
([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) and showed that a
roughly $20k teleoperation rig and about fifty demonstrations are enough to
learn fine bimanual tasks such as slotting a battery or opening a condiment cup.
And most of what is in it comes back later: predicting a chunk of future actions
instead of one, a transformer that reads image patches and robot state as a
single sequence, and a latent variable to cope with humans never demonstrating
the same thing the same way twice. Its attention is also small enough to hold in
your head. One encoder, one decoder, no language, no causal masks.

This post covers everything you need to know about ACT, and hopefully it won't
be too long!

## The setup

ALOHA is two 6-DoF ViperX arms with parallel grippers (the followers),
teleoperated by two smaller WidowX arms (the leaders) that a person moves by
hand. Four 480 × 640 RGB webcams watch the scene: one on top, one in front, one
on each wrist. Everything runs at 50 Hz.

A demonstration records what the policy will later see and what it should do.

- **Observation** \(o_t\): the four images and the 14 joint positions of the
  followers (seven per arm: six joints and the gripper).
- **Action** \(a_t\): the 14 joint positions of the *leaders* at that instant,
  which are the targets the followers were tracking.

Using the leader as the action rather than the follower's next position is a
deliberate choice. The gap between where the follower is and where it has been
told to go is what the low-level PID controller turns into force, so the leader
positions carry how hard the human was pushing.

Behaviour cloning then looks like supervised learning: given \(o_t\), predict
\(a_t\). The trouble starts at rollout. A small error moves the robot into a
state the demonstrations never visited, where predictions are worse, which moves
it further off. That is compounding error, and at 50 Hz an episode gives it
hundreds of chances.

## Action chunking

ACT's first idea is to predict the next \(k\) actions in one go. The policy is
\(\pi_\theta(a_{t:t+k} \mid o_t)\) instead of \(\pi_\theta(a_t \mid o_t)\).

{{< viz name="chunking" >}}
With \(k = 1\) the policy makes a decision at every timestep. With chunks it
decides once and commits to \(k\) actions.
{{< /viz >}}

Two things get better.

**The effective horizon shrinks by a factor of \(k\).** A 500-step episode is
five decisions at \(k = 100\). Fewer decisions, fewer chances to compound.

**Pauses stop being a trap.** Human demonstrators hesitate. In the middle of a
pause the observation barely changes while the right action is "keep waiting,
then go". A single-step policy that only sees the current frame cannot represent
that, because nothing in \(o_t\) says how long the pause has lasted. A chunk
predicted before the pause carries straight through it.

In the paper's ablation, averaged over four task settings and without temporal
ensembling, success goes from 1% at \(k = 1\) to 44% at \(k = 100\), then tapers
slightly for larger \(k\). At 50 Hz, \(k = 100\) is two seconds of motion.

The price is reactivity: executed naively, the robot runs open loop for \(k\)
steps. Temporal ensembling, at the end of this post, is the paper's answer.

## The architecture

{{< viz name="architecture" >}}
ACT during training and at inference. Shapes are for ALOHA: four cameras and
14 joints. Switch modes to see what is dropped at test time.
{{< /viz >}}

Let's start with the whole architecture, then go through each part and its
dimensions:

1. **Images to tokens.** Each camera image goes through the same ResNet-18,
   which downsamples by 32 and gives a 15 × 20 feature map with 512 channels.
   A 1 × 1 convolution projects it to the model width, a fixed 2D sinusoidal
   position embedding marks where each cell came from, and the map is flattened
   into 300 tokens. Four cameras make 1200.
2. **Two more tokens.** The joint positions are projected from 14 to 512
   dimensions and become one token. The style variable \(z\) is projected from
   32 to 512 and becomes another. That makes 1202.
3. **Transformer encoder.** Four layers of self-attention over all 1202 tokens.
   This is where the cameras, the joints and \(z\) get to look at each other. I
   will call its output the *memory*.
4. **Transformer decoder.** The input is \(k\) *action queries*, one per future
   timestep. They cross-attend to the memory, and that is their only source of
   information about the world.
5. **Action head.** A linear layer maps each of the \(k\) decoder outputs to 14
   joint targets.

| Stage | Shape |
| --- | --- |
| Camera images | 4 × 3 × 480 × 640 |
| ResNet-18 feature map, per camera | 512 × 15 × 20 |
| Image tokens | 1200 × 512 |
| Plus joints token and \(z\) token | 1202 × 512 |
| Encoder output (memory) | 1202 × 512 |
| Action queries | \(k\) × 512 |
| Predicted chunk | \(k\) × 14 |

The model has about 80M parameters, trains from scratch for each task in about
five hours on a single 11 GB RTX 2080 Ti, and takes about 10 ms per forward
pass. (That is super fast compared to VLAs!)

Wherever this post gets down to code, it follows
[LeRobot's implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act),
which is also what our own checkpoints are trained with.

## The style variable *z*

### Why have a latent at all

Ask two people to pass a cup from one gripper to the other and you get two
trajectories. Ask one person twice and you still get two. For the same
observation there are many correct chunks, and a network trained with a
regression loss settles the disagreement by averaging. The average of two good
ways around an obstacle goes through it.

ACT gives the policy one extra input, \(z\), whose job during training is to
say *which* of the valid chunks this particular demonstration was. While \(z\)
carries that, the policy is not being asked to hit two different targets from
the same input, so it does not have to hedge between them. The paper calls
\(z\) the style variable.

### How it is trained

The construction is a conditional variational autoencoder (CVAE), with the
policy playing the decoder.

The **CVAE encoder** only exists during training. It is a four-layer
transformer encoder whose input is a learned `[CLS]` token, the joint positions
and the \(k\) demonstrated actions, each projected to 512 dimensions, so
\(k + 2\) tokens. Its output at the `[CLS]` position is projected to the mean
and log-variance of a diagonal Gaussian over \(z\), which has 32 dimensions by
default (`latent_dim`). It does not see the images. The paper leaves them out to keep
training fast.

\(z\) is then sampled with the reparameterisation trick,
\(z = \mu + \sigma \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\),
so gradients reach the encoder. The policy takes the observation and this \(z\)
and predicts the chunk. The loss has two terms:

$$
\mathcal{L} =
\underbrace{\lVert a_{t:t+k} - \hat{a}_{t:t+k} \rVert_1}_{\text{reconstruction}}
\;+\;
\beta \,
\underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid a_{t:t+k}, \bar{o}_t) \,\Vert\, \mathcal{N}(0, I)\big)}_{\text{regularisation}}
$$

where \(\bar{o}_t\) is the observation without images and \(\beta = 10\). The
training section below goes through both terms in detail.

The two terms want opposite things. The reconstruction term wants \(z\) to say
as much as possible about the chunk. The KL term wants it to say nothing.

A picture helps here. The policy is a student sitting an exam, and the
observation is the exam paper. The CVAE encoder is a friend who has seen the
answer sheet and is allowed to pass the student a note. The note is \(z\).

If the note can be as long as it likes, the friend copies out the whole answer.
The student never has to read the paper, gets full marks in every practice
exam, and learns nothing. That is the shortcut the encoder invites. It has seen
the demonstrated chunk, so it can pack all of it into \(z\), and the policy can
stop looking at the cameras. On the real exam there is no friend and no note,
and the student is lost.

The KL term makes the friend pay for every word. The cheapest note is the
standard blank one, the same whatever the answer is. It costs nothing and tells
the student nothing, and it corresponds to a posterior equal to the prior. To
say something about *this* answer, the note has to differ from the blank one,
by shifting the mean of \(z\) or narrowing its spread, and the further it moves
the more it costs.

At that price the student works out everything the exam paper allows, because
that is free. The note keeps only what is worth paying for: things that cost
marks if missed and that the paper does not give away. In a robot
demonstration, those are things like how fast this attempt went and where the
demonstrator hesitated. That is the style.

### What happens at test time

There is no demonstration at test time, so there is no encoder. \(z\) is set to
zero, the mean of the prior, and the policy becomes a deterministic function of
the observation. In the exam picture, the friend is gone and the note is blank.

So ACT never samples when it runs. The latent earns its keep during training,
by soaking up variation so that the policy can learn sharp predictions instead
of blurred ones. The ablation backs this up. On scripted demonstrations, which
have one style, removing the CVAE objective changes almost nothing. On human
demonstrations, success drops from 35.3% to 2%.

### Does *z* = 0 bring the average back?

It is the obvious objection. If going left and going right both work, and the
encoder has sorted the demonstrations into "left" and "right" along \(z\), then
\(z = 0\) sits between the two groups. Has the average not just come back?

It helps to be exact about what \(z\) is. It does not hold two trajectories.
The policy is a single function of the observation and \(z\). During training,
a \(z\) from the left-hand group arrives together with a left-going target and
a \(z\) from the right-hand group with a right-going one, so the function
learns to output left for some values of \(z\) and right for others. At test
time it is evaluated at one point, \(z = 0\), and returns one chunk. Nothing in
the method promises that this chunk is a good one. What it turns out to be
depends on three things.

**The loss is L1, and L1 does not average.** Around \(z = 0\) the two groups
overlap, so the policy is trained there on a mixture of left and right targets.
Under squared error the best single answer to a mixture is its mean, which is
the path through the obstacle. Under absolute error it is the median, and the
median of "70% left, 30% right" is left, not a point 30% of the way across. A
tie is the one case that stays undecided.

**Only the first decision is ambiguous.** Once the arm has started round the
left side, the observation says so, and from there every demonstration in the
data continues to the left. Chunking helps as well, since one decision commits
the next \(k\) steps.

**The latent may not be sorting strategies at all.** The KL term makes
information in \(z\) expensive. What it can afford to carry is more plausibly
continuous variation such as speed and hesitation, where the middle setting is
still a perfectly good trajectory. That is what "style" suggests, and it is a
better description of \(z\) than "which way round".

So at test time ACT is a deterministic policy that produces one chunk per
observation. It can settle on the more common of two strategies. It cannot
represent "either", and when two strategies are evenly balanced nothing stops
it from landing between them. Policies that sample at test time, such as
diffusion and flow-matching policies, remove exactly this limitation.

### Is *z* doing anything in your model?

A check worth running on your own checkpoint: hold the observation fixed, draw
a few \(z \sim \mathcal{N}(0, I)\), and compare the predicted chunks. If they
fan out, you are looking at the variety in your demonstrations. If they are
identical, the posterior has collapsed onto the prior and the policy has
learned to ignore \(z\).

I ran this on one of our own checkpoints: a LeRobot ACT trained for 100k steps
on 30 teleoperated episodes of a 7-DoF arm putting tennis balls into a tray,
with a single camera on the wrist. Sampling \(z\) from the prior moved the
predicted actions by 0.0003 standard deviations, which is nothing. That policy
ignores its style variable. This does not show the CVAE objective was wasted on
it, because the paper's ablation is about what the objective does during
training. It does mean that at inference this particular model is ACT with the
latent deleted. The loss walk-through below shows the same thing from the
training side.

## Training and inference

### Training

LeRobot treats every frame of every episode as one training sample. For frame
\(t\) the dataset returns the images and joint positions at \(t\) and, because
ACT's config sets `action_delta_indices = list(range(chunk_size))`, the actions
at \(t, t+1, \dots, t+k-1\). When the chunk runs past the end of the episode,
the missing positions are filled by repeating the last action, and a boolean
`action_is_pad` marks them so that the loss can leave them out. Joint positions
and actions are standardised with dataset statistics. Images are scaled to
\([0, 1]\) and normalised with the ImageNet mean and standard deviation,
because the ResNet starts from ImageNet weights.

The training step is `ACTPolicy.forward`. Trimmed to its essentials:

```python
def forward(self, batch):
    # batch["observation.state"]     (B, 14)
    # batch["observation.images.*"]  (B, 3, 480, 640), one entry per camera
    # batch["action"]                (B, k, 14)
    # batch["action_is_pad"]         (B, k), True past the end of the episode
    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

    abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
    valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
    num_valid = valid_mask.sum() * abs_err.shape[-1]
    l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

    loss_dict = {"l1_loss": l1_loss.item()}

    mean_kld = (
        (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
    )
    loss_dict["kld_loss"] = mean_kld.item()
    loss = l1_loss + mean_kld * self.config.kl_weight

    return loss, loss_dict
```

Everything else happens inside `self.model(batch)`. In training mode the model
first runs the CVAE encoder on the joints and the demonstrated chunk to get
`mu` and `log_sigma_x2`, which is \(\ln \sigma^2\), then samples

```python
latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
```

and runs the policy on the observation and that sample. In evaluation mode the
encoder is skipped and `latent_sample` is all zeros.

### The loss, term by term

The `loss = ...` line is the whole objective. For one training sample:

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \beta \, \mathcal{L}_{\text{KL}}
$$

This is the standard VAE bargain. The first term rewards the policy for
rebuilding the demonstrated chunk. The second limits how much help it may get
from an encoder that has seen the answer. \(\beta\) sets the price.

#### Reconstruction

$$
\mathcal{L}_{\text{rec}} = \frac{1}{|V| \, D} \sum_{i \in V} \sum_{d=1}^{D}
\left| a_{t+i,\,d} - \hat{a}_{t+i,\,d} \right|
$$

\(V\) is the set of chunk positions that fall inside the episode and \(D\) is
the action dimension, 14 on ALOHA. Three details matter here.

**The units are standard deviations.** Targets and predictions are both
normalised, so every joint's error is measured against how much that joint
moves in the dataset. In our data the gripper command has a standard deviation
of 1.6 and the wrist roll 0.13. Without normalisation the gripper would count
twelve times as much.

**It is L1, not L2.** The paper's reason is empirical: L1 gave more precise
action sequences. Two general properties point the same way. Squared error is
minimised by the mean of the plausible answers and absolute error by their
median, which is less inclined to land between two modes. And the gradient of
an absolute error has the same size however small the error is, so the loss
keeps pushing on the last few percent.

**Padding is left out of both the sum and the count.** A chunk that starts
near the end of an episode runs out of real actions. `valid_mask` zeroes those
positions and `num_valid` counts only the real ones, so a sample with ten real
steps is averaged over ten steps and not over a hundred. In the episode shown
below, 99 of the 747 samples are padded to some degree.

{{< viz name="loss-reconstruction" >}}
The reconstruction term on real training samples: one episode replayed through
our trained checkpoint. The lower panel is what gets averaged. Drag the sample
slider to the right-hand end to reach chunks that run off the episode.
{{< /viz >}}

#### Regularisation

The encoder outputs a mean \(\mu_j\) and a log-variance \(\ln \sigma_j^2\) for
each of the 32 latent dimensions. The KL divergence between that Gaussian and
the standard normal prior has a closed form, so nothing needs to be sampled to
compute it:

$$
\mathcal{L}_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{32}
\left( \mu_j^2 + \sigma_j^2 - \ln \sigma_j^2 - 1 \right)
$$

Each dimension pays two costs: \(\mu_j^2 / 2\) for moving its mean away from
zero, and \((\sigma_j^2 - \ln \sigma_j^2 - 1) / 2\) for having a width other
than one. Both are zero only when that dimension of the posterior *is* the
prior, and then it tells the policy nothing about the chunk. The total is
measured in nats, and averaged over the data it is an upper bound on the
information \(z\) can carry about the demonstration.

The `mean_kld` line in the code is the same expression with the sign folded
in, where `log_sigma_x2_hat` is \(\ln \sigma^2\). `.sum(-1)` adds up the 32
dimensions and `.mean()` averages over the batch.

{{< viz name="loss-kl" >}}
One latent dimension. Move the posterior away from the prior to see what each
kind of deviation costs. *Our checkpoint* sets it to what the trained encoder
outputs.
{{< /viz >}}

#### The exchange rate

The two terms are not on the same scale, and the code does nothing to hide it.
The reconstruction term is an *average* over every entry of the chunk, 1400
numbers on ALOHA. The KL term is a *sum* over 32 dimensions, multiplied by
\(\beta = 10\). So one nat of information in \(z\) costs the same as being
wrong by ten standard deviations on every joint at every step. For \(z\) to be
worth using, each nat has to buy more than that.

Here is where our checkpoint ended up, evaluated over one of its training
episodes with dropout off:

| Term | Value |
| --- | --- |
| Reconstruction, \(z\) sampled from the posterior | 0.0316 |
| Reconstruction, \(z = 0\) | 0.0316 |
| KL, summed over 32 dimensions | 0.00004 nats |
| \(\beta \times\) KL | 0.0004 |
| Total | 0.0320 |

The posterior has collapsed onto the prior. In the exam picture, the note came
back blank. Whatever chunk the encoder is shown, its means stay within 0.004 of
zero and its standard deviations within 0.004 of one. That is why sampling \(z\) changed nothing earlier, and the
arithmetic says the model had little choice by the end. With a total
reconstruction error of 0.03, a \(z\) that removed *all* of it could afford to
cost at most 0.003 nats. Whether \(z\) carried anything earlier in training,
when the error was larger, the final checkpoint cannot tell us.

Two practical consequences. Log the two terms separately and not only their
sum; LeRobot reports them as `l1_loss` and `kld_loss`. If the KL term sits at
zero, the model is running without a style variable, which may be fine but is
worth knowing. And if you want \(z\) to carry more, \(\beta\) is the dial.
Lowering it lets \(z\) say more, at the risk that the policy starts to rely on
information it will not have at test time.

### Optimiser and schedule

LeRobot's defaults for ACT are AdamW with a learning rate of \(10^{-5}\) for
the transformer and the backbone alike, weight decay \(10^{-4}\) and no
learning-rate schedule, on a model with dropout 0.1, hidden size 512,
feed-forward size 3200 and 8 heads. Its training script defaults to batch size
8 for 100,000 steps. Our checkpoint used batch size 64 for the same number of
steps.

One practical note from the ACT authors that matches what people report: if
the policy is jerky or stalls mid-episode, train for longer. Smoothness and
success keep improving well after the loss curve has flattened.

### Inference

On the robot, the loop only ever asks for one action:

```python
policy.reset()                                    # empties the action queue at the start of an episode

while not done:
    action = policy.select_action(observation)    # one action per call
    observation = robot.step(action)
```

The chunking is inside `select_action`. Trimmed:

```python
def select_action(self, batch):
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
        self._action_queue.extend(actions.transpose(0, 1))
    return self._action_queue.popleft()
```

`predict_action_chunk` is one forward pass in evaluation mode, so \(z\) is
zero. `n_action_steps` defaults to the whole chunk of 100. That makes this the
open-loop version: predict \(k\) actions, hand them out one at a time, and
look at the cameras again only when the queue is empty. On every call in
between, the observation that is passed in goes unused. Normalising the
observation and un-normalising the action happen in LeRobot's pre- and
post-processors, outside the policy. The last section replaces the queue with
something better.

## Attention, step by step

Everything that makes this model a transformer happens inside one operation,
so it is worth slowing down for. Attention is how one token gathers information
from a set of other tokens, deciding for itself how much to take from each.

### A lookup that never says no

Think of a dictionary in code. You hand it a query, it finds the one entry
whose key matches, and it gives back that entry's value. Attention is the same
lookup made soft. Every entry still has a key and a value, but the query is
compared with *all* the keys, each comparison produces a score, and what comes
back is a blend of all the values in proportion to their scores. A good match
contributes a lot and a poor one almost nothing.

In a transformer the entries are tokens, and all three ingredients are vectors:

- a **query** belongs to the token doing the reading: *what am I looking for?*
- a **key** belongs to each token that can be read: *what should I be found
  by?*
- a **value** also belongs to each token that can be read: *what do I hand over
  when I am found?*

Keys and values are kept separate on purpose, because how a token is found and
what it gives need not be the same thing. An image patch showing the gripper
could be found by any query that amounts to "where is the hand?", and what it
hands over could be the exact pose it sees.

The figure follows a single query through the arithmetic as it reads from five
tokens: \(z\), the joints, and three image patches.

{{< viz name="attention" >}}
Scaled dot-product attention for one query. The vectors are invented and
two-dimensional so that "the query matches this key" is visible as "they point
the same way". Drag the orange arrowhead, or use the slider.
{{< /viz >}}

In matrix form, with one row of \(Q\) per query and one row of \(K\) and \(V\)
per token:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Each row of the softmax is one query's distribution over the tokens it can
read. Stack the rows for all the queries and you have the attention map.

### Where queries, keys and values come from

None of the three is stored anywhere. A token is a single vector \(x\), 512
numbers in ACT, and every attention layer owns three learned matrices that turn
it into its three roles:

$$
q = W_Q \, x \qquad k = W_K \, x \qquad v = W_V \, x
$$

So every token can play all three parts, and which part it plays depends on
which side of the lookup it is on. Training shapes the matrices, which means
the model itself decides what tokens look for, what they are found by and what
they pass on.

The whole operation also runs several times in parallel, in *heads*, each with
its own slice of those matrices. ACT uses 8 heads on a 512-dimensional model,
so each head works in \(d_k = 64\) dimensions, and their outputs are
concatenated and passed through one more linear layer. Different heads can look
for different things at the same time.

### Self-attention: the tokens read each other

In self-attention, the tokens that ask and the tokens that are read are the
same set. Each token takes a turn as the query, reads from all of them, itself
included, and adds what it collected to its own vector. One pass turns tokens
that each know only about themselves into tokens that each know something about
the others.

{{< viz name="attention-flow" variant="self" >}}
Self-attention over six tokens, with toy weights. Click any token on the left
to make it the one asking, or press *Play all* to give every token its turn and
fill in the whole map.
{{< /viz >}}

Two things to notice. Every token appears on both sides: it sends a query, and
it answers everyone else's queries with its key and value. And the grid on the
right is the attention map from before, one row per asking token, so for \(N\)
tokens it is \(N \times N\).

In ACT this is what the policy encoder does, four times over, on 1202 tokens.
It is where a patch from the wrist camera can find out what the top camera
sees, where image tokens pick up the joint angles, and where every token hears
from \(z\). After four rounds a token is no longer just "this patch". It is
"this patch, in context". The CVAE encoder is the same operation over `[CLS]`,
the joints and the \(k\) demonstrated actions, where `[CLS]` reads the whole
chunk and becomes its summary.

### Cross-attention: the action queries read the memory

The arithmetic does not change. What changes is the cast: the queries come from
one set of tokens and the keys and values from another. In ACT's decoder the
tokens that ask are the \(k\) action queries, and the set being read is the
encoder's output, the memory.

{{< viz name="attention-flow" variant="cross" >}}
Cross-attention from four action queries to six memory tokens, with toy
weights. Click an action query, or press *Play all*.
{{< /viz >}}

Three things are different from self-attention.

- **The memory is only read.** Its tokens supply keys and values and come out
  unchanged. Only the action queries are updated.
- **The map is not square.** It has one row per action query and one column per
  memory token, so \(k \times 1202\) in ACT.
- **It is the only bridge.** Everything an action query knows about the cameras
  and the joints arrives through this step. What it collects then passes
  through a small feed-forward network and the linear head, and comes out as
  one action.

### One training sample, as matrices

Drawn as lines between tokens, attention looks like a different kind of machine
from the networks that came before it. A convolution wires each output to a
small fixed neighbourhood. A recurrent network hands its state along one step
at a time. Attention seems to rewire itself for every input, with every token
connected to every other.

Underneath, it is ordinary matrix multiplication. Stack the tokens as the rows
of a matrix \(X\), and one attention layer is four lines:

$$
\begin{aligned}
Q &= X W_Q \qquad K = X W_K \qquad V = X W_V \\
S &= Q K^\top / \sqrt{d_k} \\
A &= \operatorname{softmax}(S) \\
\text{out} &= A V
\end{aligned}
$$

The connections between tokens are the entries of \(A\), and \(A\) is computed
from the data. That is the real difference from a CNN or an RNN, whose wiring
is fixed in advance. But computing it needs no loop over tokens. Every row of
every matrix is filled in at once, which is why transformers suit GPUs so well,
where an RNN has to finish step \(t - 1\) before it can start step \(t\).

The figure takes one training sample through a toy version of the model, one
step at a time: forward through the encoder's self-attention and the decoder's
cross-attention to the loss, then backward to the weights. The numbers are
computed in your browser from tiny matrices, with three numbers per token where
ACT has 512.

{{< viz name="training-step" >}}
One training sample through a toy encoder and decoder with single-head
attention. *Next* walks forward to the loss and then backward to the weight
update. *Apply update* keeps the new weights, so you can run the sample again
and watch the loss fall. Hover a cell of any product to see what it was made
from.
{{< /viz >}}

Three things are worth taking from the backward half.

- **Backward is more of the same.** Every gradient is a product with a
  transposed matrix from the forward pass. Whatever made the forward pass
  parallel makes the backward pass parallel too.
- **Two paths, two lessons.** The gradient reaches the value matrix directly
  through the weights \(A\), and it teaches *what to pass on*. It reaches the
  query and key matrices only through the softmax, and it teaches *where to
  look*.
- **Attention routes the gradient as well as the signal.** A token that was
  read heavily on the way forward gets a large share of the gradient on the way
  back. A token that was ignored learns almost nothing from this sample.

### Where attention sits in ACT

| Layer | Who is asking | Who can be read | Map size |
| --- | --- | --- | --- |
| CVAE encoder, self-attention (training only) | `[CLS]`, joints, \(k\) actions | the same \(k + 2\) tokens | \((k+2) \times (k+2)\) |
| Policy encoder, self-attention | 1202 observation tokens | the same 1202 tokens | 1202 × 1202 |
| Decoder, self-attention | \(k\) action queries | the same \(k\) queries | \(k \times k\) |
| Decoder, cross-attention | \(k\) action queries | 1202 memory tokens | \(k \times 1202\) |

Around every attention step there is a residual connection, a normalisation and
a small feed-forward network. Those work on each token separately. Attention is
the only place where tokens exchange information.

Three things about ACT's version are simpler than what comes later in this
series.

**No masks.** Nothing is autoregressive. All \(k\) actions come out of a single
forward pass in parallel, which is how it fits in 10 ms.

**Positions go into queries and keys, not values.** Following DETR, the
position embeddings are added to the queries and keys of each attention layer
and not to the values. Position helps decide where to look. It is not mixed
into what gets carried away.

**The action queries never talk to each other.** This one comes from reading
the code, not the paper. The decoder's input starts as zeros, with the query
embeddings entering only as positions. In the decoder's self-attention the
values are therefore all zero, and it returns the same constant vector for
every query. With LeRobot's single decoder layer that is the whole decoder, so
each action is decoded from its own query embedding and the shared memory, with
no communication between the \(k\) timesteps. Whatever makes a chunk coherent
has to come from the memory they all read and from being trained on smooth
chunks.

## Temporal ensembling

Back to the open-loop problem. Predicting \(k\) actions and running all of them
has two flaws: for \(k\) steps the robot ignores its cameras, and at each chunk
boundary the plan can jump, because two consecutive chunks never agree exactly.
When I replay a recorded episode through our checkpoint, the first action of a
new chunk differs from the last action of the chunk before it by 0.044
standard deviations on average, about three times an ordinary step between
consecutive actions in the demonstration.

Temporal ensembling queries the policy at *every* step instead. Each timestep
\(t\) then has up to \(k\) predictions, made at steps \(t, t-1, \dots, t-k+1\).
The robot executes their weighted average:

$$
a_t = \frac{\sum_i w_i \, \hat{a}_t^{(i)}}{\sum_i w_i},
\qquad w_i = \exp(-m \cdot i)
$$

where \(i = 0\) is the *oldest* prediction for that step.

{{< viz name="temporal-ensemble" >}}
A toy one-joint rollout. Every forward pass predicts the ideal trajectory plus
its own random error, so consecutive chunks disagree slightly, as real ones do.
In the lower panel, orange rows are the only passes that open-loop execution
makes, and blue marks the chunks voting on the current step. Drag across the
chart to move \(t\).
{{< /viz >}}

Three things to take from the figure.

- **Old predictions weigh more.** With \(m > 0\) the oldest prediction gets the
  largest weight, which is the opposite of what most people expect. LeRobot
  suggests \(m = 0.01\), which with \(k = 100\) gives weights from 1 down to
  about 0.37.
- **It is smooth, and slow to change its mind.** Averaging removes the jumps at
  chunk boundaries. It also means that predictions made before something
  changed keep voting for another \(k - 1\) steps. Leave "move the target" on
  and watch the blue line after step 40.
- **It costs \(k\) times the compute.** One forward pass per control step,
  where open-loop execution needs one per chunk.

In LeRobot this is a configuration change: set `temporal_ensemble_coeff`, which
is \(m\), together with `n_action_steps = 1`, which it requires. It is off by
default. In the paper's ablation it is worth 3.3 points of success rate for
ACT, which is real but modest next to chunking itself. The cheaper middle
ground is to execute only the first \(n < k\) actions of each chunk and then
re-plan.

## Summary: the good and the bad

### What ACT does well

- **It is small and fast.** Tens of millions of parameters, a few hours of
  training on one consumer GPU, and about 10 ms per forward pass. Next to a VLA
  that is tiny.
- **It needs little data.** About fifty demonstrations per task in the paper,
  and thirty episodes for the checkpoint used in this post.
- **Chunks make it smooth and precise.** Committing to \(k\) actions at a time
  cuts the number of decisions and carries the policy through short
  hesitations. It is the main reason ACT can do fine manipulation at 50 Hz.
- **It is simple.** One encoder, one decoder, one forward pass, and no sampling
  at test time. The same observation always gives the same chunk, which makes
  it easy to debug.

### Where it falls short

Both of the limitations below come from the same place. ACT decides from a
single frame. It remembers nothing of what it saw or did before, apart from the
chunk it is in the middle of executing.

- **Tasks that involve waiting.** Chunking handles a hesitation that fits
  inside one chunk. A wait that lasts longer than a chunk is a different
  matter. While the robot waits, every frame looks the same, so nothing in the
  observation says how long it has been waiting, and the policy cannot tell
  when it is time to move on.
- **Symmetric tasks, where two phases look alike.** If the scene looks the same
  on the way out and on the way back, the current frame does not say which
  phase the task is in. ACT is then asked for two opposite actions from the
  same input, and it has nothing to choose between them with.

## References

- Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.
  [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705).
  RSS 2023.
- LeRobot's ACT implementation:
  [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act).
- Nicolas Carion et al.
  [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
  (DETR), the source of the encoder-decoder and its queries.
- Ashish Vaswani et al.
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Diederik P. Kingma, Max Welling.
  [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
