import { useState, useEffect, useRef, useCallback } from "react";

const MODULES = [
  {
    id: 1,
    title: "Search",
    icon: "🔍",
    color: "#E8483F",
    sections: [
      {
        title: "What is Search?",
        content: `Search is about finding a path from a **start state** to a **goal state** through a space of possibilities. Think of it like navigating a city — you know where you are, where you want to go, and you need to find the best route.

**Key components of a search problem:**
• **Initial State** — where you start
• **Actions** — what moves you can make
• **Transition Model** — what happens when you take an action
• **Goal Test** — are we there yet?
• **Path Cost** — how expensive is this route?`,
        viz: "tree-search"
      },
      {
        title: "Breadth-First Search (BFS)",
        content: `BFS explores nodes level by level, like ripples spreading out from a stone dropped in water. It uses a **queue** (FIFO) to track which nodes to visit next.

**Properties:**
• **Complete** — always finds a solution if one exists
• **Optimal** — finds shallowest solution (cheapest if all edges cost the same)
• **Time & Space** — O(b^d) where b = branching factor, d = depth

BFS is great when all step costs are equal and the solution is near the root.`,
        viz: "bfs"
      },
      {
        title: "Uniform Cost Search (UCS)",
        content: `UCS is like BFS but for weighted graphs. Instead of expanding the shallowest node, it expands the **cheapest** node first using a **priority queue**.

**Key insight:** UCS always expands the node with the lowest total path cost g(n). It's guaranteed to find the optimal solution.

**Think of it like:** "Always follow the cheapest unexplored path so far."

UCS is equivalent to Dijkstra's algorithm.`,
        viz: "ucs"
      },
      {
        title: "A* Search",
        content: `A* combines the best of UCS and greedy search. It uses an **evaluation function**:

**f(n) = g(n) + h(n)**

• **g(n)** = actual cost from start to node n
• **h(n)** = estimated cost from n to goal (heuristic)

**Admissible heuristic:** never overestimates the true cost (optimistic). This guarantees A* finds the optimal solution.

**Consistent heuristic:** h(n) ≤ cost(n→n') + h(n'). This means the heuristic obeys the triangle inequality.

A* is optimally efficient — no other optimal algorithm expands fewer nodes.`,
        viz: "astar"
      }
    ]
  },
  {
    id: 2,
    title: "Simulated Annealing",
    icon: "🌡️",
    color: "#E07B39",
    sections: [
      {
        title: "Local Search & Hill Climbing",
        content: `Sometimes we don't care about the path — we just want the **best solution**. Local search starts with a candidate and iteratively improves it.

**Hill Climbing** is the simplest approach: always move to a better neighbor. Like climbing a mountain in fog — always go uphill.

**Problems with hill climbing:**
• **Local maxima** — stuck on a small hill, can't see the mountain
• **Plateaus** — flat areas where no neighbor is better
• **Ridges** — narrow peaks that are hard to navigate`,
        viz: "hill-climbing"
      },
      {
        title: "Simulated Annealing",
        content: `Inspired by metallurgy! When metal cools slowly (annealing), atoms find low-energy arrangements.

**The algorithm:**
1. Start at a random solution with high "temperature" T
2. Pick a random neighbor
3. If better → always accept
4. If worse → accept with probability e^(ΔE/T)
5. Gradually reduce T (cooling schedule)

**Key insight:** At high temperature, we explore freely (accept bad moves). As temperature drops, we become pickier and converge to a good solution.

**Guarantee:** With a slow enough cooling schedule, SA will find the global optimum!`,
        viz: "annealing"
      },
      {
        title: "Genetic Algorithms",
        content: `Inspired by biological evolution! Maintain a **population** of candidate solutions that evolve over generations.

**Steps each generation:**
1. **Selection** — choose fitter individuals to reproduce
2. **Crossover** — combine parts of two parents to create children
3. **Mutation** — randomly modify some children

**Example — N-Queens:** Represent each solution as a string of column positions. Crossover swaps sections between two parent boards. Mutation moves a random queen.

GAs are good for large, complex search spaces where the structure of good solutions isn't well understood.`,
        viz: "genetic"
      }
    ]
  },
  {
    id: 3,
    title: "Game Playing",
    icon: "♟️",
    color: "#4A90D9",
    sections: [
      {
        title: "Minimax Algorithm",
        content: `In two-player zero-sum games, what's good for me is bad for you. **Minimax** assumes both players play optimally.

**The idea:**
• **MAX player** (you) tries to maximize the score
• **MIN player** (opponent) tries to minimize the score
• Build a game tree, propagate values bottom-up

At MAX nodes: choose the child with **highest** value
At MIN nodes: choose the child with **lowest** value

**Time complexity:** O(b^m) where b = branching factor, m = max depth`,
        viz: "minimax"
      },
      {
        title: "Alpha-Beta Pruning",
        content: `Alpha-Beta is Minimax made smart. It skips branches that **can't possibly influence** the final decision.

**Two values tracked:**
• **α (alpha)** = best value MAX can guarantee (starts at -∞)
• **β (beta)** = best value MIN can guarantee (starts at +∞)

**Pruning rule:** If α ≥ β, stop exploring that branch — it's irrelevant.

**Best case:** With perfect move ordering, Alpha-Beta examines only O(b^(m/2)) nodes — effectively doubling the search depth!`,
        viz: "alpha-beta"
      },
      {
        title: "Advanced Game Concepts",
        content: `**Evaluation Functions:** When we can't search to the end, we estimate position quality. Good eval functions capture key features (material, mobility, position).

**Iterative Deepening:** Search deeper and deeper until time runs out. Always have a "best move so far" ready.

**Quiescent Search:** Don't evaluate "noisy" positions (e.g., mid-capture in chess). Extend search until the position is quiet.

**Horizon Effect:** Limited depth can cause the AI to delay inevitable bad outcomes by pushing them beyond the search horizon.

**Expectimax:** For games with chance (dice, cards), add "chance nodes" that average over possible outcomes.`,
        viz: null
      }
    ]
  },
  {
    id: 4,
    title: "Constraint Satisfaction",
    icon: "🧩",
    color: "#7B61C1",
    sections: [
      {
        title: "CSP Fundamentals",
        content: `A Constraint Satisfaction Problem has:
• **Variables** — things to assign values to (e.g., regions on a map)
• **Domains** — possible values for each variable (e.g., {Red, Green, Blue})
• **Constraints** — rules about which combinations are allowed

**Classic example — Map Coloring:** Color a map so no two adjacent regions share a color.

CSPs are everywhere: scheduling, Sudoku, circuit layout, resource allocation.

**Constraint Graph:** Variables are nodes, edges connect variables that share a constraint.`,
        viz: "csp-map"
      },
      {
        title: "Solving CSPs",
        content: `**Backtracking Search:** Try assigning values one variable at a time. If a constraint is violated, backtrack and try a different value.

**Smart improvements:**
• **Forward Checking** — after assigning a variable, remove inconsistent values from neighbors' domains. Detect failure early!
• **Arc Consistency (AC-3)** — for every pair of constrained variables, ensure every value in one domain has a compatible value in the other
• **MRV (Minimum Remaining Values)** — choose the variable with fewest legal values next ("most constrained first")
• **LCV (Least Constraining Value)** — try the value that rules out the fewest options for neighbors

These heuristics can turn exponential problems into near-linear ones!`,
        viz: "backtracking"
      }
    ]
  },
  {
    id: 5,
    title: "Probability",
    icon: "🎲",
    color: "#2EAF7D",
    sections: [
      {
        title: "Probability Basics",
        content: `Probability quantifies uncertainty. For an event A:

**P(A)** is between 0 (impossible) and 1 (certain)
**P(¬A) = 1 - P(A)** (complement)

**Joint Probability:** P(A ∧ B) — probability both A and B happen
**Conditional:** P(A|B) = P(A ∧ B) / P(B) — probability of A given B is true

**Independence:** A and B are independent if P(A|B) = P(A), meaning knowing B tells you nothing about A.

**Total Probability:** P(A) = Σ P(A|B=b) × P(B=b) — sum over all possible values of B`,
        viz: "probability"
      },
      {
        title: "Bayes' Rule",
        content: `The most important formula in AI:

**P(A|B) = P(B|A) × P(A) / P(B)**

This lets us flip conditional probabilities!

**Cancer screening example:**
• P(Cancer) = 0.01 (1% base rate)
• P(Positive | Cancer) = 0.9 (test sensitivity)
• P(Positive | ¬Cancer) = 0.2 (false positive rate)

What's P(Cancer | Positive)?

Using Bayes: P(C|+) = (0.9 × 0.01) / (0.9×0.01 + 0.2×0.99) = 0.009 / 0.207 ≈ **4.3%**

Even with a positive test, cancer is unlikely because the base rate is so low! This is why prior probabilities matter.`,
        viz: "bayes"
      }
    ]
  },
  {
    id: 6,
    title: "Bayes Nets",
    icon: "🕸️",
    color: "#1A8A8A",
    sections: [
      {
        title: "Bayesian Network Structure",
        content: `A Bayes Net is a directed acyclic graph (DAG) where:
• **Nodes** = random variables
• **Edges** = direct dependencies
• Each node has a **conditional probability table (CPT)**

**Key property:** Each variable is conditionally independent of its non-descendants given its parents.

**Compact representation:** Instead of storing the full joint distribution (2^n values for n binary variables), we only need local CPTs.

Example: Weather → Sprinkler, Weather → Rain, Sprinkler → WetGrass, Rain → WetGrass`,
        viz: "bayesnet"
      },
      {
        title: "D-Separation & Independence",
        content: `**D-Separation** tells us which variables are conditionally independent in a Bayes Net. Three key patterns:

**Chain: A → B → C**
A and C are independent given B. (Knowing B "blocks" information flow)

**Fork: A ← B → C**
A and C are independent given B. (The common cause explains the correlation)

**Collider (V-structure): A → B ← C**
A and C are independent UNLESS B (or its descendant) is observed. Observing B "opens" the path! This is called **explaining away**.

**Explaining away example:** If the grass is wet (observed), learning it rained makes it LESS likely the sprinkler was on.`,
        viz: "dsep"
      },
      {
        title: "Inference in Bayes Nets",
        content: `**Exact Inference:**
• **Enumeration** — sum over all hidden variables (exponential)
• **Variable Elimination** — smarter ordering of summation, reuse intermediate results. Uses "factors" that are combined and marginalized.

**Approximate Inference:**
• **Rejection Sampling** — generate samples from the full network, keep only those matching evidence. Simple but wasteful.
• **Likelihood Weighting** — fix evidence variables, weight samples by probability of evidence. More efficient!
• **Gibbs Sampling** — start with random values, repeatedly resample one variable at a time conditioned on its Markov blanket. A form of MCMC.`,
        viz: null
      }
    ]
  },
  {
    id: 7,
    title: "Machine Learning",
    icon: "🤖",
    color: "#D94E8F",
    sections: [
      {
        title: "k-Nearest Neighbors",
        content: `The simplest ML algorithm: to classify a new point, look at its **k nearest neighbors** and take a vote.

**Key choices:**
• **k** — how many neighbors? Small k = noisy, large k = smooth
• **Distance metric** — Euclidean, Manhattan, etc.
• **Cross-validation** — split data into folds, train on some, test on others to find the best k

**Pros:** No training needed, works for any shape of decision boundary
**Cons:** Slow at prediction time (must search all data), sensitive to irrelevant features, curse of dimensionality`,
        viz: "knn"
      },
      {
        title: "Naive Bayes & Gaussian Classifiers",
        content: `**Gaussian Classifier:** Assume each class generates data from a Gaussian (bell curve) distribution. Classify by finding which Gaussian is most likely.

**Decision Boundary:** Where the two Gaussians are equally likely — this creates a line (or curve) separating classes.

**Naive Bayes:** Assumes all features are **conditionally independent** given the class. Despite this "naive" assumption, it works surprisingly well!

P(Class | Features) ∝ P(Class) × Π P(Feature_i | Class)

**Why it works:** Even if the independence assumption is wrong, the resulting classifier often ranks probabilities correctly.`,
        viz: "gaussian"
      },
      {
        title: "Decision Trees",
        content: `A tree where each internal node tests a feature, each branch is an outcome, and each leaf is a class prediction.

**Building a tree — which feature to split on?**
Use **Information Gain** = how much entropy decreases after the split.

**Entropy** H(S) = -Σ p_i × log₂(p_i)
• H = 0 means pure (all same class)
• H = 1 means maximally mixed (for binary)

**Avoid overfitting:**
• **Pruning** — remove branches that don't help on validation data
• **Minimum Description Length** — prefer simpler trees
• **Random Forests** — build many trees on random subsets, take a vote
• **Boosting** — build trees sequentially, focusing on mistakes`,
        viz: "decision-tree"
      },
      {
        title: "Neural Networks",
        content: `Inspired by the brain! A network of simple units (neurons) connected by weighted edges.

**Perceptron:** A single neuron computes: output = activation(Σ w_i × x_i + bias)

**Limitations:** A single perceptron can only learn linearly separable functions (can do AND, OR but NOT XOR).

**Multilayer Networks:** Stack layers of neurons. Hidden layers can learn complex, non-linear boundaries.

**Backpropagation:** Train by:
1. Forward pass — compute output
2. Compute error at output
3. Backward pass — propagate error and adjust weights using gradient descent

**Deep Learning:** Many hidden layers can learn hierarchical features (edges → shapes → objects).`,
        viz: "neural-net"
      }
    ]
  },
  {
    id: 8,
    title: "Pattern Recognition through Time",
    icon: "📈",
    color: "#8B6914",
    sections: [
      {
        title: "Dynamic Time Warping",
        content: `When comparing time series (like dolphin whistles or gestures), signals can be stretched or compressed in time.

**Euclidean distance** fails because it compares point-by-point — a slightly shifted signal looks very different!

**DTW** finds the optimal alignment between two sequences by warping the time axis. It builds a cost matrix and finds the minimum-cost path from corner to corner.

**Sakoe-Chiba Band:** Constrains the warping path to stay near the diagonal, preventing extreme distortions.`,
        viz: "dtw"
      },
      {
        title: "Hidden Markov Models",
        content: `An HMM models a system where the true state is **hidden** but produces **observable outputs**.

**Components:**
• **States** — hidden (e.g., phonemes in speech)
• **Observations** — what we can see (e.g., sound signals)
• **Transition probabilities** — P(next state | current state)
• **Emission probabilities** — P(observation | state)
• **Initial distribution** — P(starting state)

**Three key problems:**
1. **Evaluation** — P(observations | model) → Forward algorithm
2. **Decoding** — most likely state sequence → **Viterbi algorithm**
3. **Learning** — find best model parameters → **Baum-Welch** (EM algorithm)`,
        viz: "hmm"
      }
    ]
  },
  {
    id: 9,
    title: "Deep Learning",
    icon: "🧠",
    color: "#5B3E96",
    sections: [
      {
        title: "CNNs & Modern Architectures",
        content: `**Convolutional Neural Networks** are designed for spatial data (images):

• **Convolutional layers** — slide small filters across the image, detect local patterns (edges, textures)
• **Pooling layers** — downsample to reduce size and add translation invariance
• **Fully connected layers** — final classification

**Key concepts:**
• **Parameter sharing** — same filter used everywhere = far fewer parameters
• **Hierarchical features** — early layers detect edges, later layers detect complex objects

**Residual Networks (ResNets):** Add "skip connections" that let gradients flow directly through the network, enabling training of very deep networks (100+ layers).`,
        viz: "cnn"
      },
      {
        title: "Regularization & NLP",
        content: `**Regularization** prevents overfitting:
• **L2 (Weight Decay)** — penalize large weights
• **Dropout** — randomly zero out neurons during training
• **Batch Normalization** — normalize layer inputs for stable training

**NLP with Deep Learning:**
• **Word Vectors (Word2Vec)** — represent words as dense vectors where similar words are close together
• **Sequence Models (RNNs, LSTMs)** — process text sequentially, maintaining memory
• **Attention & Transformers** — let the model focus on relevant parts of the input
• **Transfer Learning** — pre-train on large corpora, fine-tune on specific tasks

**Responsible AI:** Consider transparency, interpretability, bias, and fairness in AI systems.`,
        viz: null
      }
    ]
  },
  {
    id: 10,
    title: "Planning under Uncertainty",
    icon: "🎯",
    color: "#C74D3C",
    sections: [
      {
        title: "Markov Decision Processes",
        content: `MDPs model decision-making when outcomes are **stochastic** (uncertain).

**Components:**
• **States S** — where the agent can be
• **Actions A** — what the agent can do
• **Transition function T(s,a,s')** — probability of reaching s' from s via action a
• **Reward R(s)** — immediate reward for being in state s
• **Discount factor γ** — how much we value future rewards (0 < γ < 1)

**Goal:** Find a **policy** π(s) → a that maps each state to the best action.

**Value Iteration:** Repeatedly update V(s) = max_a [R(s) + γ Σ T(s,a,s')V(s')] until convergence. The optimal policy is then: pick the action that maximizes the right side.`,
        viz: "mdp"
      },
      {
        title: "POMDPs",
        content: `**Partially Observable MDPs** — the agent can't directly see the state!

Instead of knowing the exact state, the agent maintains a **belief state** — a probability distribution over possible states.

**Example:** A robot tour guide can't perfectly sense its location, so it maintains beliefs about where it might be and takes actions that both gather information AND make progress toward the goal.

POMDPs are much harder to solve than MDPs (PSPACE-complete!). Practical approaches use approximations like point-based solvers.

**Key insight:** Sometimes the best action is one that **reduces uncertainty** (information gathering) rather than directly pursuing the goal.`,
        viz: null
      }
    ]
  }
];


const QUIZZES = {
  0: [
    { q: "In BFS, which data structure determines node expansion order?", o: ["Stack (LIFO)", "Queue (FIFO)", "Priority Queue", "Hash Map"], a: 1, e: "BFS uses a FIFO queue — first in, first out — exploring nodes level by level." },
    { q: "What does h(n) represent in A* search?", o: ["Actual cost from start to n", "Estimated cost from n to goal", "Total path cost", "Branching factor"], a: 1, e: "h(n) is the heuristic estimate of remaining cost from n to goal. f(n) = g(n) + h(n)." },
    { q: "An admissible heuristic must:", o: ["Always overestimate", "Never overestimate", "Equal the true cost", "Be zero"], a: 1, e: "Admissible = optimistic. Never overestimates, guaranteeing A* finds optimal solution." },
    { q: "UCS is equivalent to which algorithm?", o: ["Bellman-Ford", "Floyd-Warshall", "Dijkstra's", "Prim's"], a: 2, e: "UCS expands lowest-cost node first using a priority queue — exactly Dijkstra's algorithm." },
    { q: "Time complexity of BFS with branching factor b, depth d?", o: ["O(b + d)", "O(b × d)", "O(b^d)", "O(d^b)"], a: 2, e: "BFS explores all nodes at each level: O(b^d) total nodes." },
  ],
  1: [
    { q: "Main problem with basic hill climbing?", o: ["Too slow", "Gets stuck at local maxima", "Too much memory", "Can't handle continuous spaces"], a: 1, e: "Hill climbing always moves to a better neighbor, so it gets stuck at local maxima." },
    { q: "As temperature decreases in simulated annealing:", o: ["More random moves accepted", "Fewer bad moves accepted", "Algorithm restarts", "Step size increases"], a: 1, e: "As T drops, probability of accepting worse solutions decreases, search becomes greedy." },
    { q: "What does 'crossover' do in a genetic algorithm?", o: ["Randomly changes one individual", "Combines parts of two parents", "Selects the fittest", "Removes the weakest"], a: 1, e: "Crossover combines genetic material from two parents to create a child." },
    { q: "Acceptance probability for a worse solution in SA?", o: ["Always 0", "Always 1", "e^(ΔE/T)", "1/T"], a: 2, e: "Probability is e^(ΔE/T) where ΔE < 0. Higher T → higher acceptance." },
  ],
  2: [
    { q: "In Minimax, the MAX player tries to:", o: ["Minimize score", "Maximize score", "Reach a draw", "Minimize depth"], a: 1, e: "MAX wants highest payoff; MIN tries to minimize it." },
    { q: "Alpha-beta pruning best case reduces branching to:", o: ["b/2", "√b", "b²", "log(b)"], a: 1, e: "Perfect move ordering: O(b^(m/2)) nodes — square root of branching factor." },
    { q: "When does alpha-beta pruning occur?", o: ["α < β", "α ≥ β", "α = 0", "β = 0"], a: 1, e: "When α ≥ β, the branch can't affect the final decision." },
    { q: "The horizon effect occurs when:", o: ["Tree is too wide", "Limited depth hides inevitable outcomes", "Eval function is perfect", "Pruning is too aggressive"], a: 1, e: "Limited depth can push bad outcomes beyond the search horizon." },
    { q: "In Expectimax, chance nodes compute:", o: ["Maximum of children", "Minimum of children", "Weighted average", "Median"], a: 2, e: "Chance nodes compute expected value — weighted average over outcomes." },
  ],
  3: [
    { q: "Three main components of a CSP?", o: ["Nodes, Edges, Weights", "Variables, Domains, Constraints", "States, Actions, Rewards", "Inputs, Outputs, Layers"], a: 1, e: "CSP = Variables + Domains + Constraints." },
    { q: "Forward checking does what after assignment?", o: ["Backtracks", "Removes inconsistent values from neighbors", "Assigns all remaining", "Changes constraints"], a: 1, e: "Forward checking removes values from neighbors' domains that violate constraints." },
    { q: "MRV heuristic chooses:", o: ["Most legal values", "Fewest legal values", "Most constraints", "Random variable"], a: 1, e: "MRV picks the most constrained variable — 'fail-first' strategy." },
    { q: "Arc consistency ensures:", o: ["All assigned", "Every value has a compatible neighbor value", "Solution is unique", "No backtracking needed"], a: 1, e: "Each value in one domain must have a compatible value in constrained neighbor's domain." },
  ],
  4: [
    { q: "If P(A) = 0.3, what is P(¬A)?", o: ["0.3", "0.7", "0.09", "1.3"], a: 1, e: "P(¬A) = 1 - P(A) = 0.7" },
    { q: "Bayes' Rule: P(A|B) equals:", o: ["P(B|A) × P(A)", "P(B|A) × P(A) / P(B)", "P(A) × P(B)", "P(A) + P(B)"], a: 1, e: "Bayes' Rule: P(A|B) = P(B|A) × P(A) / P(B)." },
    { q: "Events A and B are independent if:", o: ["P(A ∧ B) = 0", "P(A|B) = P(A)", "P(A) + P(B) = 1", "P(A|B) = P(B|A)"], a: 1, e: "Independence: knowing B gives no info about A." },
    { q: "Positive cancer test but low P(cancer) because:", o: ["Test unreliable", "Base rate is very low", "Bayes doesn't apply", "No false positives"], a: 1, e: "Low prior overwhelms test accuracy — the base rate fallacy." },
  ],
  5: [
    { q: "In a Bayes Net, each node depends only on:", o: ["All other nodes", "Its parent nodes", "Its child nodes", "Sibling nodes"], a: 1, e: "Each variable is conditionally independent of non-descendants given parents." },
    { q: "In collider A → C ← B, when C is observed:", o: ["A,B become independent", "A,B become dependent", "Always independent", "C becomes independent"], a: 1, e: "Observing collider C 'opens' the path — explaining away." },
    { q: "In chain A → B → C, A and C independent given B?", o: ["Yes — B blocks flow", "No — always dependent", "Only if B unobserved", "Depends on CPT"], a: 0, e: "Observing B d-separates A from C, blocking information flow." },
    { q: "Variable elimination improves on enumeration by:", o: ["Random sampling", "Reusing intermediate computations", "Ignoring hidden vars", "Only MAP estimates"], a: 1, e: "VE avoids redundant computation by combining and marginalizing factors smartly." },
  ],
  6: [
    { q: "In kNN, as k increases:", o: ["Boundary more complex", "Boundary smoother", "Training slower", "Distance metric changes"], a: 1, e: "Larger k = more voters = smoother, less complex boundary." },
    { q: "Naive Bayes assumes features are:", o: ["Correlated", "Conditionally independent given class", "Normally distributed", "Equally important"], a: 1, e: "The 'naive' assumption: features independent given the class label." },
    { q: "Information Gain measures:", o: ["Tree accuracy", "Entropy reduction after split", "Tree depth", "Features used"], a: 1, e: "IG = Entropy(parent) - weighted Entropy(children)." },
    { q: "A single perceptron CANNOT learn:", o: ["AND", "OR", "XOR", "NOT"], a: 2, e: "XOR isn't linearly separable — no single line can separate the examples." },
    { q: "Backpropagation works by:", o: ["Random weight adjustment", "Propagating errors backward via gradient descent", "Adding layers", "Removing neurons"], a: 1, e: "Backprop computes gradients via chain rule and updates weights." },
  ],
  7: [
    { q: "Why is Euclidean distance bad for time series?", o: ["Too slow", "Can't handle different lengths", "Fails with time shifts", "Only works in 2D"], a: 2, e: "Point-by-point comparison fails when signals are shifted in time." },
    { q: "Viterbi algorithm finds:", o: ["Observation probability", "Most likely hidden state sequence", "Best model parameters", "Emission probabilities"], a: 1, e: "Viterbi uses DP to find the most likely state sequence." },
    { q: "Baum-Welch is used for:", o: ["Decoding states", "Computing observation probability", "Training HMM parameters", "Time warping"], a: 2, e: "Baum-Welch (EM) iteratively improves transition and emission probabilities." },
    { q: "HMM emission probabilities describe:", o: ["P(next state|current)", "P(observation|state)", "P(state|observation)", "P(starting state)"], a: 1, e: "Emission probs define how likely each observation is given the hidden state." },
  ],
  8: [
    { q: "Advantage of conv layers over fully connected for images?", o: ["Faster training", "Parameter sharing — same filter everywhere", "No activations needed", "1D only"], a: 1, e: "Conv layers share filters, drastically reducing parameters." },
    { q: "Pooling layers in a CNN:", o: ["Add parameters", "Downsample feature maps", "Increase resolution", "Apply activation"], a: 1, e: "Pooling reduces spatial dimensions, adding translation invariance." },
    { q: "ResNets solve what problem?", o: ["Overfitting", "Vanishing gradients via skip connections", "Slow inference", "High memory"], a: 1, e: "Skip connections let gradients flow directly, enabling very deep networks." },
    { q: "Dropout works by:", o: ["Removing layers", "Randomly zeroing neurons during training", "Reducing learning rate", "Adding input noise"], a: 1, e: "Dropout deactivates random neurons, preventing co-adaptation." },
  ],
  9: [
    { q: "In an MDP, a policy π(s) maps:", o: ["Actions to states", "States to actions", "States to rewards", "Actions to rewards"], a: 1, e: "A policy tells the agent what action to take in each state." },
    { q: "Discount factor γ controls:", o: ["Transition probs", "How much future rewards are valued", "Number of states", "Exploration rate"], a: 1, e: "γ determines present value of future rewards. γ→1 = long-term thinking." },
    { q: "Value Iteration converges by:", o: ["Random exploration", "Repeatedly applying Bellman equation", "Gradient descent", "Monte Carlo"], a: 1, e: "VI repeatedly applies V(s) = max_a [R(s) + γ Σ T(s,a,s')V(s')]." },
    { q: "In a POMDP, the agent maintains a:", o: ["Complete state", "Belief state (probability distribution)", "Single guess", "Action history"], a: 1, e: "Agent can't see true state, so maintains a belief distribution." },
  ],
};

function Quiz({ modIdx, color }) {
  const qs = QUIZZES[modIdx] || [];
  const [ci, setCi] = useState(0);
  const [sel, setSel] = useState(null);
  const [done, setDone] = useState(false);
  const [score, setScore] = useState(0);
  const [fin, setFin] = useState(false);
  const [hist, setHist] = useState([]);

  if (!qs.length) return null;
  const q = qs[ci];

  const pick = idx => {
    if (done) return;
    setSel(idx); setDone(true);
    const ok = idx === q.a;
    if (ok) setScore(s => s + 1);
    setHist(h => [...h, { qi: ci, sel: idx, ok }]);
  };

  const next = () => {
    if (ci < qs.length - 1) { setCi(ci + 1); setSel(null); setDone(false); }
    else setFin(true);
  };

  const retry = () => { setCi(0); setSel(null); setDone(false); setScore(0); setFin(false); setHist([]); };

  if (fin) {
    const pct = Math.round(score / qs.length * 100);
    return (
      <div className="flex flex-col items-center gap-3 py-3">
        <div className="text-3xl">{pct === 100 ? "🏆" : pct >= 80 ? "🌟" : pct >= 60 ? "👍" : "📖"}</div>
        <p style={{ fontSize: 28, fontWeight: 800, color }}>{score}/{qs.length}</p>
        <p className="text-sm text-gray-500">{pct}% — {pct === 100 ? "Perfect!" : pct >= 80 ? "Great job!" : pct >= 60 ? "Good start!" : "Keep studying!"}</p>
        <div className="w-full max-w-md space-y-2 mt-2">
          {hist.map((h, i) => (
            <div key={i} className="flex items-start gap-2 p-2 rounded-lg text-xs" style={{ background: h.ok ? "#F0FDF4" : "#FEF2F2" }}>
              <span>{h.ok ? "✅" : "❌"}</span>
              <div>
                <p className="font-medium">{qs[h.qi].q}</p>
                {!h.ok && <p className="text-gray-600 mt-0.5">You: <span className="text-red-600">{qs[h.qi].o[h.sel]}</span> → Correct: <span className="text-green-700 font-bold">{qs[h.qi].o[qs[h.qi].a]}</span></p>}
              </div>
            </div>
          ))}
        </div>
        <button onClick={retry} className="mt-3 px-5 py-2 rounded-lg text-sm font-semibold text-white" style={{ background: color }}>🔄 Retry</button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-1">
        {qs.map((_, i) => <div key={i} className="rounded-full" style={{ width: i === ci ? 18 : 8, height: 8, background: i < ci ? (hist[i]?.ok ? "#22C55E" : "#EF4444") : i === ci ? color : "#E5E7EB", transition: "all 0.2s" }} />)}
        <span className="text-xs text-gray-400 ml-auto">{ci + 1}/{qs.length}</span>
      </div>
      <p className="text-sm font-semibold">{q.q}</p>
      <div className="flex flex-col gap-2">
        {q.o.map((opt, idx) => {
          let bg = "#fff", bdr = "#e5e7eb", tc = "#374151";
          if (done) { if (idx === q.a) { bg = "#F0FDF4"; bdr = "#22C55E"; tc = "#166534"; } else if (idx === sel) { bg = "#FEF2F2"; bdr = "#EF4444"; tc = "#991B1B"; } else { tc = "#9CA3AF"; } }
          return (
            <button key={idx} onClick={() => pick(idx)} disabled={done} className="flex items-center gap-3 p-3 rounded-lg text-left text-sm" style={{ background: bg, border: `2px solid ${bdr}`, color: tc, cursor: done ? "default" : "pointer" }}>
              <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0" style={{ background: done && idx === q.a ? "#22C55E" : done && idx === sel ? "#EF4444" : `${color}20`, color: done && (idx === q.a || idx === sel) ? "#fff" : color }}>
                {done ? (idx === q.a ? "✓" : idx === sel ? "✗" : String.fromCharCode(65 + idx)) : String.fromCharCode(65 + idx)}
              </span>
              {opt}
            </button>
          );
        })}
      </div>
      {done && <div className="p-3 rounded-lg text-xs" style={{ background: "#FFFBEB", border: "1px solid #FCD34D" }}>💡 {q.e}</div>}
      {done && <button onClick={next} className="self-end px-4 py-2 rounded-lg text-sm font-semibold text-white" style={{ background: color }}>{ci < qs.length - 1 ? "Next →" : "Results 🎯"}</button>}
    </div>
  );
}

// ============== VISUALIZATION COMPONENTS ==============

function TreeSearchViz() {
  const [expanded, setExpanded] = useState(new Set(["S"]));
  const [goal, setGoal] = useState(null);

  const nodes = {
    S: { x: 200, y: 30, children: ["A", "B"] },
    A: { x: 100, y: 100, children: ["C", "D"] },
    B: { x: 300, y: 100, children: ["E", "G"] },
    C: { x: 50, y: 170, children: [] },
    D: { x: 150, y: 170, children: [] },
    E: { x: 250, y: 170, children: [] },
    G: { x: 350, y: 170, children: [] },
  };

  const expand = (id) => {
    const newExp = new Set(expanded);
    newExp.add(id);
    if (id === "G") setGoal("G");
    else setGoal(null);
    setExpanded(newExp);
  };

  const reset = () => { setExpanded(new Set(["S"])); setGoal(null); };

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-2 opacity-70">Click nodes to expand the search tree</p>
      <svg viewBox="0 0 400 210" className="w-full max-w-md">
        {Object.entries(nodes).map(([id, node]) =>
          node.children.map(childId => {
            if (!expanded.has(id)) return null;
            const child = nodes[childId];
            return <line key={`${id}-${childId}`} x1={node.x} y1={node.y + 15} x2={child.x} y2={child.y - 15} stroke={expanded.has(childId) ? "#666" : "#ddd"} strokeWidth="2" />;
          })
        )}
        {Object.entries(nodes).map(([id, node]) => {
          const isExpanded = expanded.has(id);
          const isGoal = goal === id;
          const isClickable = !isExpanded && Object.entries(nodes).some(([pid, p]) => expanded.has(pid) && p.children.includes(id));
          return (
            <g key={id} onClick={() => isClickable ? expand(id) : null} style={{ cursor: isClickable ? "pointer" : "default" }}>
              <circle cx={node.x} cy={node.y} r="15" fill={isGoal ? "#2EAF7D" : isExpanded ? "#E8483F" : isClickable ? "#fff" : "#eee"} stroke={isClickable ? "#E8483F" : "#ccc"} strokeWidth="2" strokeDasharray={isClickable ? "4" : "0"} />
              <text x={node.x} y={node.y + 5} textAnchor="middle" fontSize="12" fontWeight="bold" fill={isExpanded || isGoal ? "#fff" : "#333"}>{id}</text>
            </g>
          );
        })}
      </svg>
      {goal && <p className="text-sm font-bold mt-1" style={{ color: "#2EAF7D" }}>🎯 Goal found!</p>}
      <button onClick={reset} className="mt-2 px-3 py-1 text-xs rounded-full border border-gray-300 hover:bg-gray-100">Reset</button>
    </div>
  );
}

function BFSViz() {
  const [step, setStep] = useState(0);
  const graph = {
    nodes: [
      { id: "A", x: 50, y: 30 }, { id: "B", x: 150, y: 30 },
      { id: "C", x: 50, y: 100 }, { id: "D", x: 150, y: 100 }, { id: "E", x: 250, y: 100 },
      { id: "F", x: 100, y: 170 }, { id: "G", x: 200, y: 170 }
    ],
    edges: [["A", "C"], ["A", "D"], ["B", "D"], ["B", "E"], ["C", "F"], ["D", "F"], ["D", "G"], ["E", "G"]]
  };
  const bfsOrder = ["A", "B", "C", "D", "E", "F", "G"];
  const visited = new Set(bfsOrder.slice(0, step));
  const current = bfsOrder[step - 1];

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">BFS explores level by level (Start: A,B → Goal: G)</p>
      <svg viewBox="0 0 300 200" className="w-full max-w-sm">
        {graph.edges.map(([a, b]) => {
          const na = graph.nodes.find(n => n.id === a);
          const nb = graph.nodes.find(n => n.id === b);
          return <line key={`${a}${b}`} x1={na.x} y1={na.y} x2={nb.x} y2={nb.y} stroke={visited.has(a) && visited.has(b) ? "#E8483F" : "#ddd"} strokeWidth="2" />;
        })}
        {graph.nodes.map(n => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r="18" fill={n.id === current ? "#E8483F" : visited.has(n.id) ? "#FECACA" : "#f3f4f6"} stroke={visited.has(n.id) ? "#E8483F" : "#ccc"} strokeWidth="2" />
            <text x={n.x} y={n.y + 5} textAnchor="middle" fontSize="12" fontWeight="bold" fill={n.id === current ? "#fff" : "#333"}>{n.id}</text>
          </g>
        ))}
      </svg>
      <div className="flex gap-2 mt-1">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">← Back</button>
        <span className="text-xs py-1">Step {step}/{bfsOrder.length}</span>
        <button onClick={() => setStep(Math.min(bfsOrder.length, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Next →</button>
      </div>
      <p className="text-xs mt-1 font-mono">Queue: [{bfsOrder.slice(step).join(", ")}]</p>
    </div>
  );
}

function UCSViz() {
  const [step, setStep] = useState(0);
  const nodes = [
    { id: "S", x: 50, y: 90 }, { id: "A", x: 150, y: 30 },
    { id: "B", x: 150, y: 150 }, { id: "C", x: 250, y: 30 },
    { id: "G", x: 300, y: 90 }
  ];
  const edges = [
    { from: "S", to: "A", cost: 1 }, { from: "S", to: "B", cost: 5 },
    { from: "A", to: "C", cost: 2 }, { from: "B", to: "G", cost: 1 },
    { from: "C", to: "G", cost: 3 }
  ];
  const ucsSteps = [
    { visited: [], current: "S", cost: 0, label: "Start at S (cost 0)" },
    { visited: ["S"], current: "A", cost: 1, label: "Expand A (cost 1)" },
    { visited: ["S", "A"], current: "C", cost: 3, label: "Expand C (cost 1+2=3)" },
    { visited: ["S", "A", "C"], current: "B", cost: 5, label: "Expand B (cost 5)" },
    { visited: ["S", "A", "C", "B"], current: "G", cost: 6, label: "Goal G found! (cost 1+2+3=6, not 5+1=6 — same!)" },
  ];
  const s = ucsSteps[Math.min(step, ucsSteps.length - 1)];

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">UCS expands the cheapest node first</p>
      <svg viewBox="0 0 350 180" className="w-full max-w-sm">
        {edges.map(e => {
          const a = nodes.find(n => n.id === e.from);
          const b = nodes.find(n => n.id === e.to);
          const active = s.visited.includes(e.from) && (s.visited.includes(e.to) || s.current === e.to);
          return (
            <g key={`${e.from}${e.to}`}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={active ? "#E07B39" : "#ddd"} strokeWidth="2" />
              <text x={(a.x + b.x) / 2 + 8} y={(a.y + b.y) / 2 - 5} fontSize="11" fill="#E07B39" fontWeight="bold">{e.cost}</text>
            </g>
          );
        })}
        {nodes.map(n => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r="18" fill={n.id === s.current ? "#E07B39" : s.visited.includes(n.id) ? "#FDE68A" : "#f3f4f6"} stroke={s.visited.includes(n.id) || n.id === s.current ? "#E07B39" : "#ccc"} strokeWidth="2" />
            <text x={n.x} y={n.y + 5} textAnchor="middle" fontSize="13" fontWeight="bold" fill={n.id === s.current ? "#fff" : "#333"}>{n.id}</text>
          </g>
        ))}
      </svg>
      <p className="text-xs font-medium mt-1">{s.label}</p>
      <div className="flex gap-2 mt-1">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">← Back</button>
        <button onClick={() => setStep(Math.min(ucsSteps.length - 1, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Next →</button>
      </div>
    </div>
  );
}

function AStarViz() {
  const [step, setStep] = useState(0);
  const nodes = [
    { id: "S", x: 40, y: 90, h: 6 }, { id: "A", x: 130, y: 30, h: 4 },
    { id: "B", x: 130, y: 150, h: 4 }, { id: "C", x: 220, y: 30, h: 2 },
    { id: "D", x: 220, y: 150, h: 3 }, { id: "G", x: 310, y: 90, h: 0 }
  ];
  const edges = [
    { from: "S", to: "A", cost: 2 }, { from: "S", to: "B", cost: 3 },
    { from: "A", to: "C", cost: 3 }, { from: "B", to: "D", cost: 2 },
    { from: "C", to: "G", cost: 2 }, { from: "D", to: "G", cost: 4 }
  ];
  const steps = [
    { visited: [], current: "S", f: "0+6=6", path: "S" },
    { visited: ["S"], current: "A", f: "2+4=6", path: "S→A" },
    { visited: ["S", "A"], current: "C", f: "5+2=7", path: "S→A→C" },
    { visited: ["S", "A", "C"], current: "G", f: "7+0=7", path: "S→A→C→G ✓" },
  ];
  const s = steps[Math.min(step, steps.length - 1)];

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">A* uses f(n) = g(n) + h(n). Numbers on nodes = h(n)</p>
      <svg viewBox="0 0 360 180" className="w-full max-w-sm">
        {edges.map(e => {
          const a = nodes.find(n => n.id === e.from);
          const b = nodes.find(n => n.id === e.to);
          const active = s.path.includes(e.from) && s.path.includes(e.to);
          return (
            <g key={`${e.from}${e.to}`}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={active ? "#E8483F" : "#ddd"} strokeWidth="2" />
              <text x={(a.x + b.x) / 2 + 8} y={(a.y + b.y) / 2} fontSize="10" fill="#888">{e.cost}</text>
            </g>
          );
        })}
        {nodes.map(n => (
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r="20" fill={n.id === s.current ? "#E8483F" : s.visited.includes(n.id) ? "#FECACA" : "#f3f4f6"} stroke={s.visited.includes(n.id) || n.id === s.current ? "#E8483F" : "#ccc"} strokeWidth="2" />
            <text x={n.x} y={n.y + 1} textAnchor="middle" fontSize="11" fontWeight="bold" fill={n.id === s.current ? "#fff" : "#333"}>{n.id}</text>
            <text x={n.x} y={n.y + 12} textAnchor="middle" fontSize="8" fill={n.id === s.current ? "#fcc" : "#999"}>h={n.h}</text>
          </g>
        ))}
      </svg>
      <p className="text-xs mt-1"><span className="font-bold">f = {s.f}</span> | Path: {s.path}</p>
      <div className="flex gap-2 mt-1">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function HillClimbingViz() {
  const [pos, setPos] = useState(20);
  const landscape = (x) => 30 * Math.sin(x * 0.08) + 15 * Math.sin(x * 0.15 + 1) + 8 * Math.sin(x * 0.3 + 2);
  const y = (x) => 100 - landscape(x);

  const moveRight = () => {
    const curr = landscape(pos);
    const next = landscape(pos + 5);
    if (next > curr) setPos(pos + 5);
  };
  const moveLeft = () => {
    const curr = landscape(pos);
    const next = landscape(pos - 5);
    if (next > curr) setPos(pos - 5);
  };

  const pathD = Array.from({ length: 60 }, (_, i) => {
    const x = i * 5;
    return `${i === 0 ? 'M' : 'L'}${x},${y(x)}`;
  }).join(' ');

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Hill climbing: try to reach the highest peak. Gets stuck at local maxima!</p>
      <svg viewBox="0 0 300 140" className="w-full max-w-sm">
        <path d={pathD + " L300,140 L0,140 Z"} fill="#E8F5E9" stroke="#2EAF7D" strokeWidth="1.5" />
        <circle cx={pos} cy={y(pos)} r="6" fill="#E8483F" stroke="#fff" strokeWidth="2" />
        <text x={pos} y={y(pos) - 12} textAnchor="middle" fontSize="14">🏃</text>
      </svg>
      <div className="flex gap-2 mt-1">
        <button onClick={moveLeft} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">← Move Left</button>
        <button onClick={moveRight} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Move Right →</button>
        <button onClick={() => setPos(Math.floor(Math.random() * 250) + 20)} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Random Restart</button>
      </div>
    </div>
  );
}

function AnnealingViz() {
  const [temp, setTemp] = useState(100);
  const [pos, setPos] = useState(60);
  const [running, setRunning] = useState(false);
  const landscape = (x) => 30 * Math.sin(x * 0.08) + 15 * Math.sin(x * 0.15 + 1) + 8 * Math.sin(x * 0.3 + 2);
  const y = (x) => 100 - landscape(x);

  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      setTemp(t => {
        if (t <= 1) { setRunning(false); return 1; }
        return t * 0.97;
      });
      setPos(p => {
        const delta = (Math.random() - 0.5) * 20;
        const newP = Math.max(5, Math.min(290, p + delta));
        const diff = landscape(newP) - landscape(p);
        if (diff > 0 || Math.random() < Math.exp(diff / (temp * 0.3))) return newP;
        return p;
      });
    }, 80);
    return () => clearInterval(interval);
  }, [running, temp]);

  const pathD = Array.from({ length: 60 }, (_, i) => {
    const x = i * 5;
    return `${i === 0 ? 'M' : 'L'}${x},${y(x)}`;
  }).join(' ');

  return (
    <div className="flex flex-col items-center">
      <div className="flex items-center gap-3 mb-1">
        <span className="text-xs">🌡️ Temp: <span className="font-bold" style={{ color: `hsl(${120 - temp * 1.2}, 80%, 40%)` }}>{temp.toFixed(0)}</span></span>
        <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all" style={{ width: `${temp}%`, background: `hsl(${120 - temp * 1.2}, 80%, 50%)` }} />
        </div>
      </div>
      <svg viewBox="0 0 300 140" className="w-full max-w-sm">
        <path d={pathD + " L300,140 L0,140 Z"} fill="#FFF3E0" stroke="#E07B39" strokeWidth="1.5" />
        <circle cx={pos} cy={y(pos)} r="6" fill="#E07B39" stroke="#fff" strokeWidth="2" />
      </svg>
      <div className="flex gap-2 mt-1">
        <button onClick={() => { setRunning(true); setTemp(100); }} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100" disabled={running}>{running ? "Running..." : "▶ Start Annealing"}</button>
        <button onClick={() => { setRunning(false); setTemp(100); setPos(60); }} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Reset</button>
      </div>
    </div>
  );
}

function GeneticViz() {
  const [gen, setGen] = useState(0);
  const [pop, setPop] = useState(() => Array.from({ length: 6 }, () => Array.from({ length: 8 }, () => Math.floor(Math.random() * 8))));
  const fitness = (ind) => {
    let clashes = 0;
    for (let i = 0; i < ind.length; i++)
      for (let j = i + 1; j < ind.length; j++)
        if (ind[i] === ind[j] || Math.abs(ind[i] - ind[j]) === j - i) clashes++;
    return 28 - clashes;
  };

  const evolve = () => {
    const scored = pop.map(p => ({ ind: p, fit: fitness(p) })).sort((a, b) => b.fit - a.fit);
    const newPop = [scored[0].ind, scored[1].ind];
    while (newPop.length < 6) {
      const p1 = scored[Math.floor(Math.random() * 3)].ind;
      const p2 = scored[Math.floor(Math.random() * 3)].ind;
      const cut = Math.floor(Math.random() * 6) + 1;
      const child = [...p1.slice(0, cut), ...p2.slice(cut)];
      if (Math.random() < 0.3) child[Math.floor(Math.random() * 8)] = Math.floor(Math.random() * 8);
      newPop.push(child);
    }
    setPop(newPop);
    setGen(g => g + 1);
  };

  const best = pop.reduce((a, b) => fitness(a) > fitness(b) ? a : b);

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">8-Queens: Evolving solutions. Fitness = 28 - conflicts</p>
      <div className="grid grid-cols-8 gap-0 border border-gray-300 mb-1" style={{ width: 160, height: 160 }}>
        {Array.from({ length: 64 }).map((_, i) => {
          const r = Math.floor(i / 8);
          const c = i % 8;
          const isQ = best[c] === r;
          return <div key={i} className="flex items-center justify-center text-xs" style={{ width: 20, height: 20, background: (r + c) % 2 === 0 ? "#f3e8ff" : "#fff" }}>{isQ ? "♛" : ""}</div>;
        })}
      </div>
      <p className="text-xs">Gen: {gen} | Best fitness: <span className="font-bold">{fitness(best)}/28</span></p>
      <div className="flex gap-2 mt-1">
        <button onClick={evolve} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Evolve</button>
        <button onClick={() => { for (let i = 0; i < 20; i++)evolve(); }} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">×20</button>
      </div>
    </div>
  );
}

function MinimaxViz() {
  const [showValues, setShowValues] = useState(false);
  const leaves = [3, 12, 8, 2, 4, 6, 14, 5, 2];
  const minL = [Math.min(leaves[0], leaves[1], leaves[2]), Math.min(leaves[3], leaves[4], leaves[5]), Math.min(leaves[6], leaves[7], leaves[8])];
  const maxRoot = Math.max(...minL);

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">MAX picks highest, MIN picks lowest</p>
      <svg viewBox="0 0 360 180" className="w-full max-w-md">
        {/* Root (MAX) */}
        <line x1="180" y1="30" x2="60" y2="80" stroke="#ccc" strokeWidth="1.5" />
        <line x1="180" y1="30" x2="180" y2="80" stroke="#ccc" strokeWidth="1.5" />
        <line x1="180" y1="30" x2="300" y2="80" stroke="#ccc" strokeWidth="1.5" />
        {/* MIN to leaves */}
        {[0, 1, 2].map(i => [0, 1, 2].map(j =>
          <line key={`${i}${j}`} x1={60 + i * 120} y1="80" x2={20 + i * 120 + j * 40} y2="150" stroke="#eee" strokeWidth="1" />
        ))}
        {/* Root */}
        <polygon points="160,10 200,10 190,35 170,35" fill={showValues ? "#E8483F" : "#f3f4f6"} stroke="#E8483F" strokeWidth="2" />
        <text x="180" y="27" textAnchor="middle" fontSize="11" fontWeight="bold" fill={showValues ? "#fff" : "#E8483F"}>MAX{showValues ? `=${maxRoot}` : ""}</text>
        {/* MIN nodes */}
        {minL.map((v, i) => (
          <g key={i}>
            <circle cx={60 + i * 120} cy="80" r="18" fill={showValues ? "#4A90D9" : "#f3f4f6"} stroke="#4A90D9" strokeWidth="2" />
            <text x={60 + i * 120} y="84" textAnchor="middle" fontSize="10" fontWeight="bold" fill={showValues ? "#fff" : "#4A90D9"}>MIN{showValues ? `=${v}` : ""}</text>
          </g>
        ))}
        {/* Leaves */}
        {leaves.map((v, i) => (
          <g key={i}>
            <rect x={12 + Math.floor(i / 3) * 120 + (i % 3) * 40} y="140" width="28" height="22" rx="4" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="1" />
            <text x={26 + Math.floor(i / 3) * 120 + (i % 3) * 40} y="155" textAnchor="middle" fontSize="11" fontWeight="bold">{v}</text>
          </g>
        ))}
      </svg>
      <button onClick={() => setShowValues(!showValues)} className="px-4 py-1 text-xs rounded-full border hover:bg-gray-100">{showValues ? "Hide" : "Show"} Minimax Values</button>
    </div>
  );
}

function AlphaBetaViz() {
  const [step, setStep] = useState(0);
  const steps = [
    { desc: "Start: α=-∞, β=+∞", pruned: [] },
    { desc: "Visit leaf 3 → MIN node α=-∞, β=3", pruned: [] },
    { desc: "Visit leaf 5 → MIN=3 (5>3, MIN keeps 3)", pruned: [] },
    { desc: "MAX updates α=3", pruned: [] },
    { desc: "Visit leaf 2 → Right MIN β=2, but α=3 > β=2 → PRUNE!", pruned: ["p1"] },
    { desc: "MAX result = 3 (pruned saved us exploring!)", pruned: ["p1"] },
  ];
  const s = steps[Math.min(step, steps.length - 1)];

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Alpha-Beta pruning skips irrelevant branches</p>
      <svg viewBox="0 0 280 150" className="w-full max-w-xs">
        <line x1="140" y1="25" x2="70" y2="60" stroke="#ccc" strokeWidth="1.5" />
        <line x1="140" y1="25" x2="210" y2="60" stroke={s.pruned.includes("p1") ? "#E8483F" : "#ccc"} strokeWidth="1.5" strokeDasharray={s.pruned.includes("p1") ? "4" : "0"} />
        <line x1="70" y1="60" x2="40" y2="115" stroke="#ccc" strokeWidth="1" />
        <line x1="70" y1="60" x2="100" y2="115" stroke="#ccc" strokeWidth="1" />
        <line x1="210" y1="60" x2="180" y2="115" stroke={s.pruned.includes("p1") ? "#fee" : "#ccc"} strokeWidth="1" />
        <line x1="210" y1="60" x2="240" y2="115" stroke={s.pruned.includes("p1") ? "#fee" : "#ccc"} strokeWidth="1" />

        <polygon points="125,8 155,8 148,30 132,30" fill={step >= 3 ? "#E8483F" : "#f5f5f5"} stroke="#E8483F" strokeWidth="2" />
        <text x="140" y="23" textAnchor="middle" fontSize="9" fontWeight="bold" fill={step >= 3 ? "#fff" : "#E8483F"}>MAX</text>

        <circle cx="70" cy="60" r="14" fill={step >= 2 ? "#4A90D9" : "#f5f5f5"} stroke="#4A90D9" strokeWidth="2" />
        <text x="70" y="64" textAnchor="middle" fontSize="9" fontWeight="bold" fill={step >= 2 ? "#fff" : "#4A90D9"}>{step >= 2 ? "3" : "MIN"}</text>

        <circle cx="210" cy="60" r="14" fill={s.pruned.includes("p1") ? "#FEE2E2" : "#f5f5f5"} stroke={s.pruned.includes("p1") ? "#E8483F" : "#4A90D9"} strokeWidth="2" />
        <text x="210" y="64" textAnchor="middle" fontSize="9" fontWeight="bold" fill={s.pruned.includes("p1") ? "#E8483F" : "#4A90D9"}>{step >= 4 ? "✂" : "MIN"}</text>

        {[{ v: 3, x: 40 }, { v: 5, x: 100 }, { v: 2, x: 180 }, { v: 8, x: 240 }].map((l, i) => (
          <g key={i} opacity={i >= 2 && s.pruned.includes("p1") ? 0.3 : 1}>
            <rect x={l.x - 12} y="108" width="24" height="20" rx="3" fill="#FEF3C7" stroke="#F59E0B" />
            <text x={l.x} y="122" textAnchor="middle" fontSize="11" fontWeight="bold">{l.v}</text>
          </g>
        ))}
        {s.pruned.includes("p1") && <text x="210" y="95" textAnchor="middle" fontSize="18">✂️</text>}
      </svg>
      <p className="text-xs font-medium mt-1 text-center max-w-xs">{s.desc}</p>
      <div className="flex gap-2 mt-1">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <span className="text-xs py-1">{Math.min(step, steps.length - 1) + 1}/{steps.length}</span>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function CSPMapViz() {
  const [colors, setColors] = useState({ WA: null, NT: null, SA: null, Q: null, NSW: null, V: null, T: null });
  const palette = ["#E8483F", "#2EAF7D", "#4A90D9"];
  const pNames = ["Red", "Green", "Blue"];
  const adj = { WA: ["NT", "SA"], NT: ["WA", "SA", "Q"], SA: ["WA", "NT", "Q", "NSW", "V"], Q: ["NT", "SA", "NSW"], NSW: ["Q", "SA", "V"], V: ["SA", "NSW"], T: [] };
  const positions = { WA: { x: 60, y: 80 }, NT: { x: 150, y: 40 }, SA: { x: 150, y: 110 }, Q: { x: 240, y: 40 }, NSW: { x: 240, y: 100 }, V: { x: 220, y: 150 }, T: { x: 230, y: 195 } };

  const conflict = (r) => {
    if (colors[r] === null) return false;
    return adj[r].some(n => colors[n] === colors[r]);
  };

  const cycleColor = (r) => {
    const curr = colors[r];
    const next = curr === null ? 0 : (curr + 1) % 4;
    setColors({ ...colors, [r]: next >= 3 ? null : next });
  };

  const allColored = Object.values(colors).every(c => c !== null);
  const noConflicts = Object.keys(colors).every(r => !conflict(r));

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Click regions to color them. No two adjacent regions can share a color!</p>
      <svg viewBox="0 0 300 220" className="w-full max-w-xs">
        {Object.entries(adj).map(([r, neighbors]) =>
          neighbors.map(n => {
            const a = positions[r], b = positions[n];
            const isConflict = colors[r] !== null && colors[n] !== null && colors[r] === colors[n];
            return <line key={`${r}-${n}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={isConflict ? "#E8483F" : "#ddd"} strokeWidth={isConflict ? 2.5 : 1.5} />;
          })
        )}
        {Object.entries(positions).map(([r, p]) => (
          <g key={r} onClick={() => cycleColor(r)} style={{ cursor: "pointer" }}>
            <circle cx={p.x} cy={p.y} r="22" fill={colors[r] !== null ? palette[colors[r]] : "#f3f4f6"} stroke={conflict(r) ? "#E8483F" : "#999"} strokeWidth={conflict(r) ? 3 : 1.5} />
            <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="11" fontWeight="bold" fill={colors[r] !== null ? "#fff" : "#333"}>{r}</text>
          </g>
        ))}
      </svg>
      {allColored && noConflicts && <p className="text-sm font-bold" style={{ color: "#2EAF7D" }}>✅ Valid coloring!</p>}
      <button onClick={() => setColors({ WA: null, NT: null, SA: null, Q: null, NSW: null, V: null, T: null })} className="mt-1 px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Reset</button>
    </div>
  );
}

function BacktrackingViz() {
  const [step, setStep] = useState(0);
  const steps = [
    { assign: {}, msg: "Start: no assignments" },
    { assign: { WA: "R" }, msg: "Assign WA=Red" },
    { assign: { WA: "R", NT: "G" }, msg: "Assign NT=Green (can't be Red — adj to WA)" },
    { assign: { WA: "R", NT: "G", SA: "B" }, msg: "Assign SA=Blue (can't be Red or Green)" },
    { assign: { WA: "R", NT: "G", SA: "B", Q: "R" }, msg: "Assign Q=Red (can't be Green or Blue)" },
    { assign: { WA: "R", NT: "G", SA: "B", Q: "R", NSW: "G" }, msg: "Assign NSW=Green (can't be Red or Blue)" },
    { assign: { WA: "R", NT: "G", SA: "B", Q: "R", NSW: "G", V: "R" }, msg: "Assign V=Red (can't be Blue or Green)" },
    { assign: { WA: "R", NT: "G", SA: "B", Q: "R", NSW: "G", V: "R", T: "R" }, msg: "Assign T=Red (no constraints!) ✅ Done!" },
  ];
  const s = steps[Math.min(step, steps.length - 1)];
  const cMap = { R: "#E8483F", G: "#2EAF7D", B: "#4A90D9" };

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Backtracking assigns one variable at a time</p>
      <div className="flex gap-1 flex-wrap justify-center mb-1">
        {["WA", "NT", "SA", "Q", "NSW", "V", "T"].map(r => (
          <span key={r} className="px-2 py-0.5 rounded text-xs font-mono" style={{ background: s.assign[r] ? cMap[s.assign[r]] : "#eee", color: s.assign[r] ? "#fff" : "#666" }}>{r}={s.assign[r] || "?"}</span>
        ))}
      </div>
      <p className="text-xs font-medium">{s.msg}</p>
      <div className="flex gap-2 mt-2">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">← Back</button>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Next →</button>
      </div>
    </div>
  );
}

function ProbabilityViz() {
  const [flips, setFlips] = useState([]);
  const flip = () => setFlips([...flips, Math.random() < 0.5 ? "H" : "T"]);
  const flipMany = () => {
    const arr = Array.from({ length: 50 }, () => Math.random() < 0.5 ? "H" : "T");
    setFlips([...flips, ...arr]);
  };
  const heads = flips.filter(f => f === "H").length;
  const pct = flips.length > 0 ? (heads / flips.length * 100).toFixed(1) : 0;

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Law of large numbers: proportion converges to true probability</p>
      <div className="flex gap-4 mb-2">
        <span className="text-xs">Flips: <b>{flips.length}</b></span>
        <span className="text-xs">Heads: <b>{heads}</b> ({pct}%)</span>
      </div>
      <div className="w-full max-w-xs h-3 bg-gray-200 rounded-full overflow-hidden mb-2">
        <div className="h-full bg-yellow-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="flex gap-2">
        <button onClick={flip} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">🪙 Flip 1</button>
        <button onClick={flipMany} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">🪙 Flip 50</button>
        <button onClick={() => setFlips([])} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Reset</button>
      </div>
    </div>
  );
}

function BayesViz() {
  const [prior, setPrior] = useState(1);
  const [sensitivity, setSensitivity] = useState(90);
  const [falsePos, setFalsePos] = useState(20);
  const pDisease = prior / 100;
  const pPosGivenD = sensitivity / 100;
  const pPosGivenNotD = falsePos / 100;
  const pPos = pPosGivenD * pDisease + pPosGivenNotD * (1 - pDisease);
  const posterior = pPos > 0 ? (pPosGivenD * pDisease / pPos * 100).toFixed(1) : 0;

  return (
    <div className="flex flex-col items-center gap-2">
      <p className="text-xs opacity-70">Adjust parameters to see Bayes' Rule in action</p>
      <div className="w-full max-w-xs space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs w-28">Prior P(D): {prior}%</span>
          <input type="range" min="1" max="50" value={prior} onChange={e => setPrior(+e.target.value)} className="flex-1" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs w-28">Sensitivity: {sensitivity}%</span>
          <input type="range" min="50" max="99" value={sensitivity} onChange={e => setSensitivity(+e.target.value)} className="flex-1" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs w-28">False +: {falsePos}%</span>
          <input type="range" min="1" max="50" value={falsePos} onChange={e => setFalsePos(+e.target.value)} className="flex-1" />
        </div>
      </div>
      <div className="text-center p-3 rounded-lg" style={{ background: "#E8F5E9" }}>
        <p className="text-xs">P(Disease | Positive Test) =</p>
        <p className="text-2xl font-bold" style={{ color: "#2EAF7D" }}>{posterior}%</p>
      </div>
    </div>
  );
}

function BayesNetViz() {
  const nodes = [
    { id: "Cloudy", x: 150, y: 20 },
    { id: "Sprinkler", x: 60, y: 90 },
    { id: "Rain", x: 240, y: 90 },
    { id: "Wet Grass", x: 150, y: 160 },
  ];
  const edges = [["Cloudy", "Sprinkler"], ["Cloudy", "Rain"], ["Sprinkler", "Wet Grass"], ["Rain", "Wet Grass"]];

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Classic Bayes Net: arrows show causal dependencies</p>
      <svg viewBox="0 0 300 190" className="w-full max-w-xs">
        <defs>
          <marker id="arrowBN" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#888" /></marker>
        </defs>
        {edges.map(([a, b]) => {
          const na = nodes.find(n => n.id === a);
          const nb = nodes.find(n => n.id === b);
          const dx = nb.x - na.x, dy = nb.y - na.y;
          const len = Math.sqrt(dx * dx + dy * dy);
          return <line key={`${a}${b}`} x1={na.x + dx / len * 22} y1={na.y + dy / len * 14} x2={nb.x - dx / len * 22} y2={nb.y - dy / len * 14} stroke="#888" strokeWidth="1.5" markerEnd="url(#arrowBN)" />;
        })}
        {nodes.map(n => (
          <g key={n.id}>
            <ellipse cx={n.x} cy={n.y} rx="45" ry="16" fill="#EDE9FE" stroke="#7B61C1" strokeWidth="1.5" />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="10" fontWeight="bold" fill="#4C1D95">{n.id}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function DSepViz() {
  const [pattern, setPattern] = useState(0);
  const patterns = [
    { name: "Chain: A → B → C", nodes: [{ id: "A", x: 40 }, { id: "B", x: 150 }, { id: "C", x: 260 }], edges: [["A", "B"], ["B", "C"]], desc: "A ⊥ C | B. Observing B blocks information flow." },
    { name: "Fork: A ← B → C", nodes: [{ id: "A", x: 40 }, { id: "B", x: 150 }, { id: "C", x: 260 }], edges: [["B", "A"], ["B", "C"]], desc: "A ⊥ C | B. Common cause B explains correlation." },
    { name: "Collider: A → B ← C", nodes: [{ id: "A", x: 40 }, { id: "B", x: 150 }, { id: "C", x: 260 }], edges: [["A", "B"], ["C", "B"]], desc: "A ⊥ C (unconditionally). But A ⊥̸ C | B! Observing B creates dependence (explaining away)." },
  ];
  const p = patterns[pattern];

  return (
    <div className="flex flex-col items-center">
      <div className="flex gap-1 mb-2">
        {patterns.map((pt, i) => (
          <button key={i} onClick={() => setPattern(i)} className={`px-2 py-1 text-xs rounded-full border ${pattern === i ? "bg-purple-100 border-purple-400" : "hover:bg-gray-100"}`}>{pt.name.split(":")[0]}</button>
        ))}
      </div>
      <svg viewBox="0 0 300 60" className="w-full max-w-xs">
        <defs><marker id="arrDS" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#7B61C1" /></marker></defs>
        {p.edges.map(([a, b]) => {
          const na = p.nodes.find(n => n.id === a);
          const nb = p.nodes.find(n => n.id === b);
          return <line key={`${a}${b}`} x1={na.x + 20} y1={30} x2={nb.x - 20} y2={30} stroke="#7B61C1" strokeWidth="2" markerEnd="url(#arrDS)" />;
        })}
        {p.nodes.map(n => (
          <g key={n.id}>
            <circle cx={n.x} cy={30} r="18" fill="#EDE9FE" stroke="#7B61C1" strokeWidth="2" />
            <text x={n.x} y={34} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#4C1D95">{n.id}</text>
          </g>
        ))}
      </svg>
      <p className="text-xs text-center mt-1 max-w-xs font-medium">{p.desc}</p>
    </div>
  );
}

function KNNViz() {
  const [k, setK] = useState(3);
  const [testPt, setTestPt] = useState({ x: 150, y: 100 });
  const data = [
    { x: 50, y: 50, c: 0 }, { x: 70, y: 80, c: 0 }, { x: 60, y: 120, c: 0 }, { x: 90, y: 60, c: 0 }, { x: 80, y: 100, c: 0 },
    { x: 200, y: 140, c: 1 }, { x: 230, y: 120, c: 1 }, { x: 220, y: 160, c: 1 }, { x: 250, y: 130, c: 1 }, { x: 210, y: 100, c: 1 },
    { x: 130, y: 40, c: 0 }, { x: 180, y: 70, c: 1 }, { x: 160, y: 150, c: 1 }, { x: 110, y: 130, c: 0 },
  ];
  const colors = ["#E8483F", "#4A90D9"];

  const dists = data.map((d, i) => ({ i, dist: Math.sqrt((d.x - testPt.x) ** 2 + (d.y - testPt.y) ** 2), c: d.c })).sort((a, b) => a.dist - b.dist);
  const neighbors = dists.slice(0, k);
  const vote0 = neighbors.filter(n => n.c === 0).length;
  const vote1 = neighbors.filter(n => n.c === 1).length;
  const prediction = vote0 > vote1 ? 0 : 1;

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Drag the ⬜ test point. k neighbors vote on the class.</p>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs">k =</span>
        {[1, 3, 5, 7].map(v => (
          <button key={v} onClick={() => setK(v)} className={`px-2 py-0.5 text-xs rounded-full border ${k === v ? "bg-gray-800 text-white" : "hover:bg-gray-100"}`}>{v}</button>
        ))}
      </div>
      <svg viewBox="0 0 300 200" className="w-full max-w-xs border border-gray-200 rounded-lg bg-white"
        onMouseMove={e => {
          const svg = e.currentTarget;
          const rect = svg.getBoundingClientRect();
          const x = (e.clientX - rect.left) / rect.width * 300;
          const y = (e.clientY - rect.top) / rect.height * 200;
          setTestPt({ x, y });
        }}
      >
        {neighbors.map(n => (
          <line key={n.i} x1={testPt.x} y1={testPt.y} x2={data[n.i].x} y2={data[n.i].y} stroke="#ddd" strokeWidth="1" strokeDasharray="3" />
        ))}
        {data.map((d, i) => (
          <circle key={i} cx={d.x} cy={d.y} r={neighbors.some(n => n.i === i) ? 7 : 5} fill={colors[d.c]} stroke={neighbors.some(n => n.i === i) ? "#333" : "none"} strokeWidth="2" opacity={neighbors.some(n => n.i === i) ? 1 : 0.5} />
        ))}
        <rect x={testPt.x - 6} y={testPt.y - 6} width="12" height="12" fill={colors[prediction]} stroke="#333" strokeWidth="2" />
      </svg>
      <p className="text-xs mt-1">Prediction: <span className="font-bold" style={{ color: colors[prediction] }}>{prediction === 0 ? "Red" : "Blue"}</span> ({vote0} red vs {vote1} blue)</p>
    </div>
  );
}

function GaussianViz() {
  const gaussian = (x, mu, sigma) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
  const xs = Array.from({ length: 100 }, (_, i) => i * 3);

  const c1 = { mu: 100, sigma: 25, color: "#E8483F" };
  const c2 = { mu: 200, sigma: 30, color: "#4A90D9" };

  const path = (cls) => xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x},${150 - gaussian(x, cls.mu, cls.sigma) * 4000}`).join(' ');

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Two Gaussian classes. The decision boundary is where they cross.</p>
      <svg viewBox="0 0 300 160" className="w-full max-w-xs">
        <line x1="0" y1="150" x2="300" y2="150" stroke="#ccc" strokeWidth="1" />
        <path d={path(c1)} fill="none" stroke={c1.color} strokeWidth="2" />
        <path d={path(c2)} fill="none" stroke={c2.color} strokeWidth="2" />
        <path d={path(c1) + ` L${xs[xs.length - 1]},150 L0,150 Z`} fill={c1.color} opacity="0.15" />
        <path d={path(c2) + ` L${xs[xs.length - 1]},150 L0,150 Z`} fill={c2.color} opacity="0.15" />
        {/* Decision boundary approximate */}
        <line x1="148" y1="10" x2="148" y2="150" stroke="#333" strokeWidth="1" strokeDasharray="4" />
        <text x="148" y="8" textAnchor="middle" fontSize="8" fill="#333">Decision Boundary</text>
        <text x={c1.mu} y="145" textAnchor="middle" fontSize="9" fill={c1.color} fontWeight="bold">Class A</text>
        <text x={c2.mu} y="145" textAnchor="middle" fontSize="9" fill={c2.color} fontWeight="bold">Class B</text>
      </svg>
    </div>
  );
}

function DecisionTreeViz() {
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">A simple decision tree for "Play Tennis?"</p>
      <svg viewBox="0 0 320 200" className="w-full max-w-sm">
        {/* Root */}
        <rect x="110" y="5" width="100" height="28" rx="6" fill="#7B61C1" /><text x="160" y="24" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#fff">Outlook?</text>
        {/* Level 1 */}
        <line x1="135" y1="33" x2="50" y2="70" stroke="#ccc" strokeWidth="1.5" /><text x="80" y="48" fontSize="8" fill="#666">Sunny</text>
        <line x1="160" y1="33" x2="160" y2="70" stroke="#ccc" strokeWidth="1.5" /><text x="172" y="50" fontSize="8" fill="#666">Overcast</text>
        <line x1="185" y1="33" x2="270" y2="70" stroke="#ccc" strokeWidth="1.5" /><text x="235" y="48" fontSize="8" fill="#666">Rain</text>

        <rect x="5" y="70" width="90" height="24" rx="6" fill="#7B61C1" /><text x="50" y="86" textAnchor="middle" fontSize="9" fontWeight="bold" fill="#fff">Humidity?</text>
        <rect x="120" y="70" width="80" height="24" rx="6" fill="#2EAF7D" /><text x="160" y="86" textAnchor="middle" fontSize="9" fontWeight="bold" fill="#fff">Yes ✓</text>
        <rect x="225" y="70" width="90" height="24" rx="6" fill="#7B61C1" /><text x="270" y="86" textAnchor="middle" fontSize="9" fontWeight="bold" fill="#fff">Wind?</text>

        {/* Level 2 */}
        <line x1="30" y1="94" x2="20" y2="135" stroke="#ccc" strokeWidth="1" /><text x="12" y="118" fontSize="7" fill="#666">High</text>
        <line x1="70" y1="94" x2="80" y2="135" stroke="#ccc" strokeWidth="1" /><text x="85" y="118" fontSize="7" fill="#666">Normal</text>
        <line x1="250" y1="94" x2="240" y2="135" stroke="#ccc" strokeWidth="1" /><text x="233" y="118" fontSize="7" fill="#666">Strong</text>
        <line x1="290" y1="94" x2="300" y2="135" stroke="#ccc" strokeWidth="1" /><text x="306" y="118" fontSize="7" fill="#666">Weak</text>

        <rect x="2" y="135" width="38" height="20" rx="4" fill="#E8483F" /><text x="21" y="149" textAnchor="middle" fontSize="8" fontWeight="bold" fill="#fff">No</text>
        <rect x="62" y="135" width="38" height="20" rx="4" fill="#2EAF7D" /><text x="81" y="149" textAnchor="middle" fontSize="8" fontWeight="bold" fill="#fff">Yes</text>
        <rect x="222" y="135" width="38" height="20" rx="4" fill="#E8483F" /><text x="241" y="149" textAnchor="middle" fontSize="8" fontWeight="bold" fill="#fff">No</text>
        <rect x="282" y="135" width="38" height="20" rx="4" fill="#2EAF7D" /><text x="301" y="149" textAnchor="middle" fontSize="8" fontWeight="bold" fill="#fff">Yes</text>
      </svg>
    </div>
  );
}

function NeuralNetViz() {
  const [weights, setWeights] = useState([0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.7, -0.5]);
  const layers = [[50], [150], [250]];
  const layerSizes = [3, 4, 2];
  const getY = (layer, idx) => 30 + idx * (140 / (layerSizes[layer]));
  const randomize = () => setWeights(weights.map(() => (Math.random() * 2 - 1)));

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">Neural network: inputs → hidden layer → outputs</p>
      <svg viewBox="0 0 300 170" className="w-full max-w-xs">
        {/* Connections */}
        {[0, 1, 2].map(i => [0, 1, 2, 3].map(j => (
          <line key={`0-${i}-${j}`} x1={50} y1={getY(0, i)} x2={150} y2={getY(1, j)} stroke="#ddd" strokeWidth="1" opacity={0.5} />
        )))}
        {[0, 1, 2, 3].map(i => [0, 1].map(j => (
          <line key={`1-${i}-${j}`} x1={150} y1={getY(1, i)} x2={250} y2={getY(2, j)} stroke="#ddd" strokeWidth="1" opacity={0.5} />
        )))}
        {/* Neurons */}
        {[0, 1, 2].map(i => <circle key={`in${i}`} cx={50} cy={getY(0, i)} r="12" fill="#4A90D9" stroke="#2563EB" strokeWidth="1.5" />)}
        {[0, 1, 2, 3].map(i => <circle key={`h${i}`} cx={150} cy={getY(1, i)} r="12" fill="#7B61C1" stroke="#5B21B6" strokeWidth="1.5" />)}
        {[0, 1].map(i => <circle key={`o${i}`} cx={250} cy={getY(2, i)} r="12" fill="#E8483F" stroke="#DC2626" strokeWidth="1.5" />)}
        {/* Labels */}
        <text x="50" y="160" textAnchor="middle" fontSize="9" fill="#4A90D9" fontWeight="bold">Input</text>
        <text x="150" y="160" textAnchor="middle" fontSize="9" fill="#7B61C1" fontWeight="bold">Hidden</text>
        <text x="250" y="160" textAnchor="middle" fontSize="9" fill="#E8483F" fontWeight="bold">Output</text>
      </svg>
      <button onClick={randomize} className="mt-1 px-3 py-1 text-xs rounded-full border hover:bg-gray-100">🔀 Randomize Weights</button>
    </div>
  );
}

function DTWViz() {
  const s1 = [1, 2, 4, 7, 6, 3, 2, 1, 2];
  const s2 = [1, 1, 3, 6, 7, 5, 3, 1, 1];
  const scale = (v) => 60 - v * 7;

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">DTW aligns two time series by warping time</p>
      <svg viewBox="0 0 300 130" className="w-full max-w-xs">
        {/* Series 1 */}
        <path d={s1.map((v, i) => `${i === 0 ? 'M' : 'L'}${20 + i * 30},${scale(v)}`).join(' ')} fill="none" stroke="#E8483F" strokeWidth="2" />
        {s1.map((v, i) => <circle key={`s1-${i}`} cx={20 + i * 30} cy={scale(v)} r="3" fill="#E8483F" />)}
        {/* Series 2 */}
        <path d={s2.map((v, i) => `${i === 0 ? 'M' : 'L'}${20 + i * 30},${70 + scale(v)}`).join(' ')} fill="none" stroke="#4A90D9" strokeWidth="2" />
        {s2.map((v, i) => <circle key={`s2-${i}`} cx={20 + i * 30} cy={70 + scale(v)} r="3" fill="#4A90D9" />)}
        {/* Warping lines */}
        {[[0, 0], [1, 1], [2, 2], [3, 4], [4, 3], [5, 5], [6, 6], [7, 7], [8, 8]].map(([i, j], k) => (
          <line key={k} x1={20 + i * 30} y1={scale(s1[i])} x2={20 + j * 30} y2={70 + scale(s2[j])} stroke="#999" strokeWidth="0.8" strokeDasharray="3" />
        ))}
        <text x="5" y="30" fontSize="8" fill="#E8483F" fontWeight="bold">S₁</text>
        <text x="5" y="100" fontSize="8" fill="#4A90D9" fontWeight="bold">S₂</text>
      </svg>
      <p className="text-xs mt-1">Dashed lines show the optimal DTW alignment</p>
    </div>
  );
}

function HMMViz() {
  const [step, setStep] = useState(0);
  const states = ["☀️ Sunny", "🌧 Rainy"];
  const observations = ["Walk", "Shop", "Walk", "Clean"];
  const viterbiPath = [0, 0, 0, 1];
  const curr = Math.min(step, observations.length - 1);

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">HMM: hidden weather states produce observed activities</p>
      <svg viewBox="0 0 300 130" className="w-full max-w-xs">
        {/* States */}
        <circle cx="80" cy="40" r="25" fill={viterbiPath[curr] === 0 ? "#FEF3C7" : "#f5f5f5"} stroke="#F59E0B" strokeWidth="2" />
        <text x="80" y="37" textAnchor="middle" fontSize="14">☀️</text>
        <text x="80" y="50" textAnchor="middle" fontSize="8" fontWeight="bold">Sunny</text>

        <circle cx="220" cy="40" r="25" fill={viterbiPath[curr] === 1 ? "#DBEAFE" : "#f5f5f5"} stroke="#4A90D9" strokeWidth="2" />
        <text x="220" y="37" textAnchor="middle" fontSize="14">🌧</text>
        <text x="220" y="50" textAnchor="middle" fontSize="8" fontWeight="bold">Rainy</text>

        {/* Transitions */}
        <path d="M105,35 Q150,10 195,35" fill="none" stroke="#888" strokeWidth="1" markerEnd="url(#arrowBN)" />
        <path d="M195,45 Q150,70 105,45" fill="none" stroke="#888" strokeWidth="1" markerEnd="url(#arrowBN)" />
        <text x="150" y="16" textAnchor="middle" fontSize="7" fill="#888">0.3</text>
        <text x="150" y="68" textAnchor="middle" fontSize="7" fill="#888">0.4</text>

        {/* Self loops */}
        <path d="M55,30 Q40,5 68,22" fill="none" stroke="#888" strokeWidth="1" />
        <text x="42" y="12" fontSize="7" fill="#888">0.7</text>
        <path d="M245,30 Q260,5 232,22" fill="none" stroke="#888" strokeWidth="1" />
        <text x="253" y="12" fontSize="7" fill="#888">0.6</text>

        {/* Observations */}
        <text x="150" y="95" textAnchor="middle" fontSize="9" fill="#666">Observations:</text>
        {observations.map((o, i) => (
          <g key={i}>
            <rect x={35 + i * 65} y="100" width="50" height="20" rx="4" fill={i <= curr ? "#E8F5E9" : "#f5f5f5"} stroke={i === curr ? "#2EAF7D" : "#ddd"} strokeWidth={i === curr ? 2 : 1} />
            <text x={60 + i * 65} y="114" textAnchor="middle" fontSize="9" fontWeight={i === curr ? "bold" : "normal"}>{o}</text>
          </g>
        ))}
      </svg>
      <p className="text-xs mt-1">Viterbi: Most likely state = <b>{states[viterbiPath[curr]]}</b></p>
      <div className="flex gap-2 mt-1">
        <button onClick={() => setStep(Math.max(0, step - 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <button onClick={() => setStep(Math.min(observations.length - 1, step + 1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function CNNViz() {
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-2 opacity-70">CNN architecture: Input → Convolution → Pooling → FC → Output</p>
      <svg viewBox="0 0 350 120" className="w-full max-w-md">
        {/* Input */}
        <rect x="5" y="20" width="40" height="80" rx="3" fill="#DBEAFE" stroke="#4A90D9" strokeWidth="1.5" />
        <text x="25" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#4A90D9">Input</text>
        {/* Conv */}
        <rect x="65" y="25" width="35" height="70" rx="3" fill="#FDE68A" stroke="#F59E0B" strokeWidth="1.5" />
        <rect x="70" y="30" width="35" height="70" rx="3" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="1.5" />
        <text x="85" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#F59E0B">Conv</text>
        {/* Pool */}
        <rect x="125" y="35" width="25" height="50" rx="3" fill="#D1FAE5" stroke="#2EAF7D" strokeWidth="1.5" />
        <text x="137" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#2EAF7D">Pool</text>
        {/* Conv 2 */}
        <rect x="170" y="35" width="25" height="50" rx="3" fill="#FDE68A" stroke="#F59E0B" strokeWidth="1.5" />
        <rect x="175" y="40" width="25" height="50" rx="3" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="1.5" />
        <text x="185" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#F59E0B">Conv</text>
        {/* Pool 2 */}
        <rect x="220" y="42" width="20" height="36" rx="3" fill="#D1FAE5" stroke="#2EAF7D" strokeWidth="1.5" />
        <text x="230" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#2EAF7D">Pool</text>
        {/* FC */}
        {[0, 1, 2, 3, 4].map(i => <circle key={i} cx="270" cy={30 + i * 15} r="5" fill="#EDE9FE" stroke="#7B61C1" strokeWidth="1" />)}
        <text x="270" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#7B61C1">FC</text>
        {/* Output */}
        {[0, 1].map(i => <circle key={i} cx="320" cy={45 + i * 30} r="8" fill="#FECACA" stroke="#E8483F" strokeWidth="1.5" />)}
        <text x="320" y="115" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#E8483F">Out</text>
        {/* Arrows */}
        {[[47, 65], [107, 125], [152, 170], [202, 220], [242, 260], [280, 310]].map(([x1, x2], i) => (
          <line key={i} x1={x1} y1="60" x2={x2} y2="60" stroke="#bbb" strokeWidth="1" markerEnd="url(#arrowBN)" />
        ))}
      </svg>
    </div>
  );
}

function MDPViz() {
  const [iteration, setIteration] = useState(0);
  const grid = 3;
  const rewards = [[0, 0, 1], [0, -10, 0], [0, 0, 0]];
  const gamma = 0.9;
  const labels = [["", "", "🎯"], ["", "💀", ""], ["🏠", "", ""]];

  const computeValues = (iters) => {
    let V = Array.from({ length: grid }, () => Array(grid).fill(0));
    for (let t = 0; t < iters; t++) {
      const newV = V.map(r => [...r]);
      for (let r = 0; r < grid; r++) {
        for (let c = 0; c < grid; c++) {
          if (r === 0 && c === 2) { newV[r][c] = 1; continue; }
          if (r === 1 && c === 1) { newV[r][c] = -10; continue; }
          const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]].filter(([nr, nc]) => nr >= 0 && nr < grid && nc >= 0 && nc < grid);
          const best = Math.max(...neighbors.map(([nr, nc]) => rewards[r][c] + gamma * V[nr][nc]));
          newV[r][c] = neighbors.length > 0 ? best : rewards[r][c];
        }
      }
      V = newV;
    }
    return V;
  };

  const values = computeValues(iteration);

  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-70">MDP Grid World: Value Iteration. 🎯=Goal, 💀=Penalty</p>
      <div className="grid grid-cols-3 gap-1 mb-2" style={{ width: 180 }}>
        {values.flat().map((v, i) => {
          const r = Math.floor(i / 3), c = i % 3;
          const bg = v > 0.5 ? `rgba(46,175,125,${Math.min(v, 1) * 0.5})` : v < -0.5 ? `rgba(232,72,63,${Math.min(-v / 10, 1) * 0.5})` : "#f9fafb";
          return (
            <div key={i} className="flex flex-col items-center justify-center border border-gray-200 rounded" style={{ width: 56, height: 56, background: bg }}>
              <span className="text-sm">{labels[r][c]}</span>
              <span className="text-xs font-mono font-bold">{v.toFixed(1)}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-2">
        <button onClick={() => setIteration(Math.max(0, iteration - 1))} className="px-2 py-1 text-xs rounded-full border hover:bg-gray-100">−</button>
        <span className="text-xs font-mono">Iteration: {iteration}</span>
        <button onClick={() => setIteration(iteration + 1)} className="px-2 py-1 text-xs rounded-full border hover:bg-gray-100">+</button>
      </div>
    </div>
  );
}

// ============== VIZ LOOKUP ==============
const vizComponents = {
  "tree-search": TreeSearchViz,
  "bfs": BFSViz,
  "ucs": UCSViz,
  "astar": AStarViz,
  "hill-climbing": HillClimbingViz,
  "annealing": AnnealingViz,
  "genetic": GeneticViz,
  "minimax": MinimaxViz,
  "alpha-beta": AlphaBetaViz,
  "csp-map": CSPMapViz,
  "backtracking": BacktrackingViz,
  "probability": ProbabilityViz,
  "bayes": BayesViz,
  "bayesnet": BayesNetViz,
  "dsep": DSepViz,
  "knn": KNNViz,
  "gaussian": GaussianViz,
  "decision-tree": DecisionTreeViz,
  "neural-net": NeuralNetViz,
  "dtw": DTWViz,
  "hmm": HMMViz,
  "cnn": CNNViz,
  "mdp": MDPViz,
};

// ============== MAIN APP ==============
export default function CS6601App() {
  const [activeModule, setActiveModule] = useState(0);
  const [activeSection, setActiveSection] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showQuiz, setShowQuiz] = useState(false);
  const [quizKey, setQuizKey] = useState(0);
  const contentRef = useRef(null);

  const mod = MODULES[activeModule];
  const sec = mod.sections[activeSection];
  const VizComp = sec.viz ? vizComponents[sec.viz] : null;
  const hasQuiz = (QUIZZES[activeModule] || []).length > 0;

  const goNext = () => {
    if (showQuiz) {
      setShowQuiz(false);
      if (activeModule < MODULES.length - 1) {
        setActiveModule(activeModule + 1);
        setActiveSection(0);
      }
      return;
    }

    if (activeSection < mod.sections.length - 1) {
      setActiveSection(activeSection + 1);
    } else if (hasQuiz) {
      setShowQuiz(true);
      setQuizKey(k => k + 1);
    } else if (activeModule < MODULES.length - 1) {
      setActiveModule(activeModule + 1);
      setActiveSection(0);
    }
  };

  const goPrev = () => {
    if (showQuiz) {
      setShowQuiz(false);
      return;
    }

    if (activeSection > 0) {
      setActiveSection(activeSection - 1);
    } else if (activeModule > 0) {
      setActiveModule(activeModule - 1);
      setActiveSection(MODULES[activeModule - 1].sections.length - 1);
    }
  };

  useEffect(() => {
    if (contentRef.current) contentRef.current.scrollTop = 0;
  }, [activeModule, activeSection, showQuiz]);

  const totalSections = MODULES.reduce((sum, m) => sum + m.sections.length, 0);
  let currentIdx = 0;
  for (let i = 0; i < activeModule; i++) currentIdx += MODULES[i].sections.length;
  currentIdx += activeSection;
  const isAtEnd = activeModule === MODULES.length - 1 && (showQuiz || (!hasQuiz && activeSection === mod.sections.length - 1));

  const renderMarkdown = (text) => {
    return text.split('\n').map((line, i) => {
      line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      line = line.replace(/\*(.+?)\*/g, '<em>$1</em>');
      if (line.startsWith('•')) {
        return <div key={i} className="pl-3 py-0.5 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: line }} />;
      }
      if (line.trim() === '') return <div key={i} className="h-2" />;
      return <p key={i} className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: line }} />;
    });
  };

  return (
    <div style={{ fontFamily: "'Crimson Pro', 'Georgia', serif", background: "#FAFAF8", minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)", color: "#fff", padding: "16px 20px", display: "flex", alignItems: "center", gap: 12 }}>
        <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-1 rounded hover:bg-white/10" style={{ fontSize: 18 }}>☰</button>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.5px", fontFamily: "'JetBrains Mono', monospace" }}>CS6601</h1>
          <p style={{ fontSize: 11, opacity: 0.7, fontFamily: "sans-serif" }}>Artificial Intelligence — Interactive Guide</p>
        </div>
        <div style={{ marginLeft: "auto", fontSize: 11, opacity: 0.6, fontFamily: "sans-serif" }}>{currentIdx + 1}/{totalSections}</div>
      </div>

      {/* Progress bar */}
      <div style={{ height: 3, background: "#e5e7eb" }}>
        <div style={{ height: "100%", width: `${(currentIdx + 1) / totalSections * 100}%`, background: mod.color, transition: "all 0.3s" }} />
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Sidebar */}
        {sidebarOpen && (
          <div style={{ width: 220, minWidth: 220, borderRight: "1px solid #e5e7eb", background: "#fff", overflowY: "auto", padding: "8px 0" }}>
            {MODULES.map((m, mi) => (
              <div key={m.id}>
                <button
                  onClick={() => { setActiveModule(mi); setActiveSection(0); setShowQuiz(false); }}
                  style={{
                    width: "100%", textAlign: "left", padding: "8px 12px", border: "none", cursor: "pointer",
                    background: activeModule === mi ? `${m.color}10` : "transparent",
                    borderLeft: activeModule === mi ? `3px solid ${m.color}` : "3px solid transparent",
                    fontFamily: "sans-serif", fontSize: 12, fontWeight: activeModule === mi ? 700 : 500,
                    color: activeModule === mi ? m.color : "#555",
                    display: "flex", alignItems: "center", gap: 8
                  }}
                >
                  <span style={{ fontSize: 16 }}>{m.icon}</span>
                  <span>{m.id}. {m.title}</span>
                </button>
                {activeModule === mi && m.sections.map((s, si) => (
                  <button
                    key={si}
                    onClick={() => { setActiveSection(si); setShowQuiz(false); }}
                    style={{
                      width: "100%", textAlign: "left", padding: "4px 12px 4px 36px", border: "none", cursor: "pointer",
                      background: !showQuiz && activeSection === si ? `${m.color}18` : "transparent",
                      fontFamily: "sans-serif", fontSize: 11, color: !showQuiz && activeSection === si ? m.color : "#888",
                      fontWeight: !showQuiz && activeSection === si ? 600 : 400
                    }}
                  >
                    {s.title}
                  </button>
                ))}
                {activeModule === mi && hasQuiz && (
                  <button
                    onClick={() => { setShowQuiz(true); setQuizKey(k => k + 1); }}
                    style={{
                      width: "100%", textAlign: "left", padding: "4px 12px 4px 36px", border: "none", cursor: "pointer",
                      background: showQuiz ? `${m.color}18` : "transparent",
                      fontFamily: "sans-serif", fontSize: 11, color: showQuiz ? m.color : "#888",
                      fontWeight: showQuiz ? 600 : 400
                    }}
                  >
                    📝 Quiz
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Main content */}
        <div ref={contentRef} style={{ flex: 1, overflowY: "auto", padding: "24px 32px", maxWidth: 720 }}>
          {/* Module badge */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 24 }}>{mod.icon}</span>
            <span style={{ fontSize: 11, fontWeight: 700, color: mod.color, textTransform: "uppercase", letterSpacing: 1.5, fontFamily: "sans-serif" }}>Module {mod.id}: {mod.title}</span>
          </div>

          {showQuiz ? (
            <>
              <h2 style={{ fontSize: 26, fontWeight: 700, color: "#1a1a2e", marginBottom: 4, lineHeight: 1.2, letterSpacing: "-0.5px" }}>📝 Module {mod.id} Quiz</h2>
              <p className="text-sm text-gray-500 mb-6 font-sans">{(QUIZZES[activeModule] || []).length} questions</p>
              <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: 20, boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
                <Quiz key={quizKey} modIdx={activeModule} color={mod.color} />
              </div>
            </>
          ) : (
            <>
              {/* Section title */}
              <h2 style={{ fontSize: 26, fontWeight: 700, color: "#1a1a2e", marginBottom: 16, lineHeight: 1.2, letterSpacing: "-0.5px" }}>{sec.title}</h2>

              {/* Content */}
              <div style={{ color: "#374151", marginBottom: 20 }}>
                {renderMarkdown(sec.content)}
              </div>

              {/* Visualization */}
              {VizComp && (
                <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, marginBottom: 20, boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: mod.color, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, fontFamily: "sans-serif" }}>
                    ▶ Interactive Visualization
                  </div>
                  <VizComp />
                </div>
              )}

              {/* Quiz Teaser */}
              {activeSection === mod.sections.length - 1 && hasQuiz && (
                <div className="flex items-center gap-3 p-4 rounded-xl mb-4" style={{ background: `${mod.color}08`, border: `1px dashed ${mod.color}40` }}>
                  <span className="text-2xl">📝</span>
                  <div className="flex-1">
                    <p className="text-sm font-semibold" style={{ color: mod.color }}>Ready to test your knowledge?</p>
                    <p className="text-xs text-gray-500">{(QUIZZES[activeModule] || []).length} questions</p>
                  </div>
                  <button onClick={() => { setShowQuiz(true); setQuizKey(k => k + 1); }} className="px-4 py-2 rounded-lg text-sm font-semibold text-white" style={{ background: mod.color }}>Take Quiz →</button>
                </div>
              )}
            </>
          )}

          {/* Navigation */}
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 24, paddingTop: 16, borderTop: "1px solid #e5e7eb" }}>
            <button
              onClick={goPrev}
              disabled={activeModule === 0 && activeSection === 0 && !showQuiz}
              style={{
                padding: "8px 16px", borderRadius: 8, border: "1px solid #ddd", background: "#fff",
                cursor: activeModule === 0 && activeSection === 0 && !showQuiz ? "not-allowed" : "pointer",
                opacity: activeModule === 0 && activeSection === 0 && !showQuiz ? 0.3 : 1,
                fontSize: 13, fontFamily: "sans-serif"
              }}
            >
              ← Previous
            </button>
            <button
              onClick={goNext}
              disabled={isAtEnd}
              style={{
                padding: "8px 16px", borderRadius: 8, border: "none", background: mod.color, color: "#fff",
                cursor: isAtEnd ? "not-allowed" : "pointer",
                opacity: isAtEnd ? 0.5 : 1,
                fontSize: 13, fontWeight: 600, fontFamily: "sans-serif"
              }}>
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
