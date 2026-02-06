import { useState, useEffect, useRef } from "react";

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

const MODS = [
  { t: "Search", i: "🔍", c: "#E8483F", s: [
    { t: "What is Search?", txt: "Search finds a path from a **start state** to a **goal state**. Like navigating a city.\n\n**Key components:** Initial State, Actions, Transition Model, Goal Test, Path Cost.\n\n**Tree Search** expands nodes from the frontier. **Graph Search** adds an explored set to avoid revisiting states.", v: "tree" },
    { t: "BFS & UCS", txt: "**BFS** explores level by level using a **queue (FIFO)**. Complete, optimal for equal costs. Time/Space: O(b^d).\n\n**UCS** expands the **cheapest** node using a **priority queue**. Equivalent to Dijkstra's. Guaranteed optimal for any non-negative edge costs.\n\nBFS = UCS when all step costs are equal.", v: "bfs" },
    { t: "A* Search", txt: "A* uses **f(n) = g(n) + h(n)**\n\n• **g(n)** = actual cost from start to n\n• **h(n)** = estimated cost from n to goal (heuristic)\n\n**Admissible:** h(n) never overestimates → A* is optimal\n**Consistent:** h(n) ≤ cost(n→n') + h(n') → never re-expand nodes\n\nA* is **optimally efficient** — no other optimal algorithm expands fewer nodes.", v: "astar" },
  ]},
  { t: "Simulated Annealing", i: "🌡️", c: "#E07B39", s: [
    { t: "Hill Climbing", txt: "**Hill Climbing:** always move to a better neighbor. Simple but gets stuck at:\n\n• **Local maxima** — small hill, not the mountain\n• **Plateaus** — flat areas, no progress\n• **Ridges** — narrow peaks\n\n**Random Restart** helps: try many starting points.", v: "hill" },
    { t: "Simulated Annealing & GAs", txt: "**Simulated Annealing** (from metallurgy):\n1. High temperature T → accept bad moves freely\n2. Pick random neighbor. Better? Accept. Worse? Accept with probability **e^(ΔE/T)**\n3. Cool down gradually\n\n**Genetic Algorithms** (from evolution):\n1. **Selection** — fitter individuals reproduce\n2. **Crossover** — combine two parents\n3. **Mutation** — random changes\n\nBoth escape local optima through randomness.", v: "anneal" },
  ]},
  { t: "Game Playing", i: "♟️", c: "#4A90D9", s: [
    { t: "Minimax", txt: "Two-player zero-sum games. Both players play optimally.\n\n• **MAX** picks child with highest value\n• **MIN** picks child with lowest value\n• Build game tree, propagate values bottom-up\n\nTime: O(b^m). Too expensive for real games!", v: "mini" },
    { t: "Alpha-Beta Pruning", txt: "Skip branches that can't affect the decision.\n\n• **α** = best MAX can guarantee (starts -∞)\n• **β** = best MIN can guarantee (starts +∞)\n• **Prune when α ≥ β**\n\nBest case: O(b^(m/2)) — doubles search depth!\n\n**Other concepts:** Iterative Deepening (always have a move ready), Quiescent Search (don't evaluate noisy positions), Expectimax (chance nodes for dice/cards).", v: "ab" },
  ]},
  { t: "Constraint Satisfaction", i: "🧩", c: "#7B61C1", s: [
    { t: "CSP Fundamentals", txt: "**Variables** + **Domains** + **Constraints**\n\nExample: Map Coloring — color regions so no adjacent pair matches.\n\n**Backtracking:** assign one variable at a time, undo on violation.\n\n**Improvements:**\n• **Forward Checking** — remove inconsistent neighbor values early\n• **Arc Consistency (AC-3)** — every value has a compatible partner\n• **MRV** — pick most constrained variable first\n• **LCV** — try least constraining value first", v: "csp" },
  ]},
  { t: "Probability", i: "🎲", c: "#2EAF7D", s: [
    { t: "Probability & Bayes' Rule", txt: "**P(A)** ∈ [0,1], **P(¬A) = 1 - P(A)**\n\n**Conditional:** P(A|B) = P(A∧B) / P(B)\n**Independence:** P(A|B) = P(A)\n\n**Bayes' Rule: P(A|B) = P(B|A) × P(A) / P(B)**\n\nExample: 1% cancer rate, 90% test sensitivity, 20% false positive.\nP(Cancer|Positive) ≈ **4.3%** — base rate matters!", v: "bayes" },
  ]},
  { t: "Bayes Nets", i: "🕸️", c: "#1A8A8A", s: [
    { t: "Structure & D-Separation", txt: "DAG where nodes = variables, edges = dependencies. Each node has a CPT.\n\n**D-Separation patterns:**\n• **Chain A→B→C:** A⊥C|B (B blocks flow)\n• **Fork A←B→C:** A⊥C|B (common cause)\n• **Collider A→B←C:** A⊥C normally, but A⊥̸C|B (explaining away!)\n\n**Inference:** Enumeration, Variable Elimination, or approximate methods (Rejection Sampling, Likelihood Weighting, Gibbs Sampling).", v: "bn" },
  ]},
  { t: "Machine Learning", i: "🤖", c: "#D94E8F", s: [
    { t: "kNN & Naive Bayes", txt: "**kNN:** Classify by majority vote of k nearest neighbors. Simple but slow at prediction. Larger k = smoother boundary.\n\n**Gaussian Classifier:** Each class is a bell curve. Decision boundary where curves cross.\n\n**Naive Bayes:** Features are conditionally independent given class.\nP(Class|Features) ∝ P(Class) × Π P(Feature_i|Class)\nWorks surprisingly well despite the 'naive' assumption!", v: "knn" },
    { t: "Trees & Neural Nets", txt: "**Decision Trees:** Split on features using **Information Gain** (entropy reduction). Avoid overfitting with pruning, **Random Forests**, or **Boosting**.\n\n**Neural Networks:**\n• Perceptron: output = activation(Σ wᵢxᵢ + bias)\n• Single perceptron can't learn XOR!\n• **Multilayer + Backpropagation** → complex boundaries\n• **Deep Learning:** many layers learn hierarchical features", v: "nn" },
  ]},
  { t: "Pattern Recognition", i: "📈", c: "#8B6914", s: [
    { t: "DTW & HMMs", txt: "**Dynamic Time Warping:** Aligns time series by warping time. Euclidean distance fails with shifts; DTW finds optimal alignment.\n\n**Hidden Markov Models:** Hidden states produce observations.\n• **Transition probs:** P(next state|current)\n• **Emission probs:** P(observation|state)\n\n**Three problems:**\n1. Evaluation → Forward algorithm\n2. Decoding → **Viterbi** (most likely state sequence)\n3. Learning → **Baum-Welch** (EM)", v: "hmm" },
  ]},
  { t: "Deep Learning", i: "🧠", c: "#5B3E96", s: [
    { t: "CNNs & Modern DL", txt: "**CNNs:** Convolutional layers (local patterns) → Pooling (downsample) → FC (classify). Parameter sharing = fewer params.\n\n**ResNets:** Skip connections enable 100+ layer networks.\n\n**Regularization:** L2, Dropout, Batch Normalization.\n\n**NLP:** Word2Vec embeddings, RNNs/LSTMs, Attention/Transformers, Transfer Learning.\n\n**Responsible AI:** Transparency, interpretability, fairness.", v: "cnn" },
  ]},
  { t: "Planning under Uncertainty", i: "🎯", c: "#C74D3C", s: [
    { t: "MDPs & POMDPs", txt: "**MDP:** States, Actions, Transitions T(s,a,s'), Rewards R(s), Discount γ.\n\n**Value Iteration:** V(s) = max_a [R(s) + γ Σ T(s,a,s')V(s')]\nConverges to optimal policy.\n\n**POMDP:** Can't observe state directly! Maintain **belief state** (probability distribution). Much harder (PSPACE-complete).\n\nKey insight: sometimes the best action **gathers information** rather than directly pursuing the goal.", v: "mdp" },
  ]},
];

// ====== VISUALIZATIONS ======
function TreeViz() {
  const [exp, setExp] = useState(new Set(["S"]));
  const ns = { S:{x:200,y:30,ch:["A","B"]}, A:{x:100,y:100,ch:["C","D"]}, B:{x:300,y:100,ch:["E","G"]}, C:{x:50,y:170,ch:[]}, D:{x:150,y:170,ch:[]}, E:{x:250,y:170,ch:[]}, G:{x:350,y:170,ch:[]} };
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-2 opacity-60">Click nodes to expand</p>
      <svg viewBox="0 0 400 200" className="w-full max-w-sm">
        {Object.entries(ns).map(([id,n]) => n.ch.map(cid => exp.has(id) ? <line key={`${id}-${cid}`} x1={n.x} y1={n.y+14} x2={ns[cid].x} y2={ns[cid].y-14} stroke={exp.has(cid)?"#666":"#ddd"} strokeWidth="2"/> : null))}
        {Object.entries(ns).map(([id,n]) => {
          const isE = exp.has(id), isG = id==="G"&&isE;
          const canClick = !isE && Object.entries(ns).some(([p,pn])=>exp.has(p)&&pn.ch.includes(id));
          return (<g key={id} onClick={()=>canClick&&setExp(new Set([...exp,id]))} style={{cursor:canClick?"pointer":"default"}}>
            <circle cx={n.x} cy={n.y} r="16" fill={isG?"#2EAF7D":isE?"#E8483F":canClick?"#fff":"#eee"} stroke={canClick?"#E8483F":"#ccc"} strokeWidth="2" strokeDasharray={canClick?"4":"0"}/>
            <text x={n.x} y={n.y+5} textAnchor="middle" fontSize="12" fontWeight="bold" fill={isE?"#fff":"#333"}>{id}</text>
          </g>);
        })}
      </svg>
      {exp.has("G") && <p className="text-sm font-bold mt-1" style={{color:"#2EAF7D"}}>🎯 Goal G found!</p>}
      <button onClick={()=>setExp(new Set(["S"]))} className="mt-2 px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Reset</button>
    </div>
  );
}

function BFSViz() {
  const [s, setS] = useState(0);
  const order = ["S","A","B","C","D","G"];
  const pos = {S:{x:40,y:80},A:{x:140,y:30},B:{x:140,y:130},C:{x:240,y:30},D:{x:240,y:130},G:{x:320,y:80}};
  const edges = [["S","A"],["S","B"],["A","C"],["B","D"],["C","G"],["D","G"]];
  const visited = new Set(order.slice(0, s));
  const cur = order[s-1];
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 360 160" className="w-full max-w-sm">
        {edges.map(([a,b])=><line key={a+b} x1={pos[a].x} y1={pos[a].y} x2={pos[b].x} y2={pos[b].y} stroke={visited.has(a)&&visited.has(b)?"#E8483F":"#ddd"} strokeWidth="2"/>)}
        {Object.entries(pos).map(([id,p])=>(
          <g key={id}><circle cx={p.x} cy={p.y} r="18" fill={id===cur?"#E8483F":visited.has(id)?"#FECACA":"#f3f4f6"} stroke={visited.has(id)?"#E8483F":"#ccc"} strokeWidth="2"/>
          <text x={p.x} y={p.y+5} textAnchor="middle" fontSize="12" fontWeight="bold" fill={id===cur?"#fff":"#333"}>{id}</text></g>
        ))}
      </svg>
      <div className="flex gap-2 mt-1">
        <button onClick={()=>setS(Math.max(0,s-1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <span className="text-xs py-1">Step {s}/{order.length}</span>
        <button onClick={()=>setS(Math.min(order.length,s+1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function AStarViz() {
  const [s, setS] = useState(0);
  const nodes = [{id:"S",x:40,y:80,h:6},{id:"A",x:140,y:30,h:4},{id:"B",x:140,y:130,h:5},{id:"C",x:240,y:30,h:2},{id:"G",x:320,y:80,h:0}];
  const steps = [{cur:"S",f:"0+6=6",path:["S"]},{cur:"A",f:"2+4=6",path:["S","A"]},{cur:"C",f:"5+2=7",path:["S","A","C"]},{cur:"G",f:"7+0=7",path:["S","A","C","G"]}];
  const st = steps[Math.min(s, steps.length-1)];
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 360 160" className="w-full max-w-sm">
        {[["S","A",2],["S","B",3],["A","C",3],["B","G",6],["C","G",2]].map(([a,b,c])=>{
          const na=nodes.find(n=>n.id===a),nb=nodes.find(n=>n.id===b);
          const active=st.path.includes(a)&&st.path.includes(b);
          return <g key={a+b}><line x1={na.x} y1={na.y} x2={nb.x} y2={nb.y} stroke={active?"#E8483F":"#ddd"} strokeWidth="2"/><text x={(na.x+nb.x)/2+8} y={(na.y+nb.y)/2-4} fontSize="10" fill="#999">{c}</text></g>;
        })}
        {nodes.map(n=>(
          <g key={n.id}><circle cx={n.x} cy={n.y} r="20" fill={n.id===st.cur?"#E8483F":st.path.includes(n.id)?"#FECACA":"#f3f4f6"} stroke={st.path.includes(n.id)?"#E8483F":"#ccc"} strokeWidth="2"/>
          <text x={n.x} y={n.y} textAnchor="middle" fontSize="11" fontWeight="bold" fill={n.id===st.cur?"#fff":"#333"}>{n.id}</text>
          <text x={n.x} y={n.y+11} textAnchor="middle" fontSize="7" fill={n.id===st.cur?"#fbb":"#aaa"}>h={n.h}</text></g>
        ))}
      </svg>
      <p className="text-xs mt-1 font-mono">f = {st.f} | {st.path.join("→")}{s>=3?" ✓":""}</p>
      <div className="flex gap-2 mt-1">
        <button onClick={()=>setS(Math.max(0,s-1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <button onClick={()=>setS(Math.min(steps.length-1,s+1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function HillViz() {
  const [pos, setPos] = useState(30);
  const fn = x => 30*Math.sin(x*0.08)+15*Math.sin(x*0.15+1)+8*Math.sin(x*0.3+2);
  const y = x => 100 - fn(x);
  const d = Array.from({length:60},(_,i)=>`${i===0?'M':'L'}${i*5},${y(i*5)}`).join(' ');
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 130" className="w-full max-w-sm">
        <path d={d+" L300,130 L0,130 Z"} fill="#E8F5E9" stroke="#2EAF7D" strokeWidth="1.5"/>
        <circle cx={pos} cy={y(pos)} r="6" fill="#E8483F" stroke="#fff" strokeWidth="2"/>
      </svg>
      <div className="flex gap-2 mt-2">
        <button onClick={()=>{if(fn(pos-5)>fn(pos))setPos(pos-5)}} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">← Climb</button>
        <button onClick={()=>{if(fn(pos+5)>fn(pos))setPos(pos+5)}} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Climb →</button>
        <button onClick={()=>setPos(Math.floor(Math.random()*250)+20)} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">🔄 Restart</button>
      </div>
    </div>
  );
}

function AnnealViz() {
  const [temp,setTemp]=useState(100);const [pos,setPos]=useState(60);const [run,setRun]=useState(false);
  const fn=x=>30*Math.sin(x*0.08)+15*Math.sin(x*0.15+1)+8*Math.sin(x*0.3+2);
  const y=x=>100-fn(x);
  useEffect(()=>{if(!run)return;const id=setInterval(()=>{setTemp(t=>{if(t<=1){setRun(false);return 1;}return t*0.97;});setPos(p=>{const np=Math.max(5,Math.min(290,p+(Math.random()-0.5)*20));const d=fn(np)-fn(p);return(d>0||Math.random()<Math.exp(d/(temp*0.3)))?np:p;});},80);return()=>clearInterval(id);},[run,temp]);
  const d=Array.from({length:60},(_,i)=>`${i===0?'M':'L'}${i*5},${y(i*5)}`).join(' ');
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1">🌡️ Temperature: <b style={{color:`hsl(${120-temp*1.2},80%,40%)`}}>{temp.toFixed(0)}</b></p>
      <svg viewBox="0 0 300 130" className="w-full max-w-sm">
        <path d={d+" L300,130 L0,130 Z"} fill="#FFF3E0" stroke="#E07B39" strokeWidth="1.5"/>
        <circle cx={pos} cy={y(pos)} r="6" fill="#E07B39" stroke="#fff" strokeWidth="2"/>
      </svg>
      <div className="flex gap-2 mt-2">
        <button onClick={()=>{setRun(true);setTemp(100);}} disabled={run} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">{run?"Running...":"▶ Start"}</button>
        <button onClick={()=>{setRun(false);setTemp(100);setPos(60);}} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">Reset</button>
      </div>
    </div>
  );
}

function MiniViz() {
  const [show,setShow]=useState(false);
  const leaves=[3,12,8,2,4,6,14,5,2];
  const mins=[Math.min(leaves[0],leaves[1],leaves[2]),Math.min(leaves[3],leaves[4],leaves[5]),Math.min(leaves[6],leaves[7],leaves[8])];
  const root=Math.max(...mins);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 360 170" className="w-full max-w-md">
        {[0,1,2].map(i=><line key={`r${i}`} x1={180} y1={28} x2={60+i*120} y2={72} stroke="#ccc" strokeWidth="1.5"/>)}
        {[0,1,2].map(i=>[0,1,2].map(j=><line key={`m${i}${j}`} x1={60+i*120} y1={78} x2={20+i*120+j*40} y2={135} stroke="#eee" strokeWidth="1"/>))}
        <polygon points="162,8 198,8 190,32 170,32" fill={show?"#E8483F":"#f5f5f5"} stroke="#E8483F" strokeWidth="2"/>
        <text x="180" y="24" textAnchor="middle" fontSize="10" fontWeight="bold" fill={show?"#fff":"#E8483F"}>MAX{show?`=${root}`:""}</text>
        {mins.map((v,i)=>(<g key={i}><circle cx={60+i*120} cy="75" r="16" fill={show?"#4A90D9":"#f5f5f5"} stroke="#4A90D9" strokeWidth="2"/><text x={60+i*120} y="79" textAnchor="middle" fontSize="9" fontWeight="bold" fill={show?"#fff":"#4A90D9"}>MIN{show?`=${v}`:""}</text></g>))}
        {leaves.map((v,i)=>(<g key={i}><rect x={14+Math.floor(i/3)*120+(i%3)*40} y="130" width="26" height="20" rx="4" fill="#FEF3C7" stroke="#F59E0B"/><text x={27+Math.floor(i/3)*120+(i%3)*40} y="144" textAnchor="middle" fontSize="10" fontWeight="bold">{v}</text></g>))}
      </svg>
      <button onClick={()=>setShow(!show)} className="px-4 py-1 text-xs rounded-full border hover:bg-gray-100">{show?"Hide":"Show"} Values</button>
    </div>
  );
}

function ABViz() {
  const [s,setS]=useState(0);
  const info=[{d:"Start: α=-∞, β=+∞",p:false},{d:"Leaf 3 → left MIN β=3",p:false},{d:"Leaf 5 → MIN stays 3",p:false},{d:"MAX sets α=3",p:false},{d:"Leaf 2 → right MIN β=2, α=3 ≥ β=2 → PRUNE! ✂️",p:true},{d:"Result: MAX = 3",p:true}];
  const st=info[Math.min(s,info.length-1)];
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 280 140" className="w-full max-w-xs">
        <line x1="140" y1="25" x2="70" y2="55" stroke="#ccc" strokeWidth="1.5"/>
        <line x1="140" y1="25" x2="210" y2="55" stroke={st.p?"#E8483F":"#ccc"} strokeWidth="1.5" strokeDasharray={st.p?"4":"0"}/>
        <line x1="70" y1="55" x2="40" y2="105" stroke="#ccc" strokeWidth="1"/><line x1="70" y1="55" x2="100" y2="105" stroke="#ccc" strokeWidth="1"/>
        <line x1="210" y1="55" x2="180" y2="105" stroke={st.p?"#fcc":"#ccc"} strokeWidth="1"/><line x1="210" y1="55" x2="240" y2="105" stroke={st.p?"#fcc":"#ccc"} strokeWidth="1"/>
        <polygon points="125,8 155,8 148,28 132,28" fill={s>=3?"#E8483F":"#f5f5f5"} stroke="#E8483F" strokeWidth="2"/>
        <text x="140" y="22" textAnchor="middle" fontSize="9" fontWeight="bold" fill={s>=3?"#fff":"#E8483F"}>MAX</text>
        <circle cx="70" cy="55" r="13" fill={s>=2?"#4A90D9":"#f5f5f5"} stroke="#4A90D9" strokeWidth="2"/>
        <text x="70" y="59" textAnchor="middle" fontSize="9" fontWeight="bold" fill={s>=2?"#fff":"#4A90D9"}>{s>=2?"3":"MIN"}</text>
        <circle cx="210" cy="55" r="13" fill={st.p?"#FEE":"#f5f5f5"} stroke={st.p?"#E8483F":"#4A90D9"} strokeWidth="2"/>
        <text x="210" y="59" textAnchor="middle" fontSize="9" fontWeight="bold" fill={st.p?"#E8483F":"#4A90D9"}>{st.p?"✂":"MIN"}</text>
        {[{v:3,x:40},{v:5,x:100},{v:2,x:180},{v:8,x:240}].map((l,i)=>(<g key={i} opacity={i>=2&&st.p?0.3:1}><rect x={l.x-11} y="100" width="22" height="18" rx="3" fill="#FEF3C7" stroke="#F59E0B"/><text x={l.x} y="113" textAnchor="middle" fontSize="10" fontWeight="bold">{l.v}</text></g>))}
      </svg>
      <p className="text-xs font-medium mt-1 text-center">{st.d}</p>
      <div className="flex gap-2 mt-1">
        <button onClick={()=>setS(Math.max(0,s-1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button>
        <button onClick={()=>setS(Math.min(info.length-1,s+1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button>
      </div>
    </div>
  );
}

function CSPViz() {
  const [colors,setColors]=useState({WA:null,NT:null,SA:null,Q:null,NSW:null,V:null,T:null});
  const pal=["#E8483F","#2EAF7D","#4A90D9"];
  const adj={WA:["NT","SA"],NT:["WA","SA","Q"],SA:["WA","NT","Q","NSW","V"],Q:["NT","SA","NSW"],NSW:["Q","SA","V"],V:["SA","NSW"],T:[]};
  const pos={WA:{x:60,y:70},NT:{x:150,y:30},SA:{x:150,y:100},Q:{x:240,y:30},NSW:{x:240,y:90},V:{x:220,y:140},T:{x:230,y:180}};
  const bad=r=>colors[r]!==null&&adj[r].some(n=>colors[n]===colors[r]);
  const cycle=r=>{const c=colors[r];setColors({...colors,[r]:c===null?0:(c+1)%4>=3?null:(c+1)%4});};
  const done=Object.values(colors).every(c=>c!==null)&&Object.keys(colors).every(r=>!bad(r));
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs mb-1 opacity-60">Click to cycle colors. No adjacent same colors!</p>
      <svg viewBox="0 0 300 200" className="w-full max-w-xs">
        {Object.entries(adj).map(([r,ns])=>ns.map(n=><line key={r+n} x1={pos[r].x} y1={pos[r].y} x2={pos[n].x} y2={pos[n].y} stroke={colors[r]!==null&&colors[n]!==null&&colors[r]===colors[n]?"#E8483F":"#ddd"} strokeWidth="1.5"/>))}
        {Object.entries(pos).map(([r,p])=>(<g key={r} onClick={()=>cycle(r)} style={{cursor:"pointer"}}><circle cx={p.x} cy={p.y} r="20" fill={colors[r]!==null?pal[colors[r]]:"#f3f4f6"} stroke={bad(r)?"#E8483F":"#999"} strokeWidth={bad(r)?3:1.5}/><text x={p.x} y={p.y+4} textAnchor="middle" fontSize="10" fontWeight="bold" fill={colors[r]!==null?"#fff":"#333"}>{r}</text></g>))}
      </svg>
      {done&&<p className="text-sm font-bold" style={{color:"#2EAF7D"}}>✅ Valid!</p>}
    </div>
  );
}

function BayesViz() {
  const [pr,setPr]=useState(1);const [se,setSe]=useState(90);const [fp,setFp]=useState(20);
  const pD=pr/100,pPD=se/100,pPN=fp/100,pP=pPD*pD+pPN*(1-pD);
  const post=pP>0?(pPD*pD/pP*100).toFixed(1):0;
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="w-full max-w-xs space-y-1">
        {[["Prior P(D)",pr,1,50,setPr],["Sensitivity",se,50,99,setSe],["False Positive",fp,1,50,setFp]].map(([label,val,min,max,set])=>(
          <div key={label} className="flex items-center gap-2"><span className="text-xs w-24">{label}: {val}%</span><input type="range" min={min} max={max} value={val} onChange={e=>set(+e.target.value)} className="flex-1"/></div>
        ))}
      </div>
      <div className="text-center p-3 rounded-lg" style={{background:"#E8F5E9"}}>
        <p className="text-xs">P(Disease | Positive) =</p>
        <p className="text-2xl font-bold" style={{color:"#2EAF7D"}}>{post}%</p>
      </div>
    </div>
  );
}

function BNViz() {
  const [pat,setPat]=useState(0);
  const pats=[
    {n:"Chain: A→B→C",d:"A ⊥ C | B — B blocks flow",e:[["A","B"],["B","C"]]},
    {n:"Fork: A←B→C",d:"A ⊥ C | B — common cause",e:[["B","A"],["B","C"]]},
    {n:"Collider: A→B←C",d:"A ⊥ C normally, but A ⊥̸ C | B!",e:[["A","B"],["C","B"]]},
  ];
  const p=pats[pat];
  return (
    <div className="flex flex-col items-center">
      <div className="flex gap-1 mb-2">{pats.map((x,i)=>(<button key={i} onClick={()=>setPat(i)} className={`px-2 py-1 text-xs rounded-full border ${pat===i?"bg-teal-100 border-teal-400":"hover:bg-gray-100"}`}>{x.n.split(":")[0]}</button>))}</div>
      <svg viewBox="0 0 300 60" className="w-full max-w-xs">
        <defs><marker id="ar" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#1A8A8A"/></marker></defs>
        {p.e.map(([a,b])=>{const ax=a==="A"?50:a==="B"?150:250,bx=b==="A"?50:b==="B"?150:250;return <line key={a+b} x1={ax+18} y1={30} x2={bx-18} y2={30} stroke="#1A8A8A" strokeWidth="2" markerEnd="url(#ar)"/>;} )}
        {["A","B","C"].map((id,i)=>(<g key={id}><circle cx={50+i*100} cy={30} r="18" fill="#E0F2F1" stroke="#1A8A8A" strokeWidth="2"/><text x={50+i*100} y={34} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#0D5C5C">{id}</text></g>))}
      </svg>
      <p className="text-xs text-center mt-1 font-medium">{p.d}</p>
    </div>
  );
}

function KNNViz() {
  const [k,setK]=useState(3);const [tp,setTp]=useState({x:150,y:100});
  const data=[{x:50,y:50,c:0},{x:70,y:80,c:0},{x:60,y:120,c:0},{x:90,y:60,c:0},{x:200,y:140,c:1},{x:230,y:120,c:1},{x:220,y:160,c:1},{x:250,y:130,c:1},{x:130,y:40,c:0},{x:180,y:70,c:1},{x:160,y:150,c:1},{x:110,y:130,c:0}];
  const cols=["#E8483F","#4A90D9"];
  const ds=data.map((d,i)=>({i,dist:Math.hypot(d.x-tp.x,d.y-tp.y),c:d.c})).sort((a,b)=>a.dist-b.dist);
  const ns=ds.slice(0,k);const v0=ns.filter(n=>n.c===0).length;const pred=v0>k/2?0:1;
  return (
    <div className="flex flex-col items-center">
      <div className="flex items-center gap-2 mb-1"><span className="text-xs">k=</span>{[1,3,5,7].map(v=><button key={v} onClick={()=>setK(v)} className={`px-2 py-0.5 text-xs rounded-full border ${k===v?"bg-gray-800 text-white":""}`}>{v}</button>)}</div>
      <svg viewBox="0 0 300 200" className="w-full max-w-xs bg-white border border-gray-200 rounded-lg" onMouseMove={e=>{const r=e.currentTarget.getBoundingClientRect();setTp({x:(e.clientX-r.left)/r.width*300,y:(e.clientY-r.top)/r.height*200});}}>
        {ns.map(n=><line key={n.i} x1={tp.x} y1={tp.y} x2={data[n.i].x} y2={data[n.i].y} stroke="#ddd" strokeWidth="1" strokeDasharray="3"/>)}
        {data.map((d,i)=><circle key={i} cx={d.x} cy={d.y} r={ns.some(n=>n.i===i)?7:5} fill={cols[d.c]} stroke={ns.some(n=>n.i===i)?"#333":"none"} strokeWidth="2" opacity={ns.some(n=>n.i===i)?1:0.4}/>)}
        <rect x={tp.x-6} y={tp.y-6} width="12" height="12" fill={cols[pred]} stroke="#333" strokeWidth="2"/>
      </svg>
    </div>
  );
}

function NNViz() {
  const ly=[3,4,2];const getY=(l,i)=>25+i*(130/ly[l]);const cx=[50,150,250];const cls=["#4A90D9","#7B61C1","#E8483F"];const lbl=["Input","Hidden","Output"];
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 170" className="w-full max-w-xs">
        {Array.from({length:ly[0]}).map((_,i)=>Array.from({length:ly[1]}).map((_,j)=><line key={`a${i}${j}`} x1={cx[0]} y1={getY(0,i)} x2={cx[1]} y2={getY(1,j)} stroke="#e5e7eb" strokeWidth="0.8"/>))}
        {Array.from({length:ly[1]}).map((_,i)=>Array.from({length:ly[2]}).map((_,j)=><line key={`b${i}${j}`} x1={cx[1]} y1={getY(1,i)} x2={cx[2]} y2={getY(2,j)} stroke="#e5e7eb" strokeWidth="0.8"/>))}
        {ly.map((sz,l)=>Array.from({length:sz}).map((_,i)=><circle key={`n${l}${i}`} cx={cx[l]} cy={getY(l,i)} r="11" fill={cls[l]} opacity="0.85"/>))}
        {lbl.map((l,i)=><text key={i} x={cx[i]} y="165" textAnchor="middle" fontSize="9" fontWeight="bold" fill={cls[i]}>{l}</text>)}
      </svg>
    </div>
  );
}

function HMMViz() {
  const [s,setS]=useState(0);
  const obs=["Walk","Shop","Walk","Clean"];const vp=[0,0,0,1];const st=["☀️ Sunny","🌧 Rainy"];const c=Math.min(s,3);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 120" className="w-full max-w-xs">
        <circle cx="80" cy="35" r="22" fill={vp[c]===0?"#FEF3C7":"#f5f5f5"} stroke="#F59E0B" strokeWidth="2"/>
        <text x="80" y="33" textAnchor="middle" fontSize="13">☀️</text><text x="80" y="45" textAnchor="middle" fontSize="7" fontWeight="bold">Sunny</text>
        <circle cx="220" cy="35" r="22" fill={vp[c]===1?"#DBEAFE":"#f5f5f5"} stroke="#4A90D9" strokeWidth="2"/>
        <text x="220" y="33" textAnchor="middle" fontSize="13">🌧</text><text x="220" y="45" textAnchor="middle" fontSize="7" fontWeight="bold">Rainy</text>
        <path d="M102,28 Q150,5 198,28" fill="none" stroke="#aaa" strokeWidth="1"/><path d="M198,42 Q150,65 102,42" fill="none" stroke="#aaa" strokeWidth="1"/>
        {obs.map((o,i)=>(<g key={i}><rect x={30+i*65} y="85" width="50" height="20" rx="4" fill={i<=c?"#E8F5E9":"#f5f5f5"} stroke={i===c?"#2EAF7D":"#ddd"} strokeWidth={i===c?2:1}/><text x={55+i*65} y="99" textAnchor="middle" fontSize="9" fontWeight={i===c?"bold":"normal"}>{o}</text></g>))}
      </svg>
      <p className="text-xs mt-1">Viterbi → <b>{st[vp[c]]}</b></p>
      <div className="flex gap-2 mt-1"><button onClick={()=>setS(Math.max(0,s-1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">←</button><button onClick={()=>setS(Math.min(3,s+1))} className="px-3 py-1 text-xs rounded-full border hover:bg-gray-100">→</button></div>
    </div>
  );
}

function CNNViz() {
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 340 110" className="w-full max-w-md">
        <rect x="5" y="15" width="35" height="70" rx="3" fill="#DBEAFE" stroke="#4A90D9" strokeWidth="1.5"/><text x="22" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#4A90D9">Input</text>
        <rect x="60" y="20" width="30" height="60" rx="3" fill="#FDE68A" stroke="#F59E0B" strokeWidth="1.5"/><text x="75" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#F59E0B">Conv</text>
        <rect x="110" y="28" width="22" height="44" rx="3" fill="#D1FAE5" stroke="#2EAF7D" strokeWidth="1.5"/><text x="121" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#2EAF7D">Pool</text>
        <rect x="152" y="28" width="22" height="44" rx="3" fill="#FDE68A" stroke="#F59E0B" strokeWidth="1.5"/><text x="163" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#F59E0B">Conv</text>
        <rect x="194" y="33" width="18" height="34" rx="3" fill="#D1FAE5" stroke="#2EAF7D" strokeWidth="1.5"/><text x="203" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#2EAF7D">Pool</text>
        {[0,1,2,3].map(i=><circle key={i} cx="250" cy={28+i*15} r="5" fill="#EDE9FE" stroke="#7B61C1" strokeWidth="1"/>)}<text x="250" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#7B61C1">FC</text>
        {[0,1].map(i=><circle key={i} cx="300" cy={38+i*24} r="7" fill="#FECACA" stroke="#E8483F" strokeWidth="1.5"/>)}<text x="300" y="100" textAnchor="middle" fontSize="7" fontWeight="bold" fill="#E8483F">Out</text>
        {[[42,58],[92,108],[134,150],[176,192],[214,240],[260,290]].map(([a,b],i)=><line key={i} x1={a} y1="50" x2={b} y2="50" stroke="#bbb" strokeWidth="1"/>)}
      </svg>
    </div>
  );
}

function MDPViz() {
  const [it,setIt]=useState(0);
  const compute=n=>{let V=[[0,0,1],[0,-10,0],[0,0,0]].map(r=>[...r]);for(let t=0;t<n;t++){const N=V.map(r=>[...r]);for(let r=0;r<3;r++)for(let c=0;c<3;c++){if(r===0&&c===2){N[r][c]=1;continue;}if(r===1&&c===1){N[r][c]=-10;continue;}const nb=[[r-1,c],[r+1,c],[r,c-1],[r,c+1]].filter(([a,b])=>a>=0&&a<3&&b>=0&&b<3);N[r][c]=nb.length?Math.max(...nb.map(([a,b])=>(r===0&&c===2?1:r===1&&c===1?-10:0)+0.9*V[a][b])):0;}V=N;}return V;};
  const V=compute(it);const lbl=[["","","🎯"],["","💀",""],["🏠","",""]];
  return (
    <div className="flex flex-col items-center">
      <div className="grid grid-cols-3 gap-1 mb-2" style={{width:174}}>
        {V.flat().map((v,i)=>{const r=Math.floor(i/3),c=i%3;return(
          <div key={i} className="flex flex-col items-center justify-center border border-gray-200 rounded" style={{width:54,height:54,background:v>0.5?`rgba(46,175,125,${Math.min(v,1)*0.5})`:v<-0.5?`rgba(232,72,63,${Math.min(-v/10,1)*0.5})`:"#f9fafb"}}>
            <span className="text-sm">{lbl[r][c]}</span><span className="text-xs font-mono font-bold">{v.toFixed(1)}</span>
          </div>);})}
      </div>
      <div className="flex items-center gap-2"><button onClick={()=>setIt(Math.max(0,it-1))} className="px-2 py-1 text-xs rounded-full border hover:bg-gray-100">−</button><span className="text-xs font-mono">Iter {it}</span><button onClick={()=>setIt(it+1)} className="px-2 py-1 text-xs rounded-full border hover:bg-gray-100">+</button></div>
    </div>
  );
}

const VIZ = { tree:TreeViz, bfs:BFSViz, astar:AStarViz, hill:HillViz, anneal:AnnealViz, mini:MiniViz, ab:ABViz, csp:CSPViz, bayes:BayesViz, bn:BNViz, knn:KNNViz, nn:NNViz, hmm:HMMViz, cnn:CNNViz, mdp:MDPViz };

// ====== QUIZ COMPONENT ======
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
        <div className="text-3xl">{pct===100?"🏆":pct>=80?"🌟":pct>=60?"👍":"📖"}</div>
        <p style={{fontSize:28,fontWeight:800,color}}>{score}/{qs.length}</p>
        <p className="text-sm text-gray-500">{pct}% — {pct===100?"Perfect!":pct>=80?"Great job!":pct>=60?"Good start!":"Keep studying!"}</p>
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
        <button onClick={retry} className="mt-3 px-5 py-2 rounded-lg text-sm font-semibold text-white" style={{background:color}}>🔄 Retry</button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-1">
        {qs.map((_, i) => <div key={i} className="rounded-full" style={{ width: i===ci?18:8, height: 8, background: i<ci?(hist[i]?.ok?"#22C55E":"#EF4444"):i===ci?color:"#E5E7EB", transition:"all 0.2s" }} />)}
        <span className="text-xs text-gray-400 ml-auto">{ci+1}/{qs.length}</span>
      </div>
      <p className="text-sm font-semibold">{q.q}</p>
      <div className="flex flex-col gap-2">
        {q.o.map((opt, idx) => {
          let bg="#fff",bdr="#e5e7eb",tc="#374151";
          if(done){if(idx===q.a){bg="#F0FDF4";bdr="#22C55E";tc="#166534";}else if(idx===sel){bg="#FEF2F2";bdr="#EF4444";tc="#991B1B";}else{tc="#9CA3AF";}}
          return (
            <button key={idx} onClick={()=>pick(idx)} disabled={done} className="flex items-center gap-3 p-3 rounded-lg text-left text-sm" style={{background:bg,border:`2px solid ${bdr}`,color:tc,cursor:done?"default":"pointer"}}>
              <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0" style={{background:done&&idx===q.a?"#22C55E":done&&idx===sel?"#EF4444":`${color}20`,color:done&&(idx===q.a||idx===sel)?"#fff":color}}>
                {done?(idx===q.a?"✓":idx===sel?"✗":String.fromCharCode(65+idx)):String.fromCharCode(65+idx)}
              </span>
              {opt}
            </button>
          );
        })}
      </div>
      {done && <div className="p-3 rounded-lg text-xs" style={{background:"#FFFBEB",border:"1px solid #FCD34D"}}>💡 {q.e}</div>}
      {done && <button onClick={next} className="self-end px-4 py-2 rounded-lg text-sm font-semibold text-white" style={{background:color}}>{ci<qs.length-1?"Next →":"Results 🎯"}</button>}
    </div>
  );
}

// ====== MAIN APP ======
export default function App() {
  const [mi, setMi] = useState(0);
  const [si, setSi] = useState(0);
  const [quiz, setQuiz] = useState(false);
  const [qk, setQk] = useState(0);
  const [sb, setSb] = useState(true);
  const ref = useRef(null);

  const mod = MODS[mi];
  const sec = mod.s[si];
  const Viz = sec.v ? VIZ[sec.v] : null;
  const hasQ = (QUIZZES[mi]||[]).length > 0;

  useEffect(() => { if(ref.current) ref.current.scrollTop = 0; }, [mi, si, quiz]);

  const goNext = () => {
    if (quiz) { setQuiz(false); if(mi<MODS.length-1){setMi(mi+1);setSi(0);} return; }
    if (si < mod.s.length-1) setSi(si+1);
    else if (hasQ) { setQuiz(true); setQk(k=>k+1); }
    else if (mi < MODS.length-1) { setMi(mi+1); setSi(0); }
  };
  const goPrev = () => {
    if (quiz) { setQuiz(false); return; }
    if (si>0) setSi(si-1);
    else if(mi>0) { setMi(mi-1); setSi(MODS[mi-1].s.length-1); }
  };

  const isEnd = mi===MODS.length-1 && (quiz || (!hasQ && si===mod.s.length-1));
  const isStart = mi===0 && si===0 && !quiz;

  const render = txt => txt.split('\n').map((line, i) => {
    line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    if(line.startsWith('•')) return <div key={i} className="pl-3 py-0.5 text-sm leading-relaxed" dangerouslySetInnerHTML={{__html:line}}/>;
    if(!line.trim()) return <div key={i} className="h-2"/>;
    return <p key={i} className="text-sm leading-relaxed" dangerouslySetInnerHTML={{__html:line}}/>;
  });

  return (
    <div style={{fontFamily:"Georgia,serif",background:"#FAFAF8",minHeight:"100vh",display:"flex",flexDirection:"column"}}>
      <div style={{background:"linear-gradient(135deg,#1a1a2e,#0f3460)",color:"#fff",padding:"14px 20px",display:"flex",alignItems:"center",gap:12}}>
        <button onClick={()=>setSb(!sb)} style={{fontSize:18,background:"none",border:"none",color:"#fff",cursor:"pointer"}}>☰</button>
        <div><div style={{fontSize:17,fontWeight:700,fontFamily:"monospace"}}>CS6601</div><div style={{fontSize:11,opacity:0.6}}>Artificial Intelligence — Interactive Guide</div></div>
      </div>
      <div style={{height:3,background:"#e5e7eb"}}><div style={{height:"100%",background:mod.c,transition:"width 0.3s",width:`${((mi+1)/MODS.length)*100}%`}}/></div>

      <div style={{display:"flex",flex:1,overflow:"hidden"}}>
        {sb && (
          <div style={{width:220,minWidth:220,borderRight:"1px solid #e5e7eb",background:"#fff",overflowY:"auto",padding:"8px 0",fontFamily:"system-ui,sans-serif"}}>
            {MODS.map((m,idx)=>(
              <div key={idx}>
                <button onClick={()=>{setMi(idx);setSi(0);setQuiz(false);}} style={{width:"100%",textAlign:"left",padding:"8px 12px",border:"none",cursor:"pointer",background:mi===idx?`${m.c}10`:"transparent",borderLeft:mi===idx?`3px solid ${m.c}`:"3px solid transparent",fontSize:12,fontWeight:mi===idx?700:500,color:mi===idx?m.c:"#555",display:"flex",alignItems:"center",gap:8}}>
                  <span style={{fontSize:15}}>{m.i}</span>{idx+1}. {m.t}
                </button>
                {mi===idx && (<>
                  {m.s.map((s,j)=>(<button key={j} onClick={()=>{setSi(j);setQuiz(false);}} style={{width:"100%",textAlign:"left",padding:"3px 12px 3px 38px",border:"none",cursor:"pointer",background:!quiz&&si===j?`${m.c}18`:"transparent",fontSize:11,color:!quiz&&si===j?m.c:"#888",fontWeight:!quiz&&si===j?600:400}}>📄 {s.t}</button>))}
                  {hasQ && <button onClick={()=>{setQuiz(true);setQk(k=>k+1);}} style={{width:"100%",textAlign:"left",padding:"3px 12px 3px 38px",border:"none",cursor:"pointer",background:quiz?`${m.c}18`:"transparent",fontSize:11,color:quiz?m.c:"#888",fontWeight:quiz?600:400}}>📝 Quiz</button>}
                </>)}
              </div>
            ))}
          </div>
        )}

        <div ref={ref} style={{flex:1,overflowY:"auto",padding:"24px 28px",maxWidth:700}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
            <span style={{fontSize:22}}>{mod.i}</span>
            <span style={{fontSize:11,fontWeight:700,color:mod.c,textTransform:"uppercase",letterSpacing:1.5,fontFamily:"system-ui"}}>Module {mi+1}: {mod.t}</span>
          </div>

          {quiz ? (<>
            <h2 style={{fontSize:24,fontWeight:700,color:"#1a1a2e",marginBottom:4}}>📝 Module {mi+1} Quiz</h2>
            <p className="text-sm text-gray-500 mb-4">{(QUIZZES[mi]||[]).length} questions</p>
            <div style={{background:"#fff",border:"1px solid #e5e7eb",borderRadius:12,padding:20,boxShadow:"0 1px 3px rgba(0,0,0,0.05)"}}>
              <Quiz key={qk} modIdx={mi} color={mod.c}/>
            </div>
          </>) : (<>
            <h2 style={{fontSize:24,fontWeight:700,color:"#1a1a2e",marginBottom:14}}>{sec.t}</h2>
            <div style={{color:"#374151",marginBottom:20}}>{render(sec.txt)}</div>
            {Viz && (<div style={{background:"#fff",border:"1px solid #e5e7eb",borderRadius:12,padding:16,marginBottom:20,boxShadow:"0 1px 3px rgba(0,0,0,0.05)"}}>
              <div style={{fontSize:10,fontWeight:700,color:mod.c,textTransform:"uppercase",letterSpacing:1,marginBottom:8,fontFamily:"system-ui"}}>▶ Interactive Visualization</div>
              <Viz/>
            </div>)}
            {si===mod.s.length-1 && hasQ && (
              <div className="flex items-center gap-3 p-4 rounded-xl mb-4" style={{background:`${mod.c}08`,border:`1px dashed ${mod.c}40`}}>
                <span className="text-2xl">📝</span>
                <div className="flex-1"><p className="text-sm font-semibold" style={{color:mod.c}}>Ready to test your knowledge?</p><p className="text-xs text-gray-500">{(QUIZZES[mi]||[]).length} questions</p></div>
                <button onClick={()=>{setQuiz(true);setQk(k=>k+1);}} className="px-4 py-2 rounded-lg text-sm font-semibold text-white" style={{background:mod.c}}>Take Quiz →</button>
              </div>
            )}
          </>)}

          <div style={{display:"flex",justifyContent:"space-between",marginTop:24,paddingTop:16,borderTop:"1px solid #e5e7eb"}}>
            <button onClick={goPrev} disabled={isStart} style={{padding:"8px 16px",borderRadius:8,border:"1px solid #ddd",background:"#fff",cursor:isStart?"not-allowed":"pointer",opacity:isStart?0.3:1,fontSize:13,fontFamily:"system-ui"}}>← Previous</button>
            <button onClick={goNext} disabled={isEnd} style={{padding:"8px 16px",borderRadius:8,border:"none",background:mod.c,color:"#fff",cursor:isEnd?"not-allowed":"pointer",opacity:isEnd?0.5:1,fontSize:13,fontWeight:600,fontFamily:"system-ui"}}>Next →</button>
          </div>
        </div>
      </div>
    </div>
  );
}
