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
• **Path Cost** — how expensive is this route?

**Tree Search vs Graph Search:**
• **Tree Search** — may revisit states, simple but can loop infinitely
• **Graph Search** — tracks an \`explored\` set to avoid revisits, uses more memory but guarantees no loops

\`\`\`python
# Generic graph search framework
def graph_search(problem, frontier):
    frontier.add(problem.initial_state)
    explored = set()

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node.solution()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored:
                frontier.add(child)
    return None  # No solution found
\`\`\`

The choice of **frontier data structure** determines which algorithm you get: queue → BFS, stack → DFS, priority queue → UCS/A*.`,
        viz: "tree-search"
      },
      {
        title: "Breadth-First Search (BFS)",
        content: `BFS explores nodes level by level, like ripples spreading out from a stone dropped in water. It uses a **queue** (FIFO) to track which nodes to visit next.

**Properties:**
• **Complete** — always finds a solution if one exists
• **Optimal** — finds shallowest solution (cheapest if all edges cost the same)
• **Time & Space** — O(b^d) where b = branching factor, d = depth

BFS is great when all step costs are equal and the solution is near the root.

**DFS (Depth-First Search)** uses a **stack** (LIFO) instead. It dives deep before backtracking. Not optimal, not complete on infinite trees, but uses only O(bm) memory — much less than BFS.

\`\`\`python
from collections import deque

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': ['F'], 'F': []
}
print(bfs(graph, 'A', 'F'))  # ['A', 'C', 'F']
\`\`\`

**Step-by-step BFS trace on the graph above:**

\`\`\`
Goal: find path from A to F

Step 1: Dequeue A
  Visit neighbors: B, C
  Queue: [B, C]    Visited: {A, B, C}

Step 2: Dequeue B
  Visit neighbors: D, E
  Queue: [C, D, E]  Visited: {A, B, C, D, E}

Step 3: Dequeue C
  Visit neighbor: F → GOAL FOUND!
  Path: A → C → F (depth 2)
\`\`\`

Notice BFS found the *shallowest* path (2 edges), not necessarily through B. DFS would have gone A→B→D (dead end), backtrack, A→B→E→F (depth 3) — valid but deeper.`,
        viz: "bfs"
      },
      {
        title: "Uniform Cost Search (UCS)",
        content: `UCS is like BFS but for weighted graphs. Instead of expanding the shallowest node, it expands the **cheapest** node first using a **priority queue**.

**Key insight:** UCS always expands the node with the lowest total path cost \`g(n)\`. It's guaranteed to find the optimal solution.

**Think of it like:** "Always follow the cheapest unexplored path so far."

UCS is equivalent to Dijkstra's algorithm. BFS is just UCS where every edge costs 1.

\`\`\`python
import heapq

def ucs(graph, start, goal):
    # Priority queue: (cost, node, path)
    frontier = [(0, start, [start])]
    explored = set()

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node == goal:
            return cost, path
        if node in explored:
            continue
        explored.add(node)
        for neighbor, edge_cost in graph[node]:
            if neighbor not in explored:
                heapq.heappush(frontier,
                    (cost + edge_cost, neighbor,
                     path + [neighbor]))
    return None

# Weighted graph example
graph = {
    'A': [('B', 1), ('C', 5)],
    'B': [('C', 2), ('D', 6)],
    'C': [('D', 2)],
    'D': []
}
print(ucs(graph, 'A', 'D'))
# (5, ['A', 'B', 'C', 'D'])
\`\`\`

Note how the path A→B→C→D (cost 5) is cheaper than A→B→D (cost 7) even though it has more steps.

**Step-by-step UCS trace:**

\`\`\`
Graph: A→B(1), A→C(5), B→C(2), B→D(6), C→D(2)

Step 1: Pop A (cost 0)
  Push B:1, C:5
  Frontier: [(1,B), (5,C)]

Step 2: Pop B (cost 1) ← cheapest!
  Push C:1+2=3, D:1+6=7
  Frontier: [(3,C), (5,C), (7,D)]

Step 3: Pop C (cost 3) ← via A→B→C
  Push D:3+2=5
  Frontier: [(5,C), (5,D), (7,D)]

Step 4: Pop C (cost 5) ← already explored, skip

Step 5: Pop D (cost 5) → GOAL! Path: A→B→C→D
\`\`\`

**Pitfall:** UCS may explore the same node multiple times with different costs. The \`explored\` set ensures we only process each node at its cheapest cost.`,
        viz: "ucs"
      },
      {
        title: "A* Search",
        content: `A* combines the best of UCS and greedy search. It uses an **evaluation function**:

**f(n) = g(n) + h(n)**

• **g(n)** = actual cost from start to node n
• **h(n)** = estimated cost from n to goal (heuristic)

**Admissible heuristic:** never overestimates the true cost (optimistic). This guarantees A* finds the optimal solution.

**Consistent heuristic:** h(n) ≤ cost(n→n') + h(n'). This means the heuristic obeys the triangle inequality. Consistency implies admissibility.

A* is optimally efficient — no other optimal algorithm expands fewer nodes.

\`\`\`python
import heapq

def a_star(graph, start, goal, h):
    # (f_cost, g_cost, node, path)
    frontier = [(h(start, goal), 0, start, [start])]
    explored = {}  # node -> best g_cost seen

    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        if node == goal:
            return g, path
        if node in explored and explored[node] <= g:
            continue
        explored[node] = g
        for nbr, cost in graph[node]:
            new_g = g + cost
            new_f = new_g + h(nbr, goal)
            heapq.heappush(frontier,
                (new_f, new_g, nbr, path + [nbr]))
    return None

# Manhattan distance (common for grids)
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
\`\`\`

**Why does admissibility matter?** If h(n) overestimates, A* might skip the optimal path thinking it's too expensive. An admissible h(n) ensures we never prematurely dismiss the best route.

**Common heuristics:**
• **Grid pathfinding** — Manhattan distance (no diagonals) or Euclidean distance
• **8-puzzle** — number of misplaced tiles, or sum of Manhattan distances per tile
• **Graph problems** — straight-line distance to goal

**Step-by-step A* trace (using same graph + heuristic):**

\`\`\`
Graph: A→B(1), A→C(5), B→C(2), B→D(6), C→D(2)
Heuristic h: A=7, B=6, C=2, D=0

Step 1: Pop A  g=0, h=7, f=0+7=7
  Push B: f=1+6=7, C: f=5+2=7
  Frontier: [(7,B), (7,C)]

Step 2: Pop B  g=1, h=6, f=7
  Push C: f=3+2=5, D: f=7+0=7
  Frontier: [(5,C), (7,C), (7,D)]

Step 3: Pop C  g=3, h=2, f=5 ← lowest f!
  Push D: f=5+0=5
  Frontier: [(5,D), (7,C), (7,D)]

Step 4: Pop D  g=5, f=5 → GOAL! Path: A→B→C→D
\`\`\`

Notice A* expanded only 4 nodes. UCS expanded 5 (including a duplicate). The heuristic guided A* to explore C (f=5) before D (f=7), finding the optimal path faster.

**Exam tip:** If h(n)=0 for all n, A* becomes UCS. If g(n)=0, A* becomes Greedy Best-First (fast but not optimal).`,
        viz: "astar"
      },
      {
        title: "Search Algorithm Comparison",
        content: `**When to use which algorithm:**

• **BFS** — all edge costs equal, solution is shallow, you need the shortest path in terms of steps
• **DFS** — memory is limited, solution is deep, you just need *any* solution (not optimal)
• **UCS** — edges have different costs, you need the cheapest path, no good heuristic available
• **A*** — you have a good admissible heuristic, need optimal solution efficiently

**Comparison at a glance:**

\`\`\`
Algorithm | Frontier   | Complete | Optimal | Time   | Space
----------|------------|----------|---------|--------|------
DFS       | Stack      | No*      | No      | O(b^m) | O(bm)
BFS       | Queue      | Yes      | Yes**   | O(b^d) | O(b^d)
UCS       | PQ (g)     | Yes      | Yes     | O(b^C) | O(b^C)
A*        | PQ (g+h)   | Yes      | Yes***  | varies | varies

*  Not complete on infinite/cyclic graphs
** Optimal only if all step costs are equal
***Optimal only if h(n) is admissible
C = optimal cost / minimum edge cost
\`\`\`

**Common exam pitfalls:**
• A* with an *inadmissible* heuristic still finds a solution — just not guaranteed optimal
• UCS and BFS are both complete, but UCS can be very slow if edge costs are tiny (many cheap expansions)
• Graph search (with explored set) fixes DFS completeness on finite graphs but may miss optimal paths unless the algorithm handles re-expansion (A* with consistent h avoids this)
• Iterative Deepening DFS (IDS) gives BFS-like completeness with DFS-like memory: O(bd) time, O(bd) space — often the best uninformed strategy`,
        viz: null
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
• **Ridges** — narrow peaks that are hard to navigate

**Random-restart hill climbing** helps escape local maxima — run hill climbing many times from random starting points and keep the best result.

\`\`\`python
import random

def hill_climbing(problem):
    current = problem.random_state()
    while True:
        neighbors = problem.get_neighbors(current)
        best = max(neighbors, key=problem.value)
        if problem.value(best) <= problem.value(current):
            return current  # Stuck at local max
        current = best

def random_restart(problem, restarts=100):
    best = None
    for _ in range(restarts):
        result = hill_climbing(problem)
        if best is None or \\
           problem.value(result) > problem.value(best):
            best = result
    return best
\`\`\`

**Key tradeoff:** Hill climbing is fast but incomplete. Random restarts improve coverage but don't guarantee the global optimum.`,
        viz: "hill-climbing"
      },
      {
        title: "Simulated Annealing",
        content: `Inspired by metallurgy! When metal cools slowly (annealing), atoms find low-energy arrangements.

**The algorithm:**
1. Start at a random solution with high "temperature" T
2. Pick a random neighbor
3. If better → always accept
4. If worse → accept with probability \`e^(ΔE/T)\`
5. Gradually reduce T (cooling schedule)

**Key insight:** At high temperature, we explore freely (accept bad moves). As temperature drops, we become pickier and converge to a good solution.

**Common cooling schedules:**
• **Geometric:** \`T = T₀ × α^t\` where α ≈ 0.95–0.99 (most common)
• **Linear:** \`T = T₀ - α·t\`
• **Logarithmic:** \`T = T₀ / ln(t+1)\` — guarantees convergence but very slow

\`\`\`python
import random, math

def simulated_annealing(problem, T=1.0,
                        cooling=0.995, min_T=1e-8):
    current = problem.random_state()
    best = current

    while T > min_T:
        neighbor = problem.random_neighbor(current)
        delta = problem.value(neighbor) \\
                - problem.value(current)

        # Accept better moves always,
        # worse moves with probability e^(delta/T)
        if delta > 0 or \\
           random.random() < math.exp(delta / T):
            current = neighbor

        if problem.value(current) > \\
           problem.value(best):
            best = current

        T *= cooling  # Geometric cooling

    return best
\`\`\`

**Step-by-step SA trace (maximizing):**

\`\`\`
T=100, current=5, best=5

Step 1: neighbor=8, Δ=+3 → better, ACCEPT
  current=8, best=8

Step 2: neighbor=3, Δ=-5 → worse
  P(accept) = e^(-5/100) = 0.95 → rand=0.2 ACCEPT
  current=3 (exploring!)

Step 3: neighbor=12, Δ=+9 → better, ACCEPT
  current=12, best=12  ← escaped local max!

... T cools to 0.1 ...

Step 99: neighbor=11, Δ=-1 → worse
  P(accept) = e^(-1/0.1) = 0.00005 → REJECT
  (Too cold — only accept improvements now)
\`\`\`

**Key insight:** Early on (high T), SA accepted a *worse* move (8→3), which let it discover 12 — a better solution it couldn't reach from 8 directly. This is what makes SA powerful.

**Guarantee:** With a slow enough cooling schedule, SA will find the global optimum! In practice, geometric cooling with α ≈ 0.995 works well.`,
        viz: "annealing"
      },
      {
        title: "Genetic Algorithms",
        content: `Inspired by biological evolution! Maintain a **population** of candidate solutions that evolve over generations.

**Steps each generation:**
1. **Selection** — choose fitter individuals to reproduce (roulette wheel, tournament)
2. **Crossover** — combine parts of two parents to create children (single-point, uniform)
3. **Mutation** — randomly modify some children to maintain diversity

**Example — N-Queens:** Represent a board as \`[2,4,1,3]\` where index = row, value = column. Crossover: swap a section between two parents. Mutation: move one queen to a random column.

\`\`\`python
import random

def genetic_algorithm(pop, fitness_fn,
                      crossover_fn, mutate_fn,
                      gens=1000, mut_rate=0.1):
    for gen in range(gens):
        pop.sort(key=fitness_fn, reverse=True)
        if fitness_fn(pop[0]) == 1.0:
            return pop[0]  # Perfect solution

        # Top half survive as parents
        parents = pop[:len(pop)//2]
        children = list(parents)

        while len(children) < len(pop):
            p1, p2 = random.sample(parents, 2)
            child = crossover_fn(p1, p2)
            if random.random() < mut_rate:
                child = mutate_fn(child)
            children.append(child)

        pop = children
    return max(pop, key=fitness_fn)

# N-Queens crossover: single-point
def crossover(p1, p2):
    cut = random.randint(1, len(p1)-1)
    return p1[:cut] + p2[cut:]

# N-Queens mutation: change one queen
def mutate(board):
    b = list(board)
    i = random.randint(0, len(b)-1)
    b[i] = random.randint(0, len(b)-1)
    return b
\`\`\`

GAs are good for large, complex search spaces where the structure of good solutions isn't well understood. They often find good-enough solutions quickly, even if they can't guarantee optimality.`,
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

**Time complexity:** O(b^m) where b = branching factor, m = max depth

\`\`\`python
def minimax(state, depth, is_max, game):
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None

    best_move = None
    if is_max:
        best_val = float('-inf')
        for move in game.get_moves(state):
            child = game.apply(state, move)
            val, _ = minimax(child, depth-1,
                             False, game)
            if val > best_val:
                best_val, best_move = val, move
        return best_val, best_move
    else:
        best_val = float('inf')
        for move in game.get_moves(state):
            child = game.apply(state, move)
            val, _ = minimax(child, depth-1,
                             True, game)
            if val < best_val:
                best_val, best_move = val, move
        return best_val, best_move
\`\`\`

**Step-by-step Minimax trace:**

\`\`\`
         MAX
        / | \\
      MIN MIN MIN
      /\\  /\\  /\\
     3 5 2 9 0 7    ← terminal values

Step 1 (bottom-up): MIN nodes pick smallest
  MIN₁ = min(3,5) = 3
  MIN₂ = min(2,9) = 2
  MIN₃ = min(0,7) = 0

Step 2: MAX node picks largest
  MAX = max(3, 2, 0) = 3

Result: MAX plays left branch, guaranteeing
at least 3 regardless of MIN's response.
\`\`\`

**Intuition:** MAX thinks "what's the worst MIN can do to me in each branch?" then picks the branch where the worst case is best. This is the *guaranteed* outcome with optimal play.

**Example:** In tic-tac-toe (b≈5, m=9), minimax explores ~2 million nodes. Chess (b≈35, m≈80) would need ~10^120 — impossible! That's why we need pruning and depth limits.`,
        viz: "minimax"
      },
      {
        title: "Alpha-Beta Pruning",
        content: `Alpha-Beta is Minimax made smart. It skips branches that **can't possibly influence** the final decision.

**Two values tracked:**
• **α (alpha)** = best value MAX can guarantee (starts at -∞)
• **β (beta)** = best value MIN can guarantee (starts at +∞)

**Pruning rule:** If α ≥ β, stop exploring that branch — it's irrelevant.

**Best case:** With perfect move ordering, Alpha-Beta examines only O(b^(m/2)) nodes — effectively doubling the search depth!

\`\`\`python
def alpha_beta(state, depth, alpha, beta,
               is_max, game):
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None

    best_move = None
    if is_max:
        value = float('-inf')
        for move in game.get_moves(state):
            child = game.apply(state, move)
            v, _ = alpha_beta(child, depth-1,
                alpha, beta, False, game)
            if v > value:
                value, best_move = v, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value, best_move
    else:
        value = float('inf')
        for move in game.get_moves(state):
            child = game.apply(state, move)
            v, _ = alpha_beta(child, depth-1,
                alpha, beta, True, game)
            if v < value:
                value, best_move = v, move
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value, best_move

# Usage:
# val, move = alpha_beta(state, 6,
#     float('-inf'), float('inf'), True, game)
\`\`\`

**Step-by-step Alpha-Beta trace:**

\`\`\`
         MAX (α=-∞, β=+∞)
        / | \\
      MIN MIN MIN
      /\\  /\\  /\\
     3 5 2 9 0 7

Left branch:
  MIN₁ sees 3 → β=3. Sees 5 → min(3,5)=3
  MAX updates α=3

Middle branch:
  MIN₂ sees 2 → β=2.
  α(3) ≥ β(2) → PRUNE! Skip 9 ✂️
  (MAX already has 3; this branch gives ≤2)

Right branch:
  MIN₃ sees 0 → β=0.
  α(3) ≥ β(0) → PRUNE! Skip 7 ✂️

Result: MAX=3, but only evaluated 4 of 6 leaves!
\`\`\`

**Move ordering matters!** If you examine the best move first, pruning is maximized. Common trick: use iterative deepening — search to depth 1, 2, 3… and use each result to order moves at the next depth.

**Common exam pitfalls:**
• Pruning does NOT change the final minimax value — it only skips irrelevant work
• α is updated at MAX nodes, β at MIN nodes — don't mix them up
• Pruning depends on evaluation *order* but the minimax result does not
• With random ordering: O(b^(3m/4)). With perfect ordering: O(b^(m/2))`,
        viz: "alpha-beta"
      },
      {
        title: "Advanced Game Concepts",
        content: `**Evaluation Functions:** When we can't search to the end, we estimate position quality. Good eval functions capture key features (material, mobility, position).

\`\`\`python
# Simple chess evaluation
def evaluate(board):
    values = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}
    score = sum(values[p.type]
                for p in board.white_pieces)
    score -= sum(values[p.type]
                 for p in board.black_pieces)
    return score  # Positive = white advantage
\`\`\`

**Iterative Deepening:** Search depth 1, 2, 3… until time runs out. Always have a "best move so far" ready. Also helps with move ordering for alpha-beta.

**Quiescent Search:** Don't evaluate "noisy" positions (e.g., mid-capture in chess). Extend search until the position is quiet.

**Horizon Effect:** Limited depth can cause the AI to delay inevitable bad outcomes by pushing them beyond the search horizon. Example: sacrificing pieces to push checkmate past the depth limit.

**Expectimax:** For games with chance (dice, cards), replace MIN nodes with **chance nodes** that compute the weighted average over possible outcomes.

\`\`\`python
def expectimax(state, depth, agent, game):
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state)

    if agent == 'max':
        return max(
            expectimax(game.apply(state, m),
                       depth-1, 'chance', game)
            for m in game.get_moves(state))
    else:  # Chance node
        moves = game.get_moves(state)
        return sum(
            (1/len(moves)) *
            expectimax(game.apply(state, m),
                       depth-1, 'max', game)
            for m in moves)
\`\`\`

**Key difference:** Alpha-beta pruning does NOT work with Expectimax because chance nodes need all children to compute the expected value.`,
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

**Constraint Graph:** Variables are nodes, edges connect variables that share a constraint.

**Simple CSP Backtracking Solver:**

\`\`\`python
def backtrack(assignment, csp):
    if len(assignment) == len(csp.variables):
        return assignment  # All variables assigned

    var = select_unassigned_variable(csp, assignment)
    for value in csp.domains[var]:
        if is_consistent(var, value, assignment, csp):
            assignment[var] = value
            result = backtrack(assignment, csp)
            if result is not None:
                return result
            del assignment[var]  # Backtrack
    return None

def is_consistent(var, value, assignment, csp):
    for neighbor in csp.neighbors[var]:
        if neighbor in assignment:
            if assignment[neighbor] == value:
                return False
    return True
\`\`\`

**Common Exam Pitfall:** Remember that basic backtracking tries values in domain order and variables in arbitrary order. Without heuristics, it's exponential!`,
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

These heuristics can turn exponential problems into near-linear ones!

**Step-by-step trace — Map coloring with Forward Checking:**

Regions: \`WA, NT, SA, Q\` (Western Australia, Northern Territory, South Australia, Queensland)
Domains: \`{R, G, B}\` for each region
Constraints: Adjacent regions ≠ same color

\`\`\`
Step 1: Assign WA=R
  Forward check: NT domain becomes {G,B}, SA domain becomes {G,B}

Step 2: Pick NT (arbitrary), assign NT=G
  Forward check: SA domain becomes {B} (can't be R or G)

Step 3: Assign SA=B
  Forward check: Q domain becomes {R,G}

Step 4: Assign Q=R (or G, both work)
  Solution found: {WA:R, NT:G, SA:B, Q:R}
\`\`\`

**AC-3 Algorithm (Arc Consistency):**

\`\`\`python
def ac3(csp):
    queue = [(Xi, Xj) for Xi in csp.variables
                      for Xj in csp.neighbors[Xi]]
    while queue:
        (Xi, Xj) = queue.pop(0)
        if revise(csp, Xi, Xj):
            if len(csp.domains[Xi]) == 0:
                return False  # No solution
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True

def revise(csp, Xi, Xj):
    revised = False
    for x in csp.domains[Xi][:]:
        if not any(is_compatible(x, y, Xi, Xj)
                   for y in csp.domains[Xj]):
            csp.domains[Xi].remove(x)
            revised = True
    return revised
\`\`\`

**Exam Tips:**
• **MRV vs LCV:** MRV picks which variable to assign next (most constrained), LCV picks which value to try first (least constraining). Don't confuse them!
• **Degree Heuristic:** When multiple variables have same MRV count, pick the one with most constraints on unassigned variables
• **Forward Checking vs AC-3:** FC only checks neighbors of assigned variable. AC-3 propagates further, checking consistency across all arcs
• **Local Search for CSPs:** Methods like min-conflicts can be faster for large problems but don't guarantee finding a solution`,
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

**Total Probability:** P(A) = Σ P(A|B=b) × P(B=b) — sum over all possible values of B

**Worked Example — Conditional Probability:**

Two dice are rolled. Given that the sum is greater than 8, what's the probability that both dice show the same number?

\`\`\`
Event A = both dice same = {(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)}
Event B = sum > 8 = {(3,6),(4,5),(4,6),(5,4),(5,5),(5,6),(6,3),(6,4),(6,5),(6,6)}

A ∧ B = {(5,5), (6,6)} (2 outcomes)
|B| = 10 outcomes

P(A|B) = |A ∧ B| / |B| = 2/10 = 0.2
\`\`\`

**Bayes Computation in Python:**

\`\`\`python
def bayes(prior_A, likelihood_B_given_A, prob_B):
    """Compute P(A|B) using Bayes' Rule"""
    return (likelihood_B_given_A * prior_A) / prob_B

# Example: Medical test
prior_disease = 0.01
sensitivity = 0.9  # P(positive | disease)
prob_positive = 0.9 * 0.01 + 0.2 * 0.99  # Total probability

posterior = bayes(prior_disease, sensitivity, prob_positive)
print(f"P(disease | positive test) = {posterior:.4f}")
# Output: 0.0435 (4.35%)
\`\`\`

**Computing Total Probability:**

\`\`\`python
def total_probability(events, probs, conditional_probs):
    """P(A) = Σ P(A|B_i) × P(B_i)"""
    return sum(conditional_probs[i] * probs[i]
               for i in range(len(events)))

# Example: Rain probability given weather forecast
weather = ['Sunny', 'Cloudy', 'Stormy']
prob_weather = [0.6, 0.3, 0.1]
prob_rain_given = [0.1, 0.5, 0.9]

prob_rain = total_probability(weather, prob_weather,
                               prob_rain_given)
print(f"P(Rain) = {prob_rain}")  # 0.6*0.1 + 0.3*0.5 + 0.1*0.9 = 0.3
\`\`\``,
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

Even with a positive test, cancer is unlikely because the base rate is so low! This is why prior probabilities matter.

**Step-by-step Cancer Screening Calculation:**

\`\`\`
Given:
  P(C) = 0.01          (prior: 1% have cancer)
  P(¬C) = 0.99         (99% don't have cancer)
  P(+|C) = 0.9         (sensitivity: test detects 90% of cancers)
  P(+|¬C) = 0.2        (false positive: 20% of healthy get positive)

Step 1: Calculate P(+) using total probability
  P(+) = P(+|C)×P(C) + P(+|¬C)×P(¬C)
  P(+) = 0.9×0.01 + 0.2×0.99
  P(+) = 0.009 + 0.198
  P(+) = 0.207

Step 2: Apply Bayes' Rule
  P(C|+) = P(+|C) × P(C) / P(+)
  P(C|+) = (0.9 × 0.01) / 0.207
  P(C|+) = 0.009 / 0.207
  P(C|+) ≈ 0.0435 = 4.35%

Interpretation:
Out of 10,000 people:
  - 100 have cancer (1%)
    → 90 test positive (90% sensitivity)
    → 10 test negative (false negatives)
  - 9,900 don't have cancer
    → 1,980 test positive (20% false positive rate)
    → 7,920 test negative

Total positive tests: 90 + 1,980 = 2,070
Of those, truly have cancer: 90
Probability: 90/2,070 ≈ 4.35%
\`\`\`

**Exam Pitfalls:**
• **Base Rate Fallacy:** Don't ignore \`P(A)\` — even a good test is unreliable if the condition is rare!
• **Independence vs Conditional Independence:** \`A\` and \`B\` can be dependent but conditionally independent given \`C\`. Example: Fire and Smoke are dependent, but given FireAlarm, they become independent (both caused by alarm state)
• **Confusing P(A|B) with P(B|A):** These are NOT the same! P(positive|cancer) ≠ P(cancer|positive)
• **Forgetting to normalize:** When computing posteriors, sum must equal 1. Use P(B) = Σ P(B|Ai)P(Ai) to normalize`,
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

Example: Weather → Sprinkler, Weather → Rain, Sprinkler → WetGrass, Rain → WetGrass

**Representing a Simple Bayes Net in Python:**

\`\`\`python
# Simple network: Cloudy → Rain, Cloudy → Sprinkler,
#                 Rain → WetGrass, Sprinkler → WetGrass

class BayesNet:
    def __init__(self):
        # CPT for Cloudy (no parents)
        self.P_Cloudy = 0.5

        # CPT for Rain given Cloudy
        self.P_Rain = {
            True: 0.8,   # P(Rain | Cloudy)
            False: 0.2   # P(Rain | ¬Cloudy)
        }

        # CPT for Sprinkler given Cloudy
        self.P_Sprinkler = {
            True: 0.1,   # P(Sprinkler | Cloudy)
            False: 0.5   # P(Sprinkler | ¬Cloudy)
        }

        # CPT for WetGrass given Rain and Sprinkler
        self.P_WetGrass = {
            (True, True): 0.99,    # P(Wet | Rain ∧ Sprinkler)
            (True, False): 0.9,    # P(Wet | Rain ∧ ¬Sprinkler)
            (False, True): 0.9,    # P(Wet | ¬Rain ∧ Sprinkler)
            (False, False): 0.0    # P(Wet | ¬Rain ∧ ¬Sprinkler)
        }

    def joint_prob(self, cloudy, rain, sprinkler, wet):
        """Compute joint probability using chain rule"""
        p = self.P_Cloudy if cloudy else (1 - self.P_Cloudy)
        p *= self.P_Rain[cloudy] if rain else (1 - self.P_Rain[cloudy])
        p *= self.P_Sprinkler[cloudy] if sprinkler else (1 - self.P_Sprinkler[cloudy])
        p *= self.P_WetGrass[(rain, sprinkler)] if wet else (1 - self.P_WetGrass[(rain, sprinkler)])
        return p

# Full joint would need 2^4 = 16 entries
# Bayes Net needs: 1 + 2 + 2 + 4 = 9 parameters!
\`\`\``,
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

**Explaining away example:** If the grass is wet (observed), learning it rained makes it LESS likely the sprinkler was on.

**Step-by-step D-Separation Check:**

Network: \`Burglary → Alarm ← Earthquake\`, \`Alarm → JohnCalls\`, \`Alarm → MaryCalls\`

\`\`\`
Query 1: Are Burglary and Earthquake independent?
  Path: Burglary → Alarm ← Earthquake
  Pattern: Collider at Alarm
  Alarm not observed → Path BLOCKED
  Answer: YES, they are independent

Query 2: Are Burglary and Earthquake independent given Alarm?
  Path: Burglary → Alarm ← Earthquake
  Pattern: Collider at Alarm
  Alarm IS observed → Path ACTIVE (explaining away!)
  Answer: NO, dependent given Alarm

Query 3: Are JohnCalls and MaryCalls independent given Alarm?
  Path: JohnCalls ← Alarm → MaryCalls
  Pattern: Fork at Alarm
  Alarm IS observed → Path BLOCKED
  Answer: YES, independent given Alarm

Query 4: Are Burglary and JohnCalls independent?
  Path: Burglary → Alarm → JohnCalls
  Pattern: Chain through Alarm
  Alarm not observed → Path ACTIVE
  Answer: NO, they are dependent
\`\`\`

**Exam Tips:**
• **Markov Blanket:** A node is conditionally independent of ALL other nodes given its Markov blanket (parents + children + children's other parents)
• **Active vs Blocked Paths:** A path is active if:
  - For chains/forks: middle node NOT observed
  - For colliders: middle node (or descendant) IS observed
• **Multiple Paths:** If ANY path is active, variables are dependent
• **D-separation algorithm:** To test if X ⊥ Y | Z:
  1. Find all undirected paths from X to Y
  2. Check if ALL paths are blocked by Z
  3. If yes → independent; if any active → dependent`,
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
• **Gibbs Sampling** — start with random values, repeatedly resample one variable at a time conditioned on its Markov blanket. A form of MCMC.

**Variable Elimination Example (Python):**

Network: \`Cloudy → Rain → WetGrass\`

Query: P(WetGrass | Cloudy=True)

\`\`\`python
def variable_elimination(query_var, evidence, bn):
    """
    Compute P(query_var | evidence) using VE
    """
    # Step 1: Create initial factors from CPTs
    factors = []
    for var in bn.variables:
        if var in evidence:
            # Fix evidence variables
            factor = bn.get_cpt(var, evidence)
        else:
            factor = bn.get_cpt(var)
        factors.append(factor)

    # Step 2: Eliminate hidden variables (not query, not evidence)
    hidden = [v for v in bn.variables
              if v != query_var and v not in evidence]

    for var in hidden:
        # Find all factors mentioning var
        relevant = [f for f in factors if var in f.vars]
        factors = [f for f in factors if var not in f.vars]

        # Multiply them together
        product = multiply_factors(relevant)

        # Sum out (marginalize) var
        marginalized = sum_out(product, var)
        factors.append(marginalized)

    # Step 3: Multiply remaining factors
    result = multiply_factors(factors)

    # Step 4: Normalize
    return normalize(result)

# Trace on small network:
# P(WetGrass | Cloudy=True)
#
# Factors: f1(Cloudy), f2(Rain|Cloudy), f3(Wet|Rain)
# Evidence: Cloudy=True → fix f1
#
# Eliminate Rain:
#   Join f2 and f3 → f4(Cloudy, Rain, Wet)
#   Sum out Rain → f5(Cloudy, Wet)
#
# Result: f5 with Cloudy=True gives P(Wet)
\`\`\`

**Rejection Sampling Trace:**

Query: P(Rain | WetGrass=true, Cloudy=false)

\`\`\`
Sample 1: Cloudy=T, Rain=T, Wet=T → REJECT (Cloudy≠false)
Sample 2: Cloudy=F, Rain=F, Wet=F → REJECT (Wet≠true)
Sample 3: Cloudy=F, Rain=T, Wet=T → ACCEPT ✓
Sample 4: Cloudy=T, Rain=F, Wet=F → REJECT (Cloudy≠false)
Sample 5: Cloudy=F, Rain=F, Wet=F → REJECT (Wet≠true)
Sample 6: Cloudy=F, Rain=T, Wet=T → ACCEPT ✓
Sample 7: Cloudy=F, Rain=F, Wet=T → ACCEPT ✓ (sprinkler caused it)
...

After 1000 samples: 37 accepted
  - 24 had Rain=true
  - 13 had Rain=false
Estimate: P(Rain|Wet,¬Cloudy) ≈ 24/37 = 0.65

Problem: Rejection rate = 96.3% → very wasteful!
\`\`\`

**Exam Tips:**
• **VE Complexity:** Depends on elimination order. Bad order → huge intermediate factors. Good order → exponential speedup!
• **Rejection Sampling:** Wasteful when evidence is unlikely. Many samples rejected.
• **Likelihood Weighting:** Better than rejection — fixes evidence variables during sampling, weights by P(evidence). No rejection!
• **Gibbs Sampling:** MCMC method. Initial samples biased, but converges to correct distribution. Each iteration only resamples one variable given Markov blanket.
• **When to use approximate:** Large networks where exact inference intractable (too many variables, high treewidth)`,
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
**Cons:** Slow at prediction time (must search all data), sensitive to irrelevant features, curse of dimensionality

**Python Implementation:**
\`\`\`python
import numpy as np
from collections import Counter

def knn_classify(train_X, train_y, test_point, k=3):
    distances = np.sqrt(np.sum((train_X - test_point)**2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = train_y[nearest_indices]
    return Counter(nearest_labels).most_common(1)[0][0]
\`\`\`

**Trace: Classifying (5,4) with k=3:**
\`\`\`
Training: (1,2)->A (2,3)->A (3,3)->A (6,5)->B (7,7)->B (8,6)->B
Distances: (6,5)->1.41 (3,3)->2.24 (2,3)->3.16 (7,7)->3.61 ...
3 nearest: B, A, A → Vote: 2A, 1B → Predict A
\`\`\`

**Exam Pitfalls:**
• Feature scaling crucial—normalize first!
• Odd k avoids ties; use cross-validation
• Curse of dimensionality: high-D → all points equidistant`,
        viz: "knn"
      },
      {
        title: "Naive Bayes & Gaussian Classifiers",
        content: `**Gaussian Classifier:** Assume each class generates data from a Gaussian (bell curve) distribution. Classify by finding which Gaussian is most likely.

**Decision Boundary:** Where the two Gaussians are equally likely — this creates a line (or curve) separating classes.

**Naive Bayes:** Assumes all features are **conditionally independent** given the class. Despite this "naive" assumption, it works surprisingly well!

P(Class | Features) ∝ P(Class) × Π P(Feature_i | Class)

**Why it works:** Even if the independence assumption is wrong, the resulting classifier often ranks probabilities correctly.

**Python Implementation:**
\`\`\`python
class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def train(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.class_probs = dict(zip(classes, counts / len(y)))

        for c in classes:
            X_c = X[y == c]
            self.feature_probs[c] = {}
            for i in range(X.shape[1]):
                vals, counts = np.unique(X_c[:, i], return_counts=True)
                # Laplace smoothing: add 1 to all counts
                self.feature_probs[c][i] = dict(zip(vals, (counts+1)/(len(X_c)+len(vals))))

    def predict(self, x):
        scores = {}
        for c in self.class_probs:
            score = self.class_probs[c]
            for i, val in enumerate(x):
                score *= self.feature_probs[c][i].get(val, 1/len(self.feature_probs[c][i]))
            scores[c] = score
        return max(scores, key=scores.get)
\`\`\`

**Example: Email spam classification**
\`\`\`
Training data:
  Email 1: "buy now"    -> Spam
  Email 2: "meeting now" -> Ham
  Email 3: "buy cheap"   -> Spam

Classify: "buy meeting"

P(Spam) = 2/3,  P(Ham) = 1/3
P("buy"|Spam) = 2/2 = 1.0,  P("buy"|Ham) = 0/1 = 0 (Laplace: 1/4)
P("meeting"|Spam) = 0/2 = 0 (Laplace: 1/4),  P("meeting"|Ham) = 1/1 = 1.0

P(Spam|"buy meeting") ∝ (2/3) × 1.0 × (1/4) = 0.167
P(Ham|"buy meeting")  ∝ (1/3) × (1/4) × 1.0 = 0.083
→ Predict Spam
\`\`\`

**Exam Tips:**
• Laplace smoothing: add 1 to counts to avoid zero probabilities
• Log probabilities prevent underflow: log(P) = log(prior) + Σ log(P(f|c))
• Independence assumption rarely holds, but ranking often correct`,
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
• **Boosting** — build trees sequentially, focusing on mistakes

**Python: Entropy & Information Gain**
\`\`\`python
import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(X_col, y):
    parent_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for v, count in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (count / len(y)) * entropy(subset_y)
    return parent_entropy - weighted_entropy
\`\`\`

**Building Tree Example:**
\`\`\`
Dataset:
  Outlook  Temp   Play?
  Sunny    Hot    No
  Sunny    Cool   Yes
  Rain     Cool   Yes
  Rain     Hot    No

Initial entropy: H = -(2/4)log₂(2/4) - (2/4)log₂(2/4) = 1.0

Split on Outlook:
  Sunny: [No, Yes] → H=1.0
  Rain:  [Yes, No] → H=1.0
  IG(Outlook) = 1.0 - (2/4)(1.0) - (2/4)(1.0) = 0.0

Split on Temp:
  Hot:  [No, No] → H=0.0
  Cool: [Yes, Yes] → H=0.0
  IG(Temp) = 1.0 - (2/4)(0.0) - (2/4)(0.0) = 1.0 ✓ Best!

Choose Temp for root split
\`\`\`

**Exam Tips:**
• IG = H(parent) - Weighted_Avg(H(children))
• Always compute weighted average (by subset size)
• Gini impurity alternative: 1 - Σ p_i²`,
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

**Deep Learning:** Many hidden layers can learn hierarchical features (edges → shapes → objects).

**Python: Simple Perceptron**
\`\`\`python
class Perceptron:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def predict(self, x):
        activation = np.dot(self.weights, x) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, X, y, lr=0.1, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                self.weights += lr * error * xi
                self.bias += lr * error
\`\`\`

**Why XOR Fails with Single Perceptron:**
\`\`\`
XOR truth table:
  (0,0) → 0    (0,1) → 1
  (1,0) → 1    (1,1) → 0

Try to find line w₁x₁ + w₂x₂ + b = 0 separating:
  Class 0: (0,0), (1,1)  vs  Class 1: (0,1), (1,0)

No single line can separate these!
XOR requires 2-layer network:
  Hidden: h₁ = AND, h₂ = OR
  Output: (h₂ AND NOT h₁) = XOR
\`\`\`

**Gradient Descent Trace (1 step):**
\`\`\`
Input x=[1,1], target y=1, weights w=[0.5,0.3], bias b=0.1

Forward: activation = 0.5(1) + 0.3(1) + 0.1 = 0.9 → output=1 ✓
Error = 1-1 = 0 (no update needed)

If target was 0: Error = 0-1 = -1
Update: w₁ = 0.5 + 0.1(-1)(1) = 0.4
        w₂ = 0.3 + 0.1(-1)(1) = 0.2
        b  = 0.1 + 0.1(-1) = 0.0
\`\`\`

**Exam Tips:**
• Perceptron: only linearly separable problems
• Learning rate too high → oscillation; too low → slow
• Activation functions: sigmoid, tanh, ReLU (ReLU avoids vanishing gradients)`,
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

**Sakoe-Chiba Band:** Constrains the warping path to stay near the diagonal, preventing extreme distortions.

**Python: DTW Algorithm**
\`\`\`python
import numpy as np

def dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)
    # Initialize cost matrix with infinity
    cost = np.full((n+1, m+1), np.inf)
    cost[0,0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            distance = abs(seq1[i-1] - seq2[j-1])
            cost[i,j] = distance + min(
                cost[i-1,j],    # insertion
                cost[i,j-1],    # deletion
                cost[i-1,j-1]   # match
            )
    return cost[n,m]
\`\`\`

**DTW Trace Example:**
\`\`\`
Seq1: [1, 2, 3]
Seq2: [1, 2, 2, 3]

Cost matrix:
       ∅   1   2   2   3
    ∅  0  ∞  ∞  ∞  ∞
    1  ∞  0  1  2  3
    2  ∞  1  1  1  2
    3  ∞  2  2  2  1

Optimal path (arrows):
  (0,0)→(1,1)→(2,2)→(2,3)→(3,4)
  Match 1-1, 2-2, skip, 3-3
  Total cost: 1

Euclidean would give: |3-4|=1, different length error!
DTW handles time warp elegantly.
\`\`\`

**Exam Tips:**
• DTW complexity: O(nm) time and space
• Sakoe-Chiba band reduces to O(nw) where w=window width
• DTW vs Euclidean: DTW allows temporal alignment
• Asymmetric: dtw(A,B) may align differently than dtw(B,A)`,
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
3. **Learning** — find best model parameters → **Baum-Welch** (EM algorithm)

**Python: Forward & Viterbi**
\`\`\`python
def forward(obs, states, start_p, trans_p, emit_p):
    alpha = [{s: start_p[s] * emit_p[s][obs[0]] for s in states}]
    for t in range(1, len(obs)):
        alpha.append({})
        for s in states:
            alpha[t][s] = sum(alpha[t-1][s0] * trans_p[s0][s]
                              for s0 in states) * emit_p[s][obs[t]]
    return sum(alpha[-1].values())

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{s: start_p[s] * emit_p[s][obs[0]] for s in states}]
    path = {s: [s] for s in states}
    for t in range(1, len(obs)):
        V.append({}); new_path = {}
        for s in states:
            probs = [(V[t-1][s0] * trans_p[s0][s] * emit_p[s][obs[t]], s0)
                     for s0 in states]
            max_prob, max_state = max(probs)
            V[t][s] = max_prob
            new_path[s] = path[max_state] + [s]
        path = new_path
    return max(V[-1], key=V[-1].get), path[max(V[-1], key=V[-1].get)]
\`\`\`

**Viterbi Trace: Weather/Umbrella**
\`\`\`
States: Rainy(R), Sunny(S)
Observations: Umbrella(U), No-umbrella(N)
Start: P(R)=0.6, P(S)=0.4
Transition: P(R|R)=0.7, P(S|R)=0.3, P(R|S)=0.4, P(S|S)=0.6
Emission: P(U|R)=0.9, P(U|S)=0.2

Obs: [U, U, N]

t=0 (U):
  V[R] = 0.6 × 0.9 = 0.54
  V[S] = 0.4 × 0.2 = 0.08

t=1 (U):
  V[R] = max(0.54×0.7, 0.08×0.4) × 0.9 = 0.34
  V[S] = max(0.54×0.3, 0.08×0.6) × 0.2 = 0.03

t=2 (N):
  V[R] = max(0.34×0.7, 0.03×0.4) × 0.1 = 0.024
  V[S] = max(0.34×0.3, 0.03×0.6) × 0.8 = 0.08 ✓

Best path: R → R → S
\`\`\`

**Exam Tips:**
• Forward: sum over all paths; Viterbi: max over paths
• Both use DP, O(T×N²) where T=time steps, N=states
• Baum-Welch: EM algorithm, iteratively improves parameters
• HMM assumes Markov property: future depends only on current state`,
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

**Residual Networks (ResNets):** Add "skip connections" that let gradients flow directly through the network, enabling training of very deep networks (100+ layers).

**Python: Convolution Operation**
\`\`\`python
def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    out_h, out_w = i_h - k_h + 1, i_w - k_w + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i,j] = np.sum(patch * kernel)
    return output
\`\`\`

**Convolution Trace:**
\`\`\`
Input (4×4):          Kernel (3×3):
  1 2 3 0              1  0 -1
  0 1 2 1              1  0 -1
  1 0 1 2              1  0 -1
  2 1 0 1

Output (2×2):
  Position (0,0): [1,2,3][0,1,2][1,0,1] ⊙ kernel
    = 1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1) + 1×1 + 0×0 + 1×(-1)
    = 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1 = -4

  After all positions: [-4, -2]
                       [-2, -1]

Max pooling (2×2): [-1] (takes max from 2×2 output)
\`\`\`

**Parameter Counting:**
Conv layer with 64 filters of size 3×3×3 (RGB):
  Params = 64 × (3×3×3 + 1) = 64 × 28 = 1,792

Fully connected 128→10:
  Params = 128×10 + 10 = 1,290

**Exam Tips:**
• Stride s: output size = (input - kernel + 2×padding) / s + 1
• Padding preserves spatial dimensions
• Pooling adds translation invariance, no learnable params
• ResNets solve vanishing gradients via skip connections`,
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

**Responsible AI:** Consider transparency, interpretability, bias, and fairness in AI systems.

**Regularization Comparison:**
\`\`\`
L2 Regularization:
  Loss = MSE + λ Σ w²
  Encourages small weights, smooth decision boundaries

Dropout (p=0.5):
  During training: randomly set 50% of neurons to 0
  During inference: use all neurons, scale by 0.5
  Forces redundancy, prevents co-adaptation

Batch Normalization:
  For each mini-batch: normalize → scale → shift
  x_norm = (x - μ_batch) / σ_batch
  output = γ × x_norm + β  (learnable γ, β)
  Stabilizes training, allows higher learning rates
\`\`\`

**Exam Tips:**
• Dropout: only during training, disable at test time
• Batch norm: different behavior train vs test (running stats)
• Transfer learning: freeze early layers, fine-tune later layers
• Attention solves RNN's long-distance dependency problem
• Transformer architecture: self-attention + positional encoding
• Vanishing gradients: deep RNNs struggle; LSTMs/GRUs help`,
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

**Value Iteration:** Repeatedly update V(s) = max_a [R(s) + γ Σ T(s,a,s')V(s')] until convergence. The optimal policy is then: pick the action that maximizes the right side.

**Python: Value Iteration**
\`\`\`python
def value_iteration(states, actions, T, R, gamma=0.9, theta=0.01):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(R[s] + gamma * sum(T(s,a,sp) * V[sp]
                       for sp in states) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = {}
    for s in states:
        policy[s] = max(actions, key=lambda a:
                        R[s] + gamma * sum(T(s,a,sp) * V[sp] for sp in states))
    return V, policy
\`\`\`

**Value Iteration Trace:**
\`\`\`
3 states: s0, s1, s2
Actions: Left, Right
R(s0)=0, R(s1)=1, R(s2)=10
γ=0.9

Transition T(s,a,s'):
  s0 --Right--> 0.8→s1, 0.2→s0
  s1 --Right--> 0.9→s2, 0.1→s1
  ...

Iteration 0: V=[0, 0, 0]

Iteration 1:
  V(s0) = max(R(s0) + 0.9[T*V]) = max(0, 0) = 0
  V(s1) = max(1 + 0.9×0, ...) = 1
  V(s2) = 10
  V=[0, 1, 10]

Iteration 2:
  V(s0) = max(0 + 0.9(0.8×1 + 0.2×0), ...) = 0.72
  V(s1) = max(1 + 0.9(0.9×10 + 0.1×1), ...) = 9.19
  V(s2) = 10
  V=[0.72, 9.19, 10]

Converges to: V*=[8.1, 9.1, 10]
Policy: π(s0)=Right, π(s1)=Right, π(s2)=any
\`\`\`

**Policy Iteration vs Value Iteration:**
• **Value Iteration:** Update values, extract policy at end
• **Policy Iteration:** Alternate policy evaluation + policy improvement
• PI often converges in fewer iterations but each iteration more expensive

**Exam Tips:**
• Bellman equation: V*(s) = max_a [R(s) + γ Σ T(s,a,s')V*(s')]
• Discount γ→1: far-sighted; γ→0: myopic
• Q-learning learns Q(s,a) without knowing T or R (model-free)
• Exploration vs exploitation: ε-greedy chooses random action with prob ε`,
        viz: "mdp"
      },
      {
        title: "POMDPs",
        content: `**Partially Observable MDPs** — the agent can't directly see the state!

Instead of knowing the exact state, the agent maintains a **belief state** — a probability distribution over possible states.

**Example:** A robot tour guide can't perfectly sense its location, so it maintains beliefs about where it might be and takes actions that both gather information AND make progress toward the goal.

POMDPs are much harder to solve than MDPs (PSPACE-complete!). Practical approaches use approximations like point-based solvers.

**Key insight:** Sometimes the best action is one that **reduces uncertainty** (information gathering) rather than directly pursuing the goal.

**POMDP Components:**
\`\`\`
Standard MDP: (S, A, T, R, γ)
POMDP adds:
• Observations O — what agent can perceive
• Observation function Z(s,a,o) — P(observation o | took action a, landed in state s)

Belief update (Bayes filter):
  b'(s') = η × Z(s',a,o) × Σ T(s,a,s') × b(s)
  where η normalizes to make b' a probability distribution
\`\`\`

**Simple Example: Tiger Problem**
\`\`\`
States: TigerLeft, TigerRight
Actions: Listen, OpenLeft, OpenRight
Observations: HearLeft, HearRight

Listen: gets noisy observation (85% accurate)
Open door: -100 if tiger, +10 if treasure

Belief: b = [P(TigerLeft), P(TigerRight)] = [0.5, 0.5]

Action: Listen → Observation: HearLeft
Update belief:
  P(TL | HearLeft) = 0.85 × 0.5 / normalizer ≈ 0.85
  P(TR | HearLeft) = 0.15 × 0.5 / normalizer ≈ 0.15

New belief [0.85, 0.15] → confidence increased!
Best action now: OpenRight (low tiger probability)
\`\`\`

**Exam Tips:**
• POMDP belief state is continuous (probability distribution)
• Optimal policy maps beliefs to actions: π(b) → a
• Information gathering actions reduce entropy of belief
• PSPACE-complete: exponentially harder than MDP`,
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
    { q: "DFS uses which data structure?", o: ["Queue (FIFO)", "Stack (LIFO)", "Priority Queue", "Deque"], a: 1, e: "DFS uses a stack (LIFO) — last in, first out — diving deep before backtracking." },
    { q: "What is the space complexity of DFS?", o: ["O(b^d)", "O(b^m)", "O(bm)", "O(d)"], a: 2, e: "DFS stores one path from root to current node: O(bm) where b = branching factor, m = max depth." },
    { q: "If A* uses h(n) = 0 for all nodes, it behaves like:", o: ["DFS", "BFS", "UCS", "Greedy Best-First"], a: 2, e: "With h(n)=0, f(n) = g(n) + 0 = g(n), which is exactly UCS — expand by cheapest path cost." },
    { q: "Which search strategy is complete but NOT optimal?", o: ["BFS", "UCS", "DFS on finite graphs", "A* with admissible h"], a: 2, e: "DFS on finite graphs will find a solution (complete) but may find a deep/expensive one first (not optimal)." },
    { q: "A consistent heuristic guarantees that A*:", o: ["Runs in O(n) time", "Never re-expands a node", "Uses no memory", "Finds all solutions"], a: 1, e: "Consistency (h(n) ≤ c(n,n') + h(n')) ensures f-values are non-decreasing along paths, so no re-expansion." },
    { q: "Iterative Deepening DFS combines the advantages of:", o: ["BFS completeness + DFS memory", "UCS optimality + BFS speed", "A* heuristics + DFS depth", "Greedy speed + UCS cost"], a: 0, e: "IDS does repeated depth-limited DFS: BFS-like completeness/optimality with DFS-like O(bd) memory." },
    { q: "What distinguishes Graph Search from Tree Search?", o: ["Uses a priority queue", "Tracks an explored set", "Always finds optimal solutions", "Uses heuristics"], a: 1, e: "Graph Search maintains an explored set to avoid revisiting states, preventing infinite loops in cyclic graphs." },
  ],
  1: [
    { q: "Main problem with basic hill climbing?", o: ["Too slow", "Gets stuck at local maxima", "Too much memory", "Can't handle continuous spaces"], a: 1, e: "Hill climbing always moves to a better neighbor, so it gets stuck at local maxima." },
    { q: "As temperature decreases in simulated annealing:", o: ["More random moves accepted", "Fewer bad moves accepted", "Algorithm restarts", "Step size increases"], a: 1, e: "As T drops, probability of accepting worse solutions decreases, search becomes greedy." },
    { q: "What does 'crossover' do in a genetic algorithm?", o: ["Randomly changes one individual", "Combines parts of two parents", "Selects the fittest", "Removes the weakest"], a: 1, e: "Crossover combines genetic material from two parents to create a child." },
    { q: "Acceptance probability for a worse solution in SA?", o: ["Always 0", "Always 1", "e^(ΔE/T)", "1/T"], a: 2, e: "Probability is e^(ΔE/T) where ΔE < 0. Higher T → higher acceptance." },
    { q: "Random-restart hill climbing helps by:", o: ["Using a priority queue", "Trying many starting points", "Increasing step size", "Adding backtracking"], a: 1, e: "Multiple random starts increase the chance of landing near the global optimum." },
    { q: "What is a 'plateau' in local search?", o: ["Global maximum", "Area where all neighbors have equal value", "Local minimum", "Area with no neighbors"], a: 1, e: "On a plateau, no neighbor is better so hill climbing stalls — all moves look equally good." },
    { q: "In SA, what happens when T is very high?", o: ["Only improvements accepted", "Almost all moves accepted", "Algorithm terminates", "Only the best neighbor chosen"], a: 1, e: "At high T, e^(ΔE/T) ≈ 1 even for bad moves, so SA behaves like random walk — maximum exploration." },
    { q: "What is the role of 'mutation' in GAs?", o: ["Combine two parents", "Maintain diversity in population", "Select best individuals", "Sort the population"], a: 1, e: "Mutation introduces random changes to prevent the population from converging too early on a suboptimal solution." },
    { q: "Which cooling schedule guarantees finding the global optimum?", o: ["Linear", "Geometric", "Logarithmic", "Constant temperature"], a: 2, e: "Logarithmic cooling T = T₀/ln(t+1) guarantees convergence, but is too slow for practical use." },
    { q: "In genetic algorithms, 'fitness' determines:", o: ["Mutation rate", "Which individuals reproduce", "Population size", "Number of generations"], a: 1, e: "Fitter individuals are more likely to be selected as parents, passing their traits to the next generation." },
    { q: "At T=0 in simulated annealing, the algorithm behaves like:", o: ["Random walk", "Hill climbing", "BFS", "Genetic algorithm"], a: 1, e: "At T=0, e^(ΔE/0) = 0 for any worse move, so only improvements are accepted — exactly hill climbing." },
    { q: "A 'ridge' problem in hill climbing means:", o: ["Solution is at the bottom", "Narrow peak where no single step improves", "Too many neighbors", "Search space is flat"], a: 1, e: "A ridge is a narrow elevated region where moving in any single dimension goes downhill, but diagonal moves would help." },
  ],
  2: [
    { q: "In Minimax, the MAX player tries to:", o: ["Minimize score", "Maximize score", "Reach a draw", "Minimize depth"], a: 1, e: "MAX wants highest payoff; MIN tries to minimize it." },
    { q: "Alpha-beta pruning best case reduces branching to:", o: ["b/2", "√b", "b²", "log(b)"], a: 1, e: "Perfect move ordering: O(b^(m/2)) nodes — square root of branching factor effectively doubles searchable depth." },
    { q: "When does alpha-beta pruning occur?", o: ["α < β", "α ≥ β", "α = 0", "β = 0"], a: 1, e: "When α ≥ β, the branch can't affect the final decision — prune remaining children." },
    { q: "The horizon effect occurs when:", o: ["Tree is too wide", "Limited depth hides inevitable outcomes", "Eval function is perfect", "Pruning is too aggressive"], a: 1, e: "Limited depth can push bad outcomes beyond the search horizon — the AI delays but can't avoid them." },
    { q: "In Expectimax, chance nodes compute:", o: ["Maximum of children", "Minimum of children", "Weighted average", "Median"], a: 2, e: "Chance nodes compute expected value — weighted average over outcomes based on probabilities." },
    { q: "Does alpha-beta pruning change the minimax result?", o: ["Yes, it finds better moves", "No, same result with less work", "Only with good ordering", "Only at shallow depths"], a: 1, e: "Alpha-beta always returns the same value as minimax — it just skips branches that can't change the outcome." },
    { q: "Alpha (α) is updated at which type of node?", o: ["MIN nodes", "MAX nodes", "Chance nodes", "All nodes"], a: 1, e: "α tracks the best value MAX can guarantee. It's updated at MAX nodes when a better child is found." },
    { q: "What is quiescent search?", o: ["Searching quietly", "Extending search past noisy positions", "Searching with no heuristic", "Limiting search depth"], a: 1, e: "Quiescent search extends evaluation past 'noisy' positions (captures, checks) to avoid misjudging volatile states." },
    { q: "Iterative deepening in game trees provides:", o: ["Optimal pruning", "Anytime best move + better move ordering", "Infinite depth search", "Lower memory usage"], a: 1, e: "ID always has a best-move-so-far ready when time runs out, and previous iterations help order moves for better pruning." },
    { q: "Why can't alpha-beta pruning be used with Expectimax?", o: ["Too slow", "Chance nodes need all children for expected value", "No MIN nodes exist", "Pruning always helps"], a: 1, e: "Chance nodes must evaluate ALL children to compute the weighted average — you can't skip any." },
    { q: "Time complexity of Minimax with branching factor b and depth m?", o: ["O(b + m)", "O(b × m)", "O(b^m)", "O(m^b)"], a: 2, e: "Minimax explores every node in the game tree: b children at each of m levels = O(b^m) total nodes." },
    { q: "An evaluation function is used when:", o: ["The game has no rules", "We can't search to terminal states", "The tree is too shallow", "We have perfect information"], a: 1, e: "Eval functions estimate position quality at non-terminal nodes when search must stop before reaching the end of the game." },
  ],
  3: [
    { q: "Three main components of a CSP?", o: ["Nodes, Edges, Weights", "Variables, Domains, Constraints", "States, Actions, Rewards", "Inputs, Outputs, Layers"], a: 1, e: "CSP = Variables + Domains + Constraints." },
    { q: "Forward checking does what after assignment?", o: ["Backtracks", "Removes inconsistent values from neighbors", "Assigns all remaining", "Changes constraints"], a: 1, e: "Forward checking removes values from neighbors' domains that violate constraints." },
    { q: "MRV heuristic chooses:", o: ["Most legal values", "Fewest legal values", "Most constraints", "Random variable"], a: 1, e: "MRV picks the most constrained variable — 'fail-first' strategy." },
    { q: "Arc consistency ensures:", o: ["All assigned", "Every value has a compatible neighbor value", "Solution is unique", "No backtracking needed"], a: 1, e: "Each value in one domain must have a compatible value in constrained neighbor's domain." },
    { q: "After forward checking detects an empty domain, the algorithm should:", o: ["Continue assigning", "Backtrack immediately", "Try AC-3", "Reset all domains"], a: 1, e: "Empty domain means no valid assignment possible — backtrack to try different value." },
    { q: "AC-3 maintains consistency by:", o: ["Checking only assigned variables", "Checking all pairs of constrained variables", "Only checking neighbors", "Random constraint checks"], a: 1, e: "AC-3 uses a queue to propagate constraints across all arcs in the constraint graph." },
    { q: "When multiple variables have the same MRV count, which tie-breaker is best?", o: ["Random selection", "Alphabetical order", "Degree heuristic (most constraints)", "Least constraining value"], a: 2, e: "Degree heuristic picks the variable involved in most constraints on remaining unassigned variables." },
    { q: "LCV (Least Constraining Value) heuristic chooses values that:", o: ["Rule out fewest options for neighbors", "Have been tried least", "Are smallest numerically", "Appear first in domain"], a: 0, e: "LCV tries the value that leaves the most flexibility for neighboring variables." },
    { q: "Backtracking without any heuristics has time complexity:", o: ["O(n)", "O(n²)", "O(d^n)", "O(n×d)"], a: 2, e: "In worst case, try all d values for each of n variables: O(d^n) — exponential!" },
    { q: "Forward checking is:", o: ["More powerful than AC-3", "Less powerful than AC-3", "Equivalent to AC-3", "Unrelated to AC-3"], a: 1, e: "FC only checks direct neighbors of assigned var. AC-3 propagates further through constraint graph." },
    { q: "In CSP, backtracking is guaranteed to:", o: ["Find optimal solution", "Find a solution if one exists", "Run in polynomial time", "Never revisit states"], a: 1, e: "Backtracking is complete for CSPs — will find solution if one exists (or prove none exist)." },
    { q: "Which is NOT a valid CSP solving approach?", o: ["Backtracking with FC", "Min-conflicts local search", "AC-3 preprocessing", "A* search"], a: 3, e: "A* is for pathfinding, not CSPs. CSPs use backtracking, local search, or constraint propagation." },
    { q: "Min-conflicts local search for CSPs:", o: ["Always finds solution", "Can get stuck in local minima", "Slower than backtracking", "Requires arc consistency"], a: 1, e: "Local search like min-conflicts can get stuck and may not find solution even if one exists." },
    { q: "A constraint graph where each node connects to k others has induced width:", o: ["Always k", "At most k", "At least k", "k²"], a: 1, e: "Induced width depends on elimination order, bounded by max clique size (at most k for k-connected graph)." },
    { q: "Which CSP structure allows tree-search algorithms to run in O(n·d²)?", o: ["Complete graphs", "Trees", "Dense graphs", "Cyclic graphs"], a: 1, e: "Tree-structured CSPs can be solved in linear time (O(n·d²)) using topological ordering." },
    { q: "Constraint propagation differs from backtracking search in that it:", o: ["Never assigns variables", "Prunes domains before/during search", "Only works on binary CSPs", "Requires a heuristic"], a: 1, e: "Propagation (like AC-3) removes inconsistent values from domains, narrowing search space." },
  ],
  4: [
    { q: "If P(A) = 0.3, what is P(¬A)?", o: ["0.3", "0.7", "0.09", "1.3"], a: 1, e: "P(¬A) = 1 - P(A) = 0.7" },
    { q: "Bayes' Rule: P(A|B) equals:", o: ["P(B|A) × P(A)", "P(B|A) × P(A) / P(B)", "P(A) × P(B)", "P(A) + P(B)"], a: 1, e: "Bayes' Rule: P(A|B) = P(B|A) × P(A) / P(B)." },
    { q: "Events A and B are independent if:", o: ["P(A ∧ B) = 0", "P(A|B) = P(A)", "P(A) + P(B) = 1", "P(A|B) = P(B|A)"], a: 1, e: "Independence: knowing B gives no info about A." },
    { q: "Positive cancer test but low P(cancer) because:", o: ["Test unreliable", "Base rate is very low", "Bayes doesn't apply", "No false positives"], a: 1, e: "Low prior overwhelms test accuracy — the base rate fallacy." },
    { q: "If P(A|B) = 0.6 and P(B) = 0.5, what is P(A ∧ B)?", o: ["0.11", "0.3", "1.1", "0.6"], a: 1, e: "P(A ∧ B) = P(A|B) × P(B) = 0.6 × 0.5 = 0.3" },
    { q: "The chain rule of probability states:", o: ["P(A,B) = P(A) + P(B)", "P(A,B) = P(A|B)×P(B)", "P(A,B) = P(A)×P(B)", "P(A,B) = P(A)/P(B)"], a: 1, e: "Chain rule: P(A,B) = P(A|B)×P(B) = P(B|A)×P(A). Generalizes to any number of variables." },
    { q: "Marginalization computes P(A) by:", o: ["P(A) = max_b P(A,b)", "P(A) = Σ_b P(A,b)", "P(A) = P(A,b)/P(b)", "P(A) = 1 - P(¬A)"], a: 1, e: "Marginalization sums over all values: P(A) = Σ_b P(A,B=b)." },
    { q: "If A and B are independent, then P(A ∧ B) equals:", o: ["P(A) + P(B)", "P(A) × P(B)", "P(A|B)", "P(A) - P(B)"], a: 1, e: "Independence means P(A ∧ B) = P(A) × P(B) — probabilities multiply." },
    { q: "Total probability theorem: P(A) = ?", o: ["Σ P(A|B_i)", "Σ P(A|B_i)×P(B_i)", "Σ P(B_i|A)", "P(A|B)×P(B)"], a: 1, e: "Total prob: P(A) = Σ_i P(A|B_i)×P(B_i) where B_i partition the space." },
    { q: "Conditional independence P(A⊥B|C) means:", o: ["P(A|C) = P(B|C)", "P(A|B,C) = P(A|C)", "P(A,B,C) = 0", "P(A|C) = 0"], a: 1, e: "Conditionally independent given C: P(A|B,C) = P(A|C). Knowing B doesn't help once you know C." },
    { q: "If P(Rain) = 0.3 and P(Traffic|Rain) = 0.8, P(Traffic|¬Rain) = 0.2, find P(Traffic):", o: ["0.5", "0.38", "0.24", "0.8"], a: 1, e: "Total prob: P(T) = 0.8×0.3 + 0.2×0.7 = 0.24 + 0.14 = 0.38" },
    { q: "In the cancer test, why is P(C|+) so low despite high test accuracy?", o: ["Math error", "Test is useless", "Low base rate P(C)", "High false negative"], a: 2, e: "Base rate fallacy: P(C)=1% is so low that even accurate tests produce many false positives." },
    { q: "If P(A)=0.4, P(B)=0.5, P(A∧B)=0.3, are A and B independent?", o: ["Yes", "No", "Not enough info", "Only if P(A|B)=P(B)"], a: 1, e: "No! If independent, P(A∧B) should equal P(A)×P(B) = 0.4×0.5 = 0.2 ≠ 0.3" },
    { q: "Normalizing probabilities means:", o: ["Subtracting mean", "Scaling so they sum to 1", "Setting largest to 1", "Dividing by variance"], a: 1, e: "Normalization scales values so Σ P(X=x) = 1, ensuring valid probability distribution." },
    { q: "Which is TRUE about joint distributions?", o: ["Always independent", "Encode all correlations", "Simpler than conditionals", "Only for binary vars"], a: 1, e: "Joint P(A,B,...,Z) contains complete information about all correlations and dependencies." },
    { q: "For three events, the chain rule gives P(A,B,C) as:", o: ["P(A)×P(B)×P(C)", "P(A|B,C)×P(B|C)×P(C)", "P(A)+P(B)+P(C)", "P(A|B)×P(B)"], a: 1, e: "Chain rule: P(A,B,C) = P(A|B,C)×P(B|C)×P(C) — condition each on all previous." },
  ],
  5: [
    { q: "In a Bayes Net, each node depends only on:", o: ["All other nodes", "Its parent nodes", "Its child nodes", "Sibling nodes"], a: 1, e: "Each variable is conditionally independent of non-descendants given parents." },
    { q: "In collider A → C ← B, when C is observed:", o: ["A,B become independent", "A,B become dependent", "Always independent", "C becomes independent"], a: 1, e: "Observing collider C 'opens' the path — explaining away." },
    { q: "In chain A → B → C, A and C independent given B?", o: ["Yes — B blocks flow", "No — always dependent", "Only if B unobserved", "Depends on CPT"], a: 0, e: "Observing B d-separates A from C, blocking information flow." },
    { q: "Variable elimination improves on enumeration by:", o: ["Random sampling", "Reusing intermediate computations", "Ignoring hidden vars", "Only MAP estimates"], a: 1, e: "VE avoids redundant computation by combining and marginalizing factors smartly." },
    { q: "In fork A ← C → B, when is the path active?", o: ["Always", "Only when C observed", "Only when C NOT observed", "Never"], a: 2, e: "Fork is active (dependent) when common cause C is unobserved. Observing C blocks the path." },
    { q: "The Markov blanket of a node consists of:", o: ["Only parents", "Parents + children", "Parents + children + children's other parents", "All neighbors"], a: 2, e: "Markov blanket = parents + children + spouses (other parents of children). Node independent of all else given blanket." },
    { q: "Observing a descendant of a collider:", o: ["Blocks the path", "Opens the path like observing the collider", "Has no effect", "Makes all nodes independent"], a: 1, e: "Observing a descendant of a collider activates explaining away — same effect as observing the collider itself." },
    { q: "A Bayes Net represents which factorization?", o: ["P = Σ P(Xᵢ)", "P = Π P(Xᵢ|Parents(Xᵢ))", "P = Π P(Xᵢ)", "P = max P(Xᵢ)"], a: 1, e: "Joint = product of local CPTs: P(X₁,...,Xₙ) = Π P(Xᵢ|Parents(Xᵢ))." },
    { q: "Rejection sampling rejects a sample when:", o: ["Probability too low", "Evidence doesn't match", "Cycle detected", "Domain too large"], a: 1, e: "Rejection sampling discards samples that don't match the observed evidence values." },
    { q: "Likelihood weighting improves on rejection by:", o: ["Using MCMC", "Fixing evidence and weighting samples", "Fewer variables", "No normalization needed"], a: 1, e: "LW fixes evidence vars to observed values and weights by P(evidence), avoiding wasteful rejection." },
    { q: "In Gibbs sampling, each iteration resamples:", o: ["All variables at once", "One variable given its Markov blanket", "Evidence variables", "Root nodes only"], a: 1, e: "Gibbs resamples one non-evidence variable at a time conditioned on current values of its Markov blanket." },
    { q: "Variable elimination complexity depends mainly on:", o: ["Number of nodes", "Elimination order", "Number of edges", "CPT size alone"], a: 1, e: "Bad elimination order → huge intermediate factors. Good order can be exponentially faster!" },
    { q: "Which inference method is exact?", o: ["Rejection sampling", "Likelihood weighting", "Variable elimination", "Gibbs sampling"], a: 2, e: "Variable elimination (and enumeration) are exact. Sampling methods are approximate." },
    { q: "Explaining away is best described as:", o: ["Common cause explains children", "Competing causes become anti-correlated given shared effect", "Effects cause parents", "Independent causes stay independent"], a: 1, e: "When a shared effect is observed, knowing one cause makes others less likely — competing explanations." },
  ],
  6: [
    { q: "In kNN, as k increases:", o: ["Boundary more complex", "Boundary smoother", "Training slower", "Distance metric changes"], a: 1, e: "Larger k = more voters = smoother, less complex boundary." },
    { q: "Naive Bayes assumes features are:", o: ["Correlated", "Conditionally independent given class", "Normally distributed", "Equally important"], a: 1, e: "The 'naive' assumption: features independent given the class label." },
    { q: "Information Gain measures:", o: ["Tree accuracy", "Entropy reduction after split", "Tree depth", "Features used"], a: 1, e: "IG = Entropy(parent) - weighted Entropy(children)." },
    { q: "A single perceptron CANNOT learn:", o: ["AND", "OR", "XOR", "NOT"], a: 2, e: "XOR isn't linearly separable — no single line can separate the examples." },
    { q: "Backpropagation works by:", o: ["Random weight adjustment", "Propagating errors backward via gradient descent", "Adding layers", "Removing neurons"], a: 1, e: "Backprop computes gradients via chain rule and updates weights." },
    { q: "Why must kNN features be normalized?", o: ["Speeds up computation", "Prevents features with large ranges from dominating distance", "Required by algorithm", "Reduces memory"], a: 1, e: "Without normalization, a feature with range [0,1000] dominates one with range [0,1] in Euclidean distance." },
    { q: "Laplace smoothing in Naive Bayes:", o: ["Removes outliers", "Prevents zero probabilities", "Speeds up training", "Increases accuracy"], a: 1, e: "Add 1 to all counts to avoid P(feature|class)=0, which would make entire posterior zero." },
    { q: "Entropy of a pure node (all same class):", o: ["1", "0", "0.5", "Infinity"], a: 1, e: "H = -1×log₂(1) = 0. No uncertainty when all examples are same class." },
    { q: "Overfitting vs underfitting in ML:", o: ["Both mean low training error", "Overfitting: high train accuracy, low test; Underfitting: low both", "Both mean high test error", "Overfitting always better"], a: 1, e: "Overfitting: model memorizes training data. Underfitting: model too simple to capture patterns." },
    { q: "Cross-validation is used to:", o: ["Speed up training", "Select hyperparameters and estimate generalization", "Reduce overfitting", "Increase dataset size"], a: 1, e: "Split data into folds, train on some and validate on others to tune hyperparameters like k in kNN." },
    { q: "Bias-variance tradeoff:", o: ["High bias = overfitting", "High variance = underfitting", "High bias = underfitting, high variance = overfitting", "Unrelated concepts"], a: 2, e: "Bias: error from wrong assumptions (underfitting). Variance: error from sensitivity to training data (overfitting)." },
    { q: "Gradient descent learning rate too high:", o: ["Slow convergence", "Oscillation or divergence", "Perfect convergence", "No effect"], a: 1, e: "Large steps can overshoot minimum, causing loss to bounce around or increase." },
    { q: "Why use log probabilities in Naive Bayes?", o: ["Faster computation", "Prevents numerical underflow", "Required by Bayes rule", "Improves accuracy"], a: 1, e: "Multiplying many small probabilities causes underflow. Log converts multiplication to addition: log(ab)=log(a)+log(b)." },
    { q: "Random Forest reduces overfitting by:", o: ["Using single deep tree", "Averaging predictions from multiple trees on random subsets", "Pruning aggressively", "Using shallow trees only"], a: 1, e: "Bootstrap aggregating (bagging) + random feature subsets creates diverse trees; averaging reduces variance." },
  ],
  7: [
    { q: "Why is Euclidean distance bad for time series?", o: ["Too slow", "Can't handle different lengths", "Fails with time shifts", "Only works in 2D"], a: 2, e: "Point-by-point comparison fails when signals are shifted in time." },
    { q: "Viterbi algorithm finds:", o: ["Observation probability", "Most likely hidden state sequence", "Best model parameters", "Emission probabilities"], a: 1, e: "Viterbi uses DP to find the most likely state sequence." },
    { q: "Baum-Welch is used for:", o: ["Decoding states", "Computing observation probability", "Training HMM parameters", "Time warping"], a: 2, e: "Baum-Welch (EM) iteratively improves transition and emission probabilities." },
    { q: "HMM emission probabilities describe:", o: ["P(next state|current)", "P(observation|state)", "P(state|observation)", "P(starting state)"], a: 1, e: "Emission probs define how likely each observation is given the hidden state." },
    { q: "DTW time complexity for sequences of length n and m:", o: ["O(n+m)", "O(n×m)", "O(n²)", "O(log n)"], a: 1, e: "DTW fills an n×m cost matrix, visiting each cell once: O(nm) time." },
    { q: "Sakoe-Chiba band in DTW:", o: ["Speeds up by constraining warping path near diagonal", "Improves accuracy", "Handles different lengths", "Required for correctness"], a: 0, e: "Constrains path to window around diagonal, reducing complexity from O(nm) to O(nw) where w=width." },
    { q: "Forward algorithm vs Viterbi in HMMs:", o: ["Forward sums paths; Viterbi finds max path", "Both find max path", "Forward finds max; Viterbi sums", "Identical algorithms"], a: 0, e: "Forward: P(obs|model) = sum over all paths. Viterbi: most likely single path using max instead of sum." },
    { q: "HMM components include all EXCEPT:", o: ["Transition probabilities", "Emission probabilities", "Gradient descent", "Initial state distribution"], a: 2, e: "HMM has states, transitions, emissions, and initial probs. Gradient descent is a training method, not a component." },
    { q: "Baum-Welch algorithm is a form of:", o: ["Gradient descent", "Expectation-Maximization (EM)", "Dynamic programming", "Greedy search"], a: 1, e: "Baum-Welch is EM for HMMs: E-step computes expected counts, M-step updates parameters." },
    { q: "Markov assumption in HMMs:", o: ["All states equally likely", "Future state depends only on current state", "Observations independent", "No hidden states"], a: 1, e: "P(s_t+1 | s_1...s_t) = P(s_t+1 | s_t). Future depends only on present, not past history." },
    { q: "DTW is asymmetric because:", o: ["Distance metric not symmetric", "Different path for dtw(A,B) vs dtw(B,A)", "Only works one direction", "Requires ordered sequences"], a: 1, e: "Warping path alignment can differ depending on which sequence is reference vs query." },
    { q: "Forward algorithm complexity for T observations, N states:", o: ["O(T)", "O(N²)", "O(T×N²)", "O(2^T)"], a: 2, e: "For each time step (T), compute alpha for each state (N), summing over all previous states (N): O(TN²)." },
    { q: "HMMs are useful for:", o: ["Static classification", "Sequence modeling (speech, DNA, gestures)", "Image recognition", "Clustering"], a: 1, e: "HMMs model temporal sequences where hidden states generate observable outputs over time." },
    { q: "Difference between DTW and Euclidean distance:", o: ["DTW faster", "DTW allows temporal alignment; Euclidean point-to-point", "Euclidean more accurate", "No difference"], a: 1, e: "DTW warps time axis for best alignment. Euclidean rigidly compares corresponding indices." },
  ],
  8: [
    { q: "Advantage of conv layers over fully connected for images?", o: ["Faster training", "Parameter sharing — same filter everywhere", "No activations needed", "1D only"], a: 1, e: "Conv layers share filters, drastically reducing parameters." },
    { q: "Pooling layers in a CNN:", o: ["Add parameters", "Downsample feature maps", "Increase resolution", "Apply activation"], a: 1, e: "Pooling reduces spatial dimensions, adding translation invariance." },
    { q: "ResNets solve what problem?", o: ["Overfitting", "Vanishing gradients via skip connections", "Slow inference", "High memory"], a: 1, e: "Skip connections let gradients flow directly, enabling very deep networks." },
    { q: "Dropout works by:", o: ["Removing layers", "Randomly zeroing neurons during training", "Reducing learning rate", "Adding input noise"], a: 1, e: "Dropout deactivates random neurons, preventing co-adaptation." },
    { q: "CNN output size with stride s, padding p, kernel k, input n:", o: ["(n-k)/s+1", "(n-k+2p)/s+1", "n/s", "n-k"], a: 1, e: "Output = (input - kernel + 2×padding) / stride + 1. Example: (32-3+2)/1+1 = 32 (same size with p=1)." },
    { q: "Batch normalization purpose:", o: ["Add noise", "Normalize layer inputs for stable training", "Reduce parameters", "Replace activation functions"], a: 1, e: "BN normalizes inputs to each layer, reducing internal covariate shift and allowing higher learning rates." },
    { q: "Transfer learning in deep learning:", o: ["Train from scratch always", "Use pretrained model, fine-tune on new task", "Only for small datasets", "Requires same output classes"], a: 1, e: "Pretrain on large dataset (ImageNet), freeze early layers, fine-tune later layers for specific task." },
    { q: "Attention mechanism allows models to:", o: ["Train faster", "Focus on relevant parts of input", "Use less memory", "Avoid overfitting"], a: 1, e: "Attention learns weights to focus on important input elements, crucial for seq2seq and transformers." },
    { q: "ReLU activation advantage over sigmoid:", o: ["Always positive", "Avoids vanishing gradient problem", "More complex", "Slower"], a: 1, e: "ReLU: f(x)=max(0,x). Gradient is 1 for x>0, avoiding saturation unlike sigmoid which saturates at 0 and 1." },
    { q: "Vanishing gradient problem occurs when:", o: ["Learning rate too high", "Gradients become extremely small in deep networks", "Too much data", "Overfitting"], a: 1, e: "In deep networks with sigmoid/tanh, gradients multiply through layers, shrinking exponentially. ReLU and ResNets help." },
    { q: "Difference between L1 and L2 regularization:", o: ["L1 creates sparse weights; L2 small weights", "L1 always better", "L2 creates sparsity", "No difference"], a: 0, e: "L1: |w| penalty encourages exact zeros (sparsity). L2: w² penalty encourages small weights but rarely exactly zero." },
    { q: "Convolutional layer parameters for 64 filters, 3×3, input depth 32:", o: ["64×3×3=576", "64×(3×3×32+1)=18,496", "3×3×32=288", "64×32=2,048"], a: 1, e: "Each filter: 3×3×32 weights + 1 bias = 289. Total: 64 filters × 289 = 18,496 parameters." },
    { q: "Pooling vs strided convolution:", o: ["Pooling has parameters; strided conv doesn't", "Both downsample; pooling no params, strided conv has params", "Identical", "Pooling slower"], a: 1, e: "Max/avg pooling downsamples with no learnable params. Strided conv downsamples while learning filters." },
    { q: "Word2Vec learns:", o: ["Word frequencies", "Dense vector representations where similar words are close", "Grammar rules", "Sentence structure"], a: 1, e: "Word2Vec embeds words in continuous space such that semantic/syntactic similarity reflects geometric proximity." },
  ],
  9: [
    { q: "In an MDP, a policy π(s) maps:", o: ["Actions to states", "States to actions", "States to rewards", "Actions to rewards"], a: 1, e: "A policy tells the agent what action to take in each state." },
    { q: "Discount factor γ controls:", o: ["Transition probs", "How much future rewards are valued", "Number of states", "Exploration rate"], a: 1, e: "γ determines present value of future rewards. γ→1 = long-term thinking." },
    { q: "Value Iteration converges by:", o: ["Random exploration", "Repeatedly applying Bellman equation", "Gradient descent", "Monte Carlo"], a: 1, e: "VI repeatedly applies V(s) = max_a [R(s) + γ Σ T(s,a,s')V(s')]." },
    { q: "In a POMDP, the agent maintains a:", o: ["Complete state", "Belief state (probability distribution)", "Single guess", "Action history"], a: 1, e: "Agent can't see true state, so maintains a belief distribution." },
    { q: "Bellman equation for optimal value function:", o: ["V(s) = R(s)", "V*(s) = max_a [R(s) + γ Σ T(s,a,s')V*(s')]", "V(s) = sum of all rewards", "V(s) = γ×R(s)"], a: 1, e: "Optimal value: immediate reward R(s) plus discounted expected value of next states, maximizing over actions." },
    { q: "Policy Iteration vs Value Iteration:", o: ["Identical algorithms", "PI alternates evaluation/improvement; VI updates values directly", "PI always slower", "VI can't find optimal policy"], a: 1, e: "PI: evaluate policy, improve policy, repeat. VI: update all values, extract policy at end. PI often fewer iterations." },
    { q: "Q-learning is:", o: ["Model-based", "Model-free: learns Q(s,a) without knowing T or R", "Requires full MDP specification", "Only for deterministic environments"], a: 1, e: "Q-learning learns action values from experience without transition/reward models: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]." },
    { q: "Exploration vs exploitation tradeoff:", o: ["Only explore", "Only exploit", "Balance trying new actions vs using best known action", "Unimportant in RL"], a: 2, e: "Exploration: try unknown actions to discover better options. Exploitation: use best known action. ε-greedy balances both." },
    { q: "ε-greedy strategy:", o: ["Always pick best action", "Pick random action with prob ε, best action with prob 1-ε", "Pick worst action sometimes", "Requires gradient"], a: 1, e: "With probability ε, explore by choosing random action. Otherwise, exploit by choosing argmax Q(s,a)." },
    { q: "Discount factor γ=0 means:", o: ["Ignore all rewards", "Only care about immediate reward", "Care about all future rewards equally", "Invalid MDP"], a: 1, e: "γ=0: V(s) = R(s), agent myopic. γ→1: agent far-sighted, values distant rewards." },
    { q: "Model-based vs model-free RL:", o: ["Model-based learns T,R; model-free learns values directly", "Both identical", "Model-free always better", "Model-based doesn't use rewards"], a: 0, e: "Model-based: learn transition T(s,a,s') and reward R, then plan. Model-free: learn values/policy from experience without model." },
    { q: "POMDP complexity compared to MDP:", o: ["Same", "POMDP easier", "POMDP much harder (PSPACE-complete)", "Only MDP solvable"], a: 2, e: "POMDP belief state is continuous probability distribution, exponentially harder than MDP's discrete states." },
    { q: "Q-learning update rule:", o: ["Q(s,a) = R(s)", "Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]", "Q(s,a) ← r", "Q(s,a) ← γQ(s,a)"], a: 1, e: "Temporal difference update: old value (1-α)Q + learning rate α × [observed r + discounted max future Q]." },
    { q: "Value function V(s) vs Q-function Q(s,a):", o: ["V is state value; Q is state-action value", "Identical", "Q always larger", "V for POMDP only"], a: 0, e: "V(s): expected return from state s. Q(s,a): expected return from state s taking action a. V(s)=max_a Q(s,a)." },
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

// ============== COLLAPSIBLE CODE BLOCK ==============
function CollapsibleCode({ lang, code }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-2 rounded-lg overflow-hidden" style={{ background: '#1e1e2e', border: '1px solid #313244' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 12px', fontSize: 11, color: '#a6adc8', background: '#181825', border: 'none', cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace" }}
      >
        <span>{lang ? `${lang}` : 'code'}</span>
        <span style={{ fontSize: 10, opacity: 0.7 }}>{open ? '▲ collapse' : '▼ expand'}</span>
      </button>
      {open && (
        <pre style={{ padding: '12px', margin: 0, fontSize: 11.5, lineHeight: 1.5, color: '#cdd6f4', fontFamily: "'JetBrains Mono', 'Fira Code', monospace", overflowX: 'auto', whiteSpace: 'pre' }}>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

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
    // Split by code blocks first, then process each part
    const parts = text.split(/(```[\s\S]*?```)/g);

    return parts.map((part, partIdx) => {
      if (part.startsWith('```')) {
        const lines = part.split('\n');
        const lang = lines[0].replace('```', '').trim();
        const code = lines.slice(1, -1).join('\n');
        return <CollapsibleCode key={partIdx} lang={lang} code={code} />;
      }

      return part.split('\n').map((line, i) => {
        line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        line = line.replace(/\*(.+?)\*/g, '<em>$1</em>');
        line = line.replace(/`([^`]+)`/g, '<code style="background:#e8e8e4;padding:1px 5px;border-radius:3px;font-size:12px;font-family:monospace">$1</code>');
        if (line.startsWith('•')) {
          return <div key={`${partIdx}-${i}`} className="pl-3 py-0.5 text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: line }} />;
        }
        if (line.trim() === '') return <div key={`${partIdx}-${i}`} className="h-2" />;
        return <p key={`${partIdx}-${i}`} className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: line }} />;
      });
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
