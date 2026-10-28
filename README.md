# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```
# Policy Evaluation Function
def policy_evaluation(pi, P, gamma, theta=1e-8):
    V = np.zeros(len(P))
    iteration = 0

    while True:
        prev_V = V.copy()
        iteration += 1

        for s in range(len(P)):
            V[s] = 0.0

            for a, action_prob in enumerate(pi[s]):
                for prob, next_state, reward, done in P[s][a]:
                    V[s] += action_prob * prob * (
                        reward
                        + gamma * prev_V[next_state] * (not done)
                    )

        # Check convergence
        if np.max(np.abs(V - prev_V)) < theta:
            break

    return V, iteration



```

---

## Output

```
# Student details
print("Name: S R NIVEDHITHA")
print("Register Number:212225240102")

# Run for both gamma values
for gamma in [0.9, 0.99]:

    V, iterations = policy_evaluation(
        policy, env.P, gamma, theta
    )

    print("\n" + "=" * 40)
    print("GAMMA =", gamma)
    print("=" * 40)

    print("Number of iterations:", iterations)

    print("\nState-Value Function:")
    print(V)

    print("\nState-Value Function as 4x4 Grid:")
    print(np.round(V.reshape(4, 4), 4))


# Close environment
env.close()




```
<img width="288" height="356" alt="image" src="https://github.com/user-attachments/assets/0c81d867-c945-439e-93d6-81448a3c6955" />


## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

1. The random policy selects all four actions (Left, Down, Right, and Up) with equal probability of 0.25 in every state.

2. When \(\gamma = 0.9\), future rewards are given less importance, so the state values are generally lower than with \(\gamma = 0.99\).

3. When \(\gamma = 0.99\), future rewards are given more importance, resulting in higher state values for states that have a greater chance of reaching the goal.

4. The policy evaluation algorithm updates the state values repeatedly until the difference between consecutive iterations becomes smaller than the convergence threshold \(\theta = 10^{-8}\). The goal and hole states are terminal states, so no future rewards are added after reaching them.








---


