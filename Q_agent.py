# coding=utf-8
from collections import defaultdict
import random
from environment import *
import matplotlib.pyplot as plt

Q_PERIOD = 3


class Q_Agent():
    def __init__(self, actions, epsilon=0, discount=1, alpha=0.8):
        self.actions = actions
        self.game = Game()
        self.Q = defaultdict(float)
        self.initial_epsilon = epsilon
        self.discount = discount
        self.alpha = alpha

    def select_action(self, state):
        # Exploration and Exploitation
        if random.random() < self.epsilon:
            return np.random.choice(self.game.action_space.n)

        qValues = [self.Q.get((state, action), 0) for action in self.actions]
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return np.random.choice(self.game.action_space.n)

    def update_Q(self, state, action, reward, next_state):
        """Update the Q value based on Q-Learning"""

        next_Q = [self.Q.get((next_state, a), 0) for a in self.actions]
        best_value = max(next_Q)
        self.Q[(state, action)] = self.Q.get((state, action), 0) + self.alpha * \
                                  (reward + self.discount * best_value - self.Q.get((state, action), 0))

    def train(self, n_iters, n_iters_eval):
        """ Train the agent"""
        self.game.seed(random.randint(0, 100))
        done = False
        max_scores = [0]
        avg_scores = [0]

        for i in range(n_iters):
            self.epsilon = self.initial_epsilon
            sars_list = []

            for j in range(Q_PERIOD):
                score = 0
                total_reward = 0
                ob = self.game.reset()
                state = self.game.getGameState()
                while True:
                    action = self.select_action(state)
                    next_state, reward, done, _ = self.game.step(action)
                    sars_list.append((state, action, reward, next_state))
                    state = next_state

                    total_reward += reward
                    if reward >= 1:
                        score += 1
                    if done:
                        break

            for (state, action, reward, next_state) in sars_list:
                self.update_Q(state, action, reward, next_state)

            if i % 250 == 0:
                print("Iter: ", i)

            # Evaluate the model after every 500 iterations
            if (i + 1) % 500 == 0:
                max_score, avg_score = self.evaluate(n_iter=n_iters_eval)
                max_scores.append(max_score)
                avg_scores.append(avg_score)

        draw_graph(max_scores, avg_scores, n_iters)
        self.game.close()

    def evaluate(self, n_iter):
        """evaluates the agent"""
        self.epsilon = 0
        self.game.seed(0)

        max_score = 0
        results = defaultdict(int)

        for i in range(n_iter):
            score = 0
            total_reward = 0
            ob = self.game.reset()
            state = self.game.getGameState()

            while True:
                action = self.select_action(state)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break

            results[score] += 1
            if score > max_score:
                max_score = score

        self.game.close()
        avg_score = sum([k*v for (k, v) in results.items()]) / n_iter

        print("Max Score on Evaluation: ", max_score)
        print("Average Score on Evaluation: ", avg_score)

        return max_score, avg_score


def draw_graph(max_scores, avg_scores, iter_num):
    x = np.arange(0, iter_num + 1, step=500)
    x_ticks = np.arange(0, iter_num + 1, step=2000)

    plt.plot(x, max_scores, color="red", label="Max Scores")
    plt.plot(x, avg_scores, color="darkviolet", linestyle="dashed", label="Average Scores")
    plt.legend(loc="upper left")
    plt.xticks(x_ticks)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Scores')
    plt.title('Forward Q-Learning & No ϵ-Greedy')
    plt.savefig('figures/graph.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    agent = Q_Agent(actions=[0, 1])
    agent.train(20000, 100)
