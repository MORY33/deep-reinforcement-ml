import gym
import numpy as np
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory


class CartPoleAgent:
    def __init__(self, env):
        self.env = env
        self.states = env.observation_space.shape[0]
        self.actions = env.action_space.n
        self.model = None
        self.dqn = None
        self.policy = None

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self.states)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.actions, activation="linear"))
        self.model = model

    def choose_policy(self, policy_type):
        if policy_type == "boltzmann":
            return BoltzmannQPolicy()
        elif policy_type == "epsilon_greedy":
            return EpsGreedyQPolicy()
        elif policy_type == "max_boltzmann":
            return MaxBoltzmannQPolicy()
        else:
            raise ValueError("Invalid policy type provided.")

    def build_agent(self, policy):
        memory = SequentialMemory(limit=50000, window_length=1)
        self.dqn = DQNAgent(
            model=self.model,
            memory=memory,
            policy=policy,
            nb_actions=self.actions,
            nb_steps_warmup=10,
            target_model_update=0.01
        )

    def train(self, policy_type="boltzmann"):
        self.policy = self.choose_policy(policy_type)
        self.build_model()
        self.build_agent(self.policy)
        self.dqn.compile(Adam(learning_rate=0.001), metrics=["mae"])
        self.dqn.fit(self.env, nb_steps=50000, visualize=False, verbose=1)
        self.save_model_with_policy(policy_type)

    def test(self):
        results = self.dqn.test(self.env, nb_episodes=10, visualize=True)
        print(np.mean(results.history['episode_reward']))

    def save_model_with_policy(self, policy_name):
        save_model(self.model, f'cartpole_model_{policy_name}.h5')

    def load_model_with_policy(self, filename):
        policy_name = filename.split('_')[-1].split('.')[0]
        self.policy = self.choose_policy(policy_name)
        self.model = load_model(filename)
        self.build_agent(self.policy)
        self.dqn.compile(Adam(learning_rate=0.001), metrics=["mae"])

    def run(self):
        choice = input("Do you want to [train] a new model or [load] an existing model? ").strip().lower()

        if choice == 'train':
            policy_type = input(
                "Enter the type of policy (boltzmann, epsilon_greedy, max_boltzmann) or press Enter for default (boltzmann): "
            ).strip().lower()
            if policy_type not in ["boltzmann", "epsilon_greedy", "max_boltzmann"]:
                print("Unknown policy type. Using default (Boltzmann).")
                policy_type = "boltzmann"
            self.train(policy_type)
        elif choice == 'load':
            print("Loading existing model...")
            filename = input("Enter the filename of the model to load (e.g., 'cartpole_model_boltzmann.h5'): ").strip()
            self.load_model_with_policy(filename)

        self.test()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = CartPoleAgent(env)
    agent.run()
