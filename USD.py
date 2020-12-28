import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

###### wspólne metody


def map_to_grayscale(state):

    # return cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
  return image
    Y = range(len(state))
    X = range(len(state[0]))
    C = range(len(state[0][0]))
    mapped = np.array([[0 for col in X] for row in Y])
    for y in Y:
        for x in X:
            val = 0
            for c in C:
                val += state[y][x][c]
            mapped[y][x] = val / 3
    return mapped

###### /wspólne metody


romName = "Seaquest-v0"

###### aktor-krytyk

algorithm_name = "actor_critic"
version = "1"



load_snapshot = False


seed = 0
gamma = 0.99 #discount factor
max_steps_per_episode = 10000
env = gym.make(romName)
env.seed(seed)
eps = np.finfo(np.float32).eps.item()

screenY = 210
screenX = 160
num_inputs = screenY * screenX
hidden_layer_1_units = 128
num_actions = 5

inputs = layers.Input(shape=(num_inputs,))

hidden_layer_1 = layers.Dense(hidden_layer_1_units, activation="relu")(inputs)
normalized = layers.LayerNormalization()(hidden_layer_1)
actions = layers.Dense(num_actions, activation="softmax")(normalized)
critic = layers.Dense(1)(normalized)
model = keras.Model(inputs=inputs, outputs=[actions, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)


critic_value_history = []
action_probs_history = []
rewards_history = []
episode_count = 0
num_episodes = 100

while episode_count in range(num_episodes):
    state = env.reset()
    state = map_to_grayscale(state)
    episode_reward = 0
    with tf.GradientTape() as tape:
        done = False
        while not done:
            env.render()
            
            state = np.reshape(state, num_inputs)
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
        
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])
        
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
        
            state, reward, done, _ = env.step(action)
            state = map_to_grayscale(state)
            rewards_history.append(reward)
            episode_reward += reward
        
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        huber_loss = keras.losses.Huber()
        
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        
        grads = tape.gradient(sum(actor_losses) + sum(critic_losses), model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        
    print("episode {}: rewarded with {}".format(episode_count, episode_reward))
    episode_count += 1
    if(not episode_count % 5):
        model.save('./model_snapshots/model_{}_{}_ep{}'.format(algorithm_name, version, episode_count))

###### /aktor-krytyk

###### Q-learning

###### /Q-learning

###### Porównanie

###### /Porównanie