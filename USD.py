import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib
import matplotlib.pyplot as plt

###### wspólne klasy metody

class GeneralSettings:
    def __init__(self):
        self.romName = "Seaquest-v0"
        self.load = None
        self.imageSizeX = 210
        self.imageSizeY = 160
        self.imageScale = 0.25
        self.possibleActions = 5
        self.episodes = 5
        self.maxStepsPerEpisode = 10000
        self.denseUnits = 32
        self.rewardPenalty = 0
        self.seed = 0
        self.gamma = 0.99 #discount factor
        self._lambda = 0.9
        self.eps = np.finfo(np.float32).eps.item()

    def resizedPixelCount(self):
        return int(self.imageSizeX*self.imageScale)*int(self.imageSizeY*self.imageScale)

class AKLambdaSettings:
    def __init__(self):
        self.general = GeneralSettings()

class RenderResult:
    def __init__(self):
        self.actionProb = 0
        self.criticVal = 0
        self.reward = 0
        self.actorLoss = 0
        self.criticLoss = 0

def to_compressedGrayscale(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

###### /wspólne metody


###### aktor-krytyk

def trainAKLambda(romName, s = AKLambdaSettings()):

    #tf.compat.v1.enable_eager_execution()

    env = gym.make(s.general.romName)
    env.seed(s.general.seed)
    stateShape = layers.Input(shape=(s.general.resizedPixelCount(),))
    
    
 
    if(s.general.load is not None):
        model = keras.models.load_model(s.general.load)
    else:
        normalized = layers.LayerNormalization()(layers.Dense(s.general.denseUnits, activation="relu")(stateShape))
        actions = layers.Dense(env.action_space.n, activation="softmax")(normalized)
        critic = layers.Dense(1)(normalized)
        model = keras.Model(inputs=stateShape, outputs=[actions, critic])

    optimizer = keras.optimizers.Adam()


    episodesRewards = []
    for currEpisode in range(s.general.episodes):       
        episodeReward = 0
        observation = env.reset()


        done = False
        history = []
        with tf.GradientTape() as gTape:
            while not done:
                
                #env.render()           
                observation = to_compressedGrayscale(observation, s.general.imageScale)
                observation = np.reshape(observation, s.general.resizedPixelCount())
                state = tf.convert_to_tensor(observation)
                state = tf.expand_dims(state, 0)

                actionsProb, criticVal = model(state)
                chosenAction = np.random.choice(env.action_space.n, p=np.squeeze(actionsProb))          
                observation, reward, done, _ = env.step(chosenAction)
                
                rResult = RenderResult()
                rResult.criticVal = criticVal[0, 0]
                rResult.action = tf.math.log(actionsProb[0, chosenAction])
                rResult.reward = reward - s.general.rewardPenalty
                history.append(rResult)

                episodeReward += (reward - s.general.rewardPenalty)
                

            for currRender in history:
                currRender.timeDiff = currRender.reward - currRender.criticVal + (0 if currRender is history[-1] else history[1 + history.index(currRender)].criticVal * s.general.gamma)


            discRewards = []        
            for currRender in history[::-1]:
                discRewards.insert(0, currRender.reward + s.general.gamma * (0 if currRender is history[-1] else history[1 + history.index(currRender)].reward))


            discRewards = np.array(discRewards)
            discRewards = (discRewards - np.mean(discRewards)) / (np.std(discRewards) + s.general.eps)
            discRewards = discRewards.tolist()

            for discReward in discRewards:
                history[discRewards.index(discReward)].reward = discReward
            
            huberLosses = keras.losses.Huber()
            
            prevRender = None
            for currRender in history:
                currRender.actorLoss = s.general._lambda * s.general.gamma * 0 if prevRender is None else prevRender.actorLoss - currRender.action * (currRender.reward - currRender.criticVal)
                currRender.criticLoss =  s.general._lambda * s.general.gamma * 0 if prevRender is None else prevRender.criticLoss + huberLosses(tf.expand_dims(currRender.criticVal, 0), tf.expand_dims(currRender.reward, 0))
                prevRender = currRender
            
            
            
            gradient = gTape.gradient(sum([rHist.actorLoss + rHist.criticLoss for rHist in history]), model.trainable_variables)
                            
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            
                
            print("episode {}: rewarded with {}".format(currEpisode, episodeReward))
            episodesRewards.append(episodeReward)

            if currEpisode%10 == 0:
                model.save('./model_snapshots/model_ep{}'.format(currEpisode))
    
    
    fig, ax = plt.subplots()
    ax.plot(range(1,s.general.episodes+1), episodesRewards)

    ax.set(xlabel='epizod', ylabel='wynik',
        title='Wyniki dla kolejnych epizodów')
    ax.grid()
    plt.show()
    

###### /aktor-krytyk

###### Q-learning

def trainQLearn(romName):
    x = 1

###### /Q-learning

###### Porównanie

romName = "Seaquest-v0"

AKLambdaModel = trainAKLambda(romName)
#QLearnModel = trainQLearn(romName)

###### /Porównanie