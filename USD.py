import numpy as np
import gym
from gym import wrappers
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import cv2
import matplotlib
import matplotlib.pyplot as plt
from cv2 import cv2
import pickle
import math



###### klasy metody

class GeneralSettings:
    def __init__(self):
        self.romName = "Seaquest-v0"
        self.load = None
        self.imageSizeX = 210
        self.imageSizeY = 160
        self.imageScale = 0.25
        self.possibleActions = [1,2,3,4,5]
        self.episodes = 2000000000
        self.denseUnits1 = 32
        self.denseUnits2 = 32
        self.seed = 0
        self.gamma = 0.99 #discount factor
        self.eps = np.finfo(np.float32).eps.item()
        self.rewardPenalty = 0.01
        self.learningRate = 0.00001
        self.diveMoves = 50
        
    def resizedShape(self):
        return (int(self.imageSizeX*self.imageScale), int(self.imageSizeY*self.imageScale))

    def resizedPixelCount(self):
        return int(self.imageSizeX*self.imageScale)* int(self.imageSizeY*self.imageScale)


class QLearningSettings:
    def __init__(self):
        self.general = GeneralSettings()
        self.epsilon = 1.0 
        self.epsilonMin = 0.3
        self.epsilonMax = 1.0 
        self.batchSize = 32
        self.epsilonRandomFrames = 20000
        self.epsilonGreedyFrames = 2000000
        self.maxMemoryLength = 10000
        self.updateAfterActions = 5
        self.updateTargetNetwork = 10000
        self.submergeMoves = 50

    def epsilonInterval(self):
        self.QLearning = QLearningSettings()
        return self.QLearning.epsilonMax - self.QLearning.epsilonMin


class ACLambdaSettings:
    def __init__(self, _lambda = 0.99):
        self.general = GeneralSettings()
        self._lambda = _lambda
        self.lgConst = tf.constant(_lambda * self.general.gamma)


def to_compressedGrayscale(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def setupNetwork(s):
    inputs = layers.Input(shape=s.general.resizedPixelCount())
    h1 = layers.Dense(s.general.denseUnits1, activation="relu")(inputs)
    h2 = layers.LayerNormalization()(layers.Dense(s.general.denseUnits2, activation="relu")(h1))

    actor = layers.Dense(len(s.general.possibleActions), activation="softmax")(h2)
    critic = layers.Dense(1)(h2)
    return keras.Model(inputs=inputs, outputs=[actor, critic])


class ActorCriticModel(keras.Model):
    def __init__(self, s=ACLambdaSettings()):
        super(ActorCriticModel, self).__init__()
        self.s = s
        self.model = setupNetwork(s)
        self.optimiser = self.compile(s.general.learningRate)
 
    
    def call(self, state):
        state = to_compressedGrayscale(state.numpy(), self.s.general.imageScale)
        state = np.reshape(state, self.s.general.resizedPixelCount())
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.model(state)
    
    def compile(self, lr):
        optimiser = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer = optimiser)    
        return optimiser


class ActorCriticAgent:
    def __init__(self, s=ACLambdaSettings()):
        self.s = s
        self.action = None
        self.actorCritic = ActorCriticModel(s)
        self.zy = None     

    def chooseAction(self, state):
        probs, _ = self.actorCritic(state)
        actionProbs = tfp.distributions.Categorical(probs=probs)
        self.action = actionProbs.sample()    
        return self.action.numpy()[0]

    def learn(self, state, reward, stateNext, done):    
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            probsA, criticValue= self.actorCritic(state)
            _, criticValueNext = self.actorCritic(stateNext)
            criticValue = tf.squeeze(criticValue)
            criticValueNext = tf.squeeze(criticValueNext)

            timeDiff = reward + self.s.general.gamma*criticValueNext * (0 if done else 1) - criticValue
            logProbs = [tf.math.log(x+ self.s.general.eps) for x in probsA[0]]
            
            target = [logProbs, criticValue]
            if self.zy is None:
                self.zy = tape.gradient(target, self.actorCritic.trainable_variables)
            else:
                tempMult = [(grad if (grad is None) else tf.scalar_mul(self.s.lgConst, grad)) for grad in self.zy]
                tempGrads = tape.gradient(target, self.actorCritic.trainable_variables)
                self.zy = [tempGrads[idx] + tempMult[idx] for idx in range(len(tempMult))]

            self.actorCritic.optimiser.apply_gradients(zip([tf.scalar_mul(-timeDiff, x) for x in self.zy], self.actorCritic.trainable_variables))

    def saveModel(self):
        self.actorCritic.model.save('./ACmodel')
    
    def loadModel(self):
        self.actorCritic.model = keras.models.load_model('./ACmodel')

    
        
###### aktor-krytyk

def runACLambda(s = ACLambdaSettings(), demo = False):

    episodesRewards = []  

    env = gym.make(s.general.romName)
    env.seed(s.general.seed)

    agent = ActorCriticAgent()
    bestScore = 0
    historyScore = []
    scoreSomeFramesBefore = 0

    if demo:
        agent.loadModel()

    for currEpisode in range(s.general.episodes):       
        state = env.reset()

        episodeReward = 0
        done = False
        frame = 0
        while not done:
            frame +=1
            if demo:
                env.render()  

            action = agent.chooseAction(state) + 1 

            stateNext, reward, done, _ = env.step(action)
            episodeReward +=reward

            if not demo:
                if frame > 200 and episodeReward == scoreSomeFramesBefore and frame%50 == 0: 
                    if frame%500 == 0:
                        agent.learn(state,-100,stateNext, True)
                        break
                    else:
                        agent.learn(state,-10,stateNext, done)
                else:     
                    scoreSomeFramesBefore = episodeReward
                
                agent.learn(state, reward, stateNext, done)
            
            state = stateNext

            
        historyScore.append(episodeReward)
        avgScore = np.mean(historyScore[-100:])

        if avgScore > bestScore:
            bestScore = avgScore
            if not demo:
                agent.saveModel()

        print("episode {}: rewarded with {}".format(currEpisode, episodeReward))

        if currEpisode%10 == 0 and not demo:
            fig, ax = plt.subplots()
            ax.plot(historyScore)

            ax.set(xlabel='epizod', ylabel='wynik',
                title='Wyniki dla kolejnych epizodów')
            ax.grid()
            plt.savefig('ACPlot')

    env.close()

###### /aktor-krytyk

###### Q-learning

def createQModel(s, optimiser):

    inputs = layers.Input(shape=s.general.resizedPixelCount())

    b = layers.Dense(s.general.denseUnits1, activation="relu")(inputs)
    b = layers.LayerNormalization()(layers.Dense(s.general.denseUnits2, activation="relu")(b))
    actions = layers.Dense(len(s.general.possibleActions), activation="softmax")(b)
    
    model = keras.Model(inputs=inputs, outputs=actions)
    model.compile(optimizer = optimiser) 
    return model
   

def runQLearn(s = QLearningSettings(), demo = False):
    
    episodesRewards =[]

    env = gym.make(s.general.romName)
    env.seed(s.general.seed)

    if demo:
        model = keras.models.load_model('./Qmodel')
        modelTarget = keras.models.load_model('./QmodelT')
    else:
        optimiser = keras.optimizers.Adam(learning_rate=s.general.learningRate, clipnorm=1.0) 
        model = createQModel(s, optimiser = optimiser)
        modelTarget = createQModel(s, optimiser = optimiser)
  
        
    
    env = gym.make(s.general.romName)
    env.seed(s.general.seed)

    

    actionHistory = []
    stateHistory = []
    stateNextHistory = []
    rewardsHistory = []
    doneHistory = []
    episodeRewardHistory = []
    runningReward = 0
    episodeCount = 0
    frameCount = 0
    bestReward = 0
    
    lossFunction = keras.losses.Huber()
    epsilon = s.epsilon


    informOfRandomPeriodEnd = True
    for currEpisode in range(s.general.episodes):       
        state = np.array(env.reset())
        state = to_compressedGrayscale(state, s.general.imageScale)
        state = np.reshape(state, s.general.resizedPixelCount())
        sumbmergeMove = 0

        episodeReward = 0

        done = False
        while not done:
            if demo or True:
                env.render(); 
            frameCount = frameCount + 1

            if not demo and (epsilon > np.random.rand(1)[0] or frameCount < s.epsilonRandomFrames) :
                if informOfRandomPeriodEnd is True and (s.epsilonRandomFrames - frameCount)% 100 is 1:
                    print("Random period for " + str(s.epsilonRandomFrames - frameCount) + " frames")
                if sumbmergeMove < s.submergeMoves:                  
                    action = 5
                    sumbmergeMove += 1
                else:
                    action = np.random.choice(s.general.possibleActions)
            else:
                if not demo and informOfRandomPeriodEnd:
                    print("RANDOM PERIOD END")
                    informOfRandomPeriodEnd = False

                sTensor = tf.convert_to_tensor(state)
                sTensor = tf.expand_dims(sTensor,0)
                actionProbs = model(sTensor, training=False)
                action = tf.argmax(actionProbs[0]).numpy() + 1

            epsilon = epsilon - s.epsilonInterval() / s.epsilonGreedyFrames
            epsilon = max(epsilon, s.epsilonMin)

            stateNext, reward, done, _ = env.step(action)
            stateNext = to_compressedGrayscale(stateNext, s.general.imageScale)
            stateNext = np.reshape(stateNext, s.general.resizedPixelCount())

            episodeReward += reward

            actionHistory.append(action)
            stateHistory.append(state)
            stateNextHistory.append(stateNext)
            doneHistory.append(done)
            rewardsHistory.append(reward)
            state = stateNext

            if not demo and frameCount % s.updateAfterActions == 0 and len(doneHistory) > s.batchSize:

                indices = np.random.choice(range(len(doneHistory)), size=s.batchSize)

                stateSample = np.array([stateHistory[i] for i in indices])
                stateNextSample = np.array([stateNextHistory[i] for i in indices])
                rewardsSample = [rewardsHistory[i] for i in indices]
                actionSample = [actionHistory[i] for i in indices]
                doneSample = tf.convert_to_tensor( [float(doneHistory[i]) for i in indices])


                futureRewards = modelTarget.predict( tf.convert_to_tensor(stateNextSample))
                updatedQValues = rewardsSample + s.general.gamma * tf.reduce_max(futureRewards, axis=1)

                updatedQValues = updatedQValues - updatedQValues * doneSample - doneSample

                masks = tf.one_hot(actionSample, len(s.general.possibleActions))

                with tf.GradientTape() as tape:
                    qValues = model(stateSample)

                    qAction = tf.reduce_sum(tf.multiply(qValues, masks), axis=1)
                    loss = lossFunction(updatedQValues, qAction)

                grads = tape.gradient(loss, model.trainable_variables)
                optimiser.apply_gradients(zip(grads, model.trainable_variables))

            if not demo and frameCount % s.updateTargetNetwork == 0:
                modelTarget.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(runningReward, episodeCount, frameCount))

            if len(rewardsHistory) > s.maxMemoryLength:
                del rewardsHistory[:1]
                del stateHistory[:1]
                del stateNextHistory[:1]
                del actionHistory[:1]
                del doneHistory[:1]

        episodeRewardHistory.append(episodeReward)
        if len(episodeRewardHistory) > 100:
            del episodeRewardHistory[:1]
        runningReward = np.mean(episodeRewardHistory)
        
        if not demo and bestReward < runningReward:
            bestReward = runningReward
            model.save('./Qmodel')
            modelTarget.save('./QmodelT')

        episodeCount += 1

        print("episode {}: rewarded with {}".format(currEpisode, episodeReward))
        episodesRewards.append(episodeReward)

        if not demo and currEpisode%20 == 0:
            fig, ax = plt.subplots()
            ax.plot(episodesRewards)

            ax.set(xlabel='epizod', ylabel='wynik',
                title='Wyniki dla kolejnych epizodów')
            ax.grid()
            plt.savefig('QmodelPlot')
        
            
    env.close()
    



runAC = False
demoOfTrainedModels = False

if runAC:
    runACLambda(demo = demoOfTrainedModels)
else:
    runQLearn(demo = demoOfTrainedModels)