# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodWeight = 10.0
        ghostWeight = 10.0

        score = successorGameState.getScore()

        distToGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distToGhost > 0:
            score = score - ghostWeight / distToGhost

        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
            score = score + foodWeight / min(distancesToFood)
        
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numOfGhosts = gameState.getNumAgents() - 1
        return self.findMax(gameState, 1, numOfGhosts)

    def findMax(self, gameState, depth, numOfGhosts):
        """
          maximize agent in minimax
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        maxScore = float("-inf")
        
        scoreAction = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          newScore = self.findMin(successor, depth, 1, numOfGhosts)
          
          if newScore > maxScore:
            maxScore = newScore
            scoreAction = action

        if depth > 1:
          return maxScore
        else: return scoreAction

    def findMin(self, gameState, depth, agentIndex, numOfGhosts):
        """
          minimize agent in minimax 
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        minScore = float("inf")
        action = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in action]
        if agentIndex == numOfGhosts:
          if depth < self.depth:
            for successor in successorGameStates:
              minScore = min(minScore, self.findMax(successor, depth + 1, numOfGhosts))
          else:
            for successor in successorGameStates:
              minScore = min(minScore, self.evaluationFunction(successor))
        else:
          for successor in successorGameStates:
            minScore = min(minScore, self.findMin(successor, depth, agentIndex + 1, numOfGhosts))
        return minScore

"helper class ScoreAction"
class ScoreAction:
    def __init__(self, score, action):
        self.score = score
        self.action = action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBetaSearch(gameState, agentIndex = 0, depth = self.depth).action

    def lastAgent(self, gameState, agentIndex):
        return agentIndex == gameState.getNumAgents() - 1

    def firstAgent(self, agentIndex):
        return agentIndex == 0
    
    """a represents alpha, b represents beta."""
    def findmax(self, gameState, agentIndex, depth, a, b):
            action = gameState.getLegalActions(agentIndex)
            scoreAction = Directions.STOP
            if self.lastAgent(gameState, agentIndex):
                nextAgent = 0
                nextDepth = depth - 1
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth
            score, action = float('-inf'), scoreAction
            for legalAction in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, legalAction)
                newScoreAction = self.AlphaBetaSearch(successorGameState, nextAgent, nextDepth, a, b)
                if newScoreAction.score > score:
                    score = newScoreAction.score 
                    action = legalAction
                if newScoreAction.score > b:
                    return newScoreAction
                a = max(a, newScoreAction.score)
            return ScoreAction(score, action)

    def findmin(self, gameState, agentIndex, depth, a, b):
            action = gameState.getLegalActions(agentIndex)
            if self.lastAgent(gameState, agentIndex):
                nextAgent = 0
                nextDepth = depth - 1
            else:
                nextAgent = agentIndex + 1 
                nextDepth = depth
            score, action = float('inf'), Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newScoreAction = self.AlphaBetaSearch(successorGameState, nextAgent, nextDepth, a, b)
                if newScoreAction.score < score:
                    score, action = newScoreAction.score, action
                if newScoreAction.score < a:
                    return newScoreAction
                b = min(b, newScoreAction.score)
            return ScoreAction(score, action)

    

    def AlphaBetaSearch(self, gameState, agentIndex, depth, a = float('-inf'), b = float('inf')):
        if depth == 0  or gameState.isWin() or gameState.isLose():
            scoreAction = Directions.STOP
            return ScoreAction(self.evaluationFunction(gameState), scoreAction)
        else:
            if self.firstAgent(agentIndex):
                return self.findmax(gameState, agentIndex, depth, a, b)
            else:
                return self.findmin(gameState, agentIndex, depth, a, b)

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numOfGhosts = gameState.getNumAgents() - 1
        return self.findmax(gameState, 1, numOfGhosts)

    def findmax(self, gameState, depth, numOfGhosts):
        """
          maximizing agent in expectimax
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        maxScore = float("-inf")
        scoreAction = Directions.STOP
        for action in gameState.getLegalActions(0):
          successor = gameState.generateSuccessor(0, action)
          score = self.getExpectedValue(successor, depth, 1, numOfGhosts)
          if score > maxScore:
            maxScore = score
            scoreAction = action

        
        if depth <= 1:
          return scoreAction
        return maxScore
        

    def getExpectedValue(self, gameState, depth, agentIndex, numOfGhosts):
        """
          minimizing agent in minimax 
        """
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        action = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in action]
        
        expectedValue = 0
        potentialSuccessor = 10.0 / len(action)
        if agentIndex == numOfGhosts:
          if depth >= self.depth:
            for successor in successorGameStates:
              expectedValue += potentialSuccessor * self.evaluationFunction(successor)
          else:
            for successor in successorGameStates:
              expectedValue += potentialSuccessor * self.findmax(successor, depth + 1, numOfGhosts)
        else:
          for successor in successorGameStates:
            expectedValue += potentialSuccessor * self.getExpectedValue(successor, depth, agentIndex + 1, numOfGhosts)
        return expectedValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    temp = 9999999
    minusTemp = -9999999
    if currentGameState.isWin():
        return temp
    if currentGameState.isLose():
        return minusTemp

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghostPos = currentGameState.getGhostPositions()
    nearestGhost = ghostPos[0] 
    disToNearestFood = min(map(lambda x: manhattanDistance(x, newPos), newFood.asList()))
    for newGhostPos in ghostPos:
        if manhattanDistance(newGhostPos, newPos) < manhattanDistance(nearestGhost, newPos):
            nearestGhost = newGhostPos
    if manhattanDistance(nearestGhost, newPos) == 0:
        return minusTemp
    if not newScaredTimes[ghostPos.index(nearestGhost)] == 0.0:
      ghostFactor = 0
    else: ghostFactor = 0.01 #ghost factor

    

    #NearestFoodScore
    if newFood[newPos[0]][newPos[1]]:
      nearestFoodScore = 10
    else:
      nearestFoodScore = 10/float(disToNearestFood)
    
    ghostScore = manhattanDistance(nearestGhost, newPos)
    finalGhostScore = ghostFactor * ghostScore/(disToNearestFood + 1)
    currentScore = currentGameState.getScore()

    finalScore = currentScore + finalGhostScore + nearestFoodScore - currentGameState.getNumFood()
    return finalScore
    

# Abbreviation
better = betterEvaluationFunction


