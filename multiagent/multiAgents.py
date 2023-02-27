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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0

        gameStateScoreWeight = 3
        diffGameStateScore = successorGameState.getScore() - currentGameState.getScore()

        closestFoodWeight = 2
        if (len(newFood.asList()) != 0):
            closestFood = min([manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])
            closestFoodOld = min([manhattanDistance(currentGameState.getPacmanPosition(), foodPos) for foodPos in currentGameState.getFood().asList()])
            diffClosestFood = closestFoodOld - closestFood
        else:
            diffClosestFood = 1

        closestGhostWeight = 1
        closestGhost = min([manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])
        closestGhostOld = min([manhattanDistance(currentGameState.getPacmanPosition(), ghostPos) for ghostPos in currentGameState.getGhostPositions()])
        diffClosestGhost = closestGhost - closestGhostOld

        score = gameStateScoreWeight * diffGameStateScore + closestFoodWeight * diffClosestFood + closestGhostWeight * diffClosestGhost
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        move = self.maxHelper(gameState, self.depth)
        return move[1]

    def maxHelper(self, gameState: GameState, depth: int):
        if (depth == 0):
            return (self.evaluationFunction(gameState), 'Stop')
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = float('-inf')
        bestAction = ''
        move = (v, bestAction)
        for action in gameState.getLegalActions(0):
            successor = self.minHelper(gameState.generateSuccessor(0, action), depth, 1)
            if (successor[0] > move[0]):
                move = (successor[0], action)
        return move
    
    def minHelper(self, gameState: GameState, depth: int, agent: int):
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = float('inf')
        bestAction = ''
        move = (v, bestAction)
        if (agent == gameState.getNumAgents() - 1 and depth != 0):
            for action in gameState.getLegalActions(agent):     
                successor = self.maxHelper(gameState.generateSuccessor(agent, action), depth - 1)
                if (successor[0] < move[0]):
                    move = (successor[0], action)
        if (agent == gameState.getNumAgents() - 1 and depth == 0):
            for action in gameState.getLegalActions(agent):     
                return (self.evaluationFunction(gameState.generateSuccessor(agent, action)), 'Stop')
        if (agent != gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agent):
                successor = self.minHelper(gameState.generateSuccessor(agent, action), depth, agent + 1)
                if (successor[0] < move[0]):
                    move = (successor[0], action)
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        move = self.maxHelper(gameState, self.depth, alpha, beta)
        return move[1]
        
    def maxHelper(self, gameState: GameState, depth: int, alpha: int, beta: int):
        if (depth == 0):
            return (self.evaluationFunction(gameState), 'Stop')
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = float('-inf')
        bestAction = ''
        move = (v, bestAction)
        for action in gameState.getLegalActions(0):
            successor = self.minHelper(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
            if (successor[0] > move[0]):
                move = (successor[0], action)
            if (move[0] > beta):
                return move
            alpha = max(alpha, move[0])
        return move

    def minHelper(self, gameState: GameState, depth: int, agent: int, alpha: int, beta: int):
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = float('inf')
        bestAction = ''
        move = (v, bestAction)
        if (agent == gameState.getNumAgents() - 1 and depth != 0):
            for action in gameState.getLegalActions(agent):     
                successor = self.maxHelper(gameState.generateSuccessor(agent, action), depth - 1, alpha, beta)
                if (successor[0] < move[0]):
                    move = (successor[0], action)
                if (move[0] < alpha):
                    return move
                beta = min(beta, move[0])
        if (agent == gameState.getNumAgents() - 1 and depth == 0):
            for action in gameState.getLegalActions(agent):     
                return (self.evaluationFunction(gameState.generateSuccessor(agent, action)), 'Stop')
        if (agent != gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agent):
                successor = self.minHelper(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                if (successor[0] < move[0]):
                    move = (successor[0], action)
                if (move[0] < alpha):
                    return move
                beta = min(beta, move[0])
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        move = self.maxHelper(gameState, self.depth)
        return move[1]

    def maxHelper(self, gameState: GameState, depth: int):
        if (depth == 0):
            return (self.evaluationFunction(gameState), 'Stop')
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = float('-inf')
        bestAction = ''
        move = (v, bestAction)
        for action in gameState.getLegalActions(0):
            successor = self.expHelper(gameState.generateSuccessor(0, action), depth, 1)
            if (successor[0] > move[0]):
                move = (successor[0], action)
        return move
    
    def expHelper(self, gameState: GameState, depth: int, agent: int):
        if (gameState.isWin() or gameState.isLose()):
            return (self.evaluationFunction(gameState), 'Stop')

        v = 0
        bestAction = ''
        move = (v, bestAction)
        if (agent == gameState.getNumAgents() - 1 and depth != 0):
            for action in gameState.getLegalActions(agent):     
                successor = self.maxHelper(gameState.generateSuccessor(agent, action), depth - 1)
                p = 1/len(gameState.getLegalActions(agent))
                move = (move[0] + (p * successor[0]), action)
        if (agent == gameState.getNumAgents() - 1 and depth == 0):
            for action in gameState.getLegalActions(agent):     
                return (self.evaluationFunction(gameState.generateSuccessor(agent, action)), 'Stop')
        if (agent != gameState.getNumAgents() - 1):
            for action in gameState.getLegalActions(agent):
                successor = self.expHelper(gameState.generateSuccessor(agent, action), depth, agent + 1)
                p = 1/len(gameState.getLegalActions(agent))
                move = (move[0] + (p * successor[0]), action)
        return move

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
