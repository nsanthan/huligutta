'''
Huligutta (Goats and Tigers)
file: game.py
Description: GUI of the game using TKinter
'''

__author__ = "Narayana Santhanam"
__email__ = "nsanthan@hawaii.edu"
__status__ = "v2.0"

import time
from tkinter import *
from tkinter import messagebox
import os
import numpy as np
from numpy import random as npr
import sys
from PIL import ImageTk, Image
from random import randint, choice
import itertools
import copy
import pickle
import pdb

verbose = False

class Player():
    def __init__(self, game):
        self.playeridentity = None
        self.game = game
        self.board = self.game.gameBoard
        self.positionone = None
        self.positiontwo = None
        self.pieces = None
        self.waitingoninput = True

    def identity(self):
        return self.playeridentity
    
    def input(self, *, position=None):
        if verbose:
            print('Entering Player:input')
        if position == None:
            return self.fit()
        else:
            return self.handle_input(position)

    def handle_input(self, position):
        if verbose:
            print('Entering Player:handle_input')
        if self.positionone == None:
            if verbose:
                print('Assign positionone: ', position.address)
            self.positionone = position
            return self.fit()
        else:
            if verbose:
                print('Assign positiontwo: ', position.address)
            self.positiontwo = position
            return self.fit()

    def fit(self):
        '''
        Called either after self.positionone or both
        self.positionone/self.positiontwo assigned.  GoatPlayer must
        override this function to account for the Place phase of the
        game.

        '''
        if verbose:
            print('Entering Player:fit ', self.positionone, self.positiontwo)
        if self.positiontwo == None:
            '''Only positionone is assigned, we are waiting on another input. '''
            piece = self.positionone.content
            if piece != None:
                if verbose:
                    print('Clicked on a ',piece.identity())
                if piece.identity() == self.identity():
                    if verbose:
                        print('Lifting piece..')
                    func = piece.lift
                    self.waitingoninput = True
                    return func, piece, piece.position
                else:
                    if verbose:
                        print('Choose a', self.playeridentity, ' to move!')
                    self.positionone = None
                    self.positiontwo = None
                    self.waitingoninput = True
                    return None, None, None
            else:
                if verbose:
                    print('Clicked on an empty square')
                self.positionone = None
                self.positiontwo = None
                self.waitingoninput = True
                return None, None, None
        else:
            if verbose:
                print('Moving piece..')
            piece = self.game.liftedpiece
            func = piece.move
            self.waitingoninput = False
            return func, piece, self.positiontwo.address

    def reset(self):
        self.positionone = None
        self.positiontwo = None
        self.waitingoninput = True
        
    def lift(position):
        pass

    def place(position):
        pass

class greedyTiger(Player):
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Tiger"
        self.pieces = self.game.state.tigers
        self.waitingoninput = False

    def predict(self):
        if verbose:
            print('Entering greedyTiger:predict')
        randomorder = choice(list(itertools.permutations(self.pieces)))
        
        for tiger in randomorder:
            moves, captures, goats = tiger.allmoves()
            for move in moves:
                if verbose:
                    print('Possible moves: ', move[1].position.address,move[2])

            for capture in captures:
                if verbose:
                    print('Possible captures: ', capture[1].position.address, capture[2])
            if captures:
                onecapture = choice(captures)
                return onecapture[0], onecapture[1], onecapture[2]

        for tiger in randomorder:
            moves, captures, goats = tiger.allmoves()
            if moves:
                onemove = choice(moves)
                return onemove[0], onemove[1], onemove[2]

    def reset(self):
        self.positionone = None
        self.positiontwo = None
        self.waitingoninput = False

class greedyGoat(Player):
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Goat"
        self.pieces = self.game.state.goats
        self.waitingoninput = False

    def predict(self):
        if verbose:
            print('Entering greedyGoat:predict')

        remaininggoats = self.pieces.copy()
        randomorder = []
        while len(remaininggoats)>0:
            nextgoat = choice(remaininggoats)
            remaininggoats.remove(nextgoat)
            randomorder.append(nextgoat)

        toreturn = [None, None, None]
            
        if self.game.state.phase == 'place':
            for goat in self.pieces:
                if goat.state == 'Unused':
                    allmoves = goat.allmoves()
                    break
            if allmoves:
               onemove = choice(allmoves)
               if verbose:
                   print(onemove)
               toreturn = onemove
        else:           
            for goat in randomorder:
                moves = goat.allmoves()
                if moves:
                    onemove = choice(moves)
                    toreturn = onemove
                    if verbose:
                        print(toreturn)
                    
        # Check if any goat in danger, if so, ask for human input
        randomorder = choice(list(itertools.permutations(self.game.players[0].pieces)))
        
        for tiger in randomorder:
            moves, captures, goats = tiger.allmoves()
            for capture in captures:
                if verbose:
                    print('Possible captures: ', capture[1].position.address, capture[2])
                toreturn = [None, None, None]
        if toreturn[0] != None:
            if self.game.state.phase == 'place':
                if verbose:
                    print('Chosen move: ', toreturn[1].position, toreturn[2])
            else:
                if verbose:
                    print('Chosen move: ', toreturn[1].position.address, toreturn[2])
        else:
            print('Falling back on human input')

        return toreturn[0], toreturn[1], toreturn[2]

    def fit(self):
        ''' Called only after self.positionone assigned. '''
        if verbose:
            print('Entering Goat.fit()')
        if self.game.state.getphase() == 'place':
            if verbose:
                print('In place phase')
                print('Movecount: ', self.game.state.getmovecount())
            piece = self.pieces[self.game.state.getmovecount()] 
            func = piece.place
            dest = self.positionone.address
            self.waitingoninput = False
            return func, piece, dest
        else:
            if verbose:
                print('In move phase')
                print('Movecount: ', self.game.state.getmovecount())
            return super().fit()

    def reset(self):
        self.positionone = None
        self.positiontwo = None
        if self.game.state.getmovecount() == 0:
            self.waitingoninput = False    
        

        
class TigerPlayer(Player):
    ''' Human input interface '''
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Tiger"
        self.pieces = self.game.state.tigers

    def predict(self):
        pass

        
class GoatPlayer(Player):
    '''Human input interface'''
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Goat"
        self.pieces = self.game.state.goats
        
    def fit(self):
        ''' Called only after self.positionone assigned. '''
        if verbose:
            print('Entering Goat.fit()')
        if self.game.state.getphase() == 'place':
            if verbose:
                print('In place phase')
                print('Movecount: ', self.game.state.getmovecount())
            piece = self.pieces[self.game.state.getmovecount()] 
            func = piece.place
            dest = self.positionone.address
            self.waitingoninput = False
            return func, piece, dest
        else:
            if verbose:
                print('In move phase')
                print('Movecount: ', self.game.state.getmovecount())
            return super().fit()

    def needinput(self):
        return True
    
    def predict(game):
        pass


class autoGoat(Player):
    ''' Uses a neural network to make moves. '''
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Goat"
        self.pieces = self.game.state.goats
        self.waitingoninput = False
        self.W = None
        model_pkl_file = 'nngoat.pkl'
        with open(model_pkl_file, 'rb') as file:
            self.W = pickle.load(file)

    def reset(self):
        self.positionone = None
        self.positiontwo = None
        self.waitingoninput = False

    def needinput(self):
        return self.waitingoninput

    def matrix2board(self, matrix):
        empty = np.array([1,0,0])
        goat = np.array([0,1,0])
        tiger = np.array([0,0,1])

        tempBoard = Board(graphics=False)
        row = 0
        numgoats = 0
        numtigers = 0
        for address in tempBoard.addresslist:
            if (matrix[row,:] == empty).all():
               tempBoard.positions[address].content = None
            elif (matrix[row,:] == goat).all():
                Goat(tempBoard,numgoats).place(address)
                numgoats = numgoats + 1
            elif (matrix[row,:] == tiger).all():
                Tiger(tempBoard, numtigers).place(address)
                numtigers = numtigers + 1
            row = row+1
        return tempBoard
        
    
    def move2matrix(self, matrix, move):
        nextmatrix = copy.deepcopy(matrix)
        movefunc, piece, address = move
        empty = np.array([1,0,0])
        goat = np.array([0,1,0])
        tiger = np.array([0,0,1])
        
        if movefunc == piece.place:
            row = self.board.addresslist.index(address)
            nextmatrix[row,:] = goat
        elif movefunc == piece.move:
            row1 = self.board.addresslist.index(piece.position.address)
            row2 = self.board.addresslist.index(address)
            nextmatrix[row1,:] = empty
            nextmatrix[row2,:] = goat

        return nextmatrix

    def predict(self):
        pass
    
class neuralGoat(autoGoat):
    def softmax(self, x):
        x = np.array(x)
        vector = np.exp(x)/np.sum(np.exp(x))
        if np.isnan(vector).any():
            print(x)
            vector = np.ones(x.shape)
            vector = vector/len(vector)
        return vector

    def layerpredict(self, X, u):
            w = u.copy()
            a = np.zeros((X @ w[0]).shape)
            n = w[1].shape[0]
            w[1]= w[1].reshape(1,n)
            a = X@ w[0] + w[1]
            a = np.where(a>0, a, 0)
            return a

    def nnvaluefunc(self, matrix):
        funmatrix = copy.deepcopy(matrix)
        if self.game.state.phase == 'place':
            lastrow = np.array([1,0,0]).reshape(1,3)
        else:
            lastrow = np.array([0,1,0]).reshape(1,3)
        captured = 0
        for goat in self.pieces:
            if goat.state == 'Captured':
                captured = captured + 1
        lastrow[0,2] = captured

        X = np.vstack((funmatrix, lastrow)).ravel()
        w0 = self.W[0]
        w1 = self.W[1]
        w2 = self.W[2]
        w3 = self.W[3]

        myoutput = self.layerpredict(
            self.layerpredict(
                self.layerpredict(X, w0),w1),w2) @ w3[0]+w3[1]
        return myoutput[0,0]

    def predict(self):
        currentmatrix = copy.deepcopy(self.game.state.board2matrix(self.board))

        if self.game.state.phase == 'place':
            for goat in self.pieces:
                if goat.state == 'Unused':
                    allmoves = goat.allmoves()
        elif self.game.state.phase == 'move':
            allmoves = []
            for goat in self.pieces:
                allmoves.extend(goat.allmoves())
        if verbose:
            print('All moves obtained! Possible moves: ', len(allmoves))
        values = []
        if len(allmoves) > 50:
            allmoves = allmoves[:50]
            
        for move in allmoves:
            nextmatrix = self.move2matrix(currentmatrix, move)
            values.append(self.nnvaluefunc(nextmatrix))

        moveind = npr.choice(np.arange(len(allmoves)), p=self.softmax(values))
        movefun = allmoves[moveind]
        if self.game.state.phase == 'place':
            print('Chosen move: ', movefun[1].position, movefun[2])
        else:
            print('Chosen move: ', movefun[1].position.address, movefun[2])
        return movefun

    
class valueGoat(autoGoat):
    def softmax(self, x):
        x = np.array(x)
        vector = np.exp(x)
        if np.isinf(vector).any():
            if np.nanmax(vector) == np.inf:
                vector = np.where(vector==np.inf, 1, 0)
            if np.nanmin(vector) == -np.inf:
                breakpoint()
        if np.isnan(vector).any():
            breakpoint()
        vector = vector/np.sum(vector)
        return vector

    def valuefunc(self, matrix):
        board = self.matrix2board(matrix)
        tigermobility = 0
        numcaptures = 0
        goatsindanger = []
        unsafe = len(goatsindanger)
        center = ['b0', 'b1', 'b2', 'b3', 'c1','c2','c3', 'd1','d2','d3','e1','e2','e3']
        positionadv = 0

        for address in board.addresslist:
            position = board.positions[address]
            if position.content != None:
                if position.content.identity() == 'Tiger':
                    moves, captures, goats = position.content.allmoves()
                    tigermobility = tigermobility + len(moves) + len(captures)
                    numcaptures = numcaptures + len(captures)
                    goatsindanger.extend(goats)

        unsafe = len(goatsindanger)
        
        for address in board.addresslist:
            position = board.positions[address]
            if position.content != None:
                if position.content.identity() == 'Goat':
                    if position.address in center:
                        if position.content not in goatsindanger:
                            positionadv = positionadv + 1

        
        captured = 0
        for goat in self.pieces:
            if goat.state == 'Captured':
                captured = captured + 1

        if tigermobility == 0:
            return np.inf
        elif captured >5:
            return -np.inf
        
        return 1/(tigermobility+1e-2) - 5*(2+captured+unsafe)**2 + 3*(3+positionadv)**1.5

    
    def considermove(self):
        if self.game.state.phase == 'place':
            for goat in self.pieces:
                if goat.state == 'Unused':
                    allmoves = goat.allmoves()
        elif self.game.state.phase == 'move':
            allmoves = []
            for goat in self.pieces:
                allmoves.extend(goat.allmoves())
        elif self.game.state.phase == 'over':
            allmoves = []
        if verbose:
            print('All moves obtained! Possible moves: ', len(allmoves))
        return allmoves
    
    def predict(self):
        currentmatrix = copy.deepcopy(self.game.state.board2matrix(self.board))
        values = []
        allmoves = self.considermove()
        if len(allmoves) > 50:
            allmoves = allmoves[:50]

        for move in allmoves:
            nextmatrix = self.move2matrix(currentmatrix, move)
            values.append(self.valuefunc(nextmatrix))

        moveind = npr.choice(np.arange(len(allmoves)), p=self.softmax(values))
        movefun = allmoves[moveind]
        if self.game.state.phase == 'place':
            print('Chosen move: ', movefun[1].position, movefun[2])
        else:
            print('Chosen move: ', movefun[1].position.address, movefun[2])
        return movefun


class twostepGoat(valueGoat):
    def partialsoftmax(self, x):
        x = np.array(x)
        vector = np.exp(x)
        if np.isinf(vector).any():
            if np.nanmax(vector) == np.inf:
                return np.inf
            else:
                return np.log(np.sum(vector))
        else:
            return np.log(np.sum(vector))
        
    def assign(self, values, indmove, num):
        if len(indmove) == 1:
            values[indmove[0]] = num
        else:
            self.assign(values[indmove[0]], indmove[1:], num)
        
    def nextmoves(self, oldtempgame, indmove):
        if len(indmove) == self.lookahead:
            if oldtempgame.state.winner != None:
                ''' Game has ended with a winner. '''
                if oldtempgame.winner.identity() == 'Tiger':
                    self.assign(self.values, indmove, -np.inf)
                        
                if oldtempgame.winner.identity() == 'Goat':
                    self.assign(self.values, indmove, np.inf)
                return None
            currentmatrix = copy.deepcopy(oldtempgame.state.board2matrix(oldtempgame.gameBoard))
            goatmoves = oldtempgame.players[1].considermove()
            self.assign(self.values, indmove, [0]*len(goatmoves))
            indmove_i = 0
            for lastmove in goatmoves:
                tempmatrix =  oldtempgame.players[1].move2matrix(currentmatrix,
                                                                 lastmove)
                tempindmove = indmove.copy()
                tempindmove.append(indmove_i)
                self.assign(self.values,
                            tempindmove,
                            oldtempgame.players[1].valuefunc(tempmatrix))
                indmove_i = indmove_i + 1
            return goatmoves
        else:
            if oldtempgame.state.winner != None:
                ''' Game has ended with a winner. '''
                if oldtempgame.winner.identity() == 'Tiger':
                    self.assign(self.values, indmove, -np.inf)
                        
                if oldtempgame.winner.identity() == 'Goat':
                    self.assign(self.values, indmove, np.inf)
                return None

            ''' Come here only if no winner yet. '''
            goatmoves = oldtempgame.players[1].considermove()

            if len(indmove) > 0: 
                if len(goatmoves) > 0:
                    self.assign(self.values, indmove, [0]*len(goatmoves))
                else:
                    ''' game not over, but no goat moves? Should never reach!'''
                    breakpoint()
            else:
                self.values = [0]*len(goatmoves)

            indmove_i = 0

            
            for nextgoatmove in goatmoves:
                tempgame = Game(oldtempgame.state)
                temptigerplayer = greedyTiger(tempgame)
                tempgoatplayer = valueGoat(tempgame)
                tempgame.addplayers(temptigerplayer, tempgoatplayer)     

                movefunc, oldtemppiece, dest = nextgoatmove
                piece = tempgame.state.goats[oldtemppiece.number]
                if movefunc == oldtemppiece.place:
                    movefunc = piece.place
                if movefunc == oldtemppiece.move:
                    movefunc = piece.move                

                tempgame.makemove((movefunc,piece,dest))

                if tempgame.state.phase == 'over':
                    tempindmove = indmove.copy()
                    tempindmove.append(indmove_i)
                    self.nextmoves(tempgame, tempindmove)
                else:
                    tigermove = temptigerplayer.predict()
                    tempgame.makemove(tigermove)

                    ''' Merge the forks here, no distinction at this point.'''
                    if tempgame.state.phase == 'over':
                        tempindmove = indmove.copy()
                        tempindmove.append(indmove_i)
                        self.nextmoves(tempgame, tempindmove)
                    else:
                        tempindmove = indmove.copy()
                        tempindmove.append(indmove_i)
                        self.nextmoves(tempgame, tempindmove)
                indmove_i = indmove_i + 1
            return goatmoves
        
    def vtoprob(self, values):
        if not isinstance(values, list):
            return values
        ''' Values is a list. '''
        nummoves = len(values)
        myvalues = [0]*nummoves
        
        ListOfNumbers = True

        for i in range(nummoves):
            myvalues[i] = self.vtoprob(values[i])

        return self.partialsoftmax(myvalues)

    def depth(self, values, d):
        if not isinstance(values, list):
            if values == np.inf:
                return d
            else:
                return np.inf

        nummoves = len(values)
        depth = [np.inf]*nummoves

        for i in range(nummoves):
            depth[i] = self.depth(values[i], d+1)
                    
        return np.nanmin(depth)
            
        
    def predict(self):
        self.values = []
        indmove = []
        if self.game.state.movecount < 5:
            self.lookahead = 1
        else:
            if self.game.state.movecount < 10:
                self.lookahead = 2
            else:
                self.lookahead = 3
            
        tempgame = Game(self.game.state)
        temptigerplayer = greedyTiger(tempgame)
        tempgoatplayer = valueGoat(tempgame)
        tempgame.addplayers(temptigerplayer, tempgoatplayer)             

        ''' Randomize over tiger moves.'''
        self.randommoves = 2
        valuesarray = [0]*self.randommoves
        depth = [0]*self.randommoves
        for trial in range(self.randommoves):
            allmoves = self.nextmoves(tempgame, indmove)

            valuesarray[trial] = [0]*len(self.values)
            depth[trial] = [np.inf]*len(self.values)
            for movenumber in range(len(self.values)):
                valuesarray[trial][movenumber] = self.vtoprob(self.values[movenumber])
                depth[trial][movenumber] = self.depth(self.values[movenumber], 0)

        finalvalues = np.min(np.array(valuesarray), axis = 0)
        finaldepth = np.nanmax(np.array(depth), axis = 0)
        moveprobs = [0]*len(allmoves)
        '''
        At this point, self.values contain the "effective"
        value function for the lookahead approach.

        Task 1: write code that collects all the state, effective
        value function pairs as you play.
        
        '''
        if np.nanmin(finaldepth) != np.inf:
            winnextmove = 0
            for i in range(len(finaldepth)):
                if finaldepth[i] == 0:
                    moveprobs[i] = 1
                    winnextmove = 1
            moveprobs = moveprobs/np.sum(moveprobs)
            
            if not winnextmove:
                for i in range(len(finaldepth)):
                    if finaldepth[i] == np.inf:
                        finaldepth[i] = 0
                    else:
                        finaldepth[i] = 10**(-finaldepth[i]-1)
                moveprobs = finaldepth/np.sum(finaldepth)
        else:
            moveprobs = self.softmax(finalvalues)
        
        if verbose:
            for moves in range(len(allmoves)):
                print(allmoves[moves], moveprobs[moves])
            
        moveind = npr.choice(np.arange(len(self.values)), p = moveprobs)
        selectedmove = allmoves[moveind]

        movefunc, temppiece, dest = selectedmove
        piece = self.game.state.goats[temppiece.number]
        if movefunc == temppiece.place:
            movefunc = piece.place
        if movefunc == temppiece.move:
            movefunc = piece.move                


        if self.game.state.phase == 'place':
            print('Chosen move: ', piece.position, dest)
        else:
            print('Chosen move: ', piece.position.address, dest)
        return (movefunc, piece, dest)

        
    def oldpredict(self):
        currentmatrix = copy.deepcopy(self.game.state.board2matrix(self.board))
        allmoves = self.considermove()        
        values = []
        for firstmove in allmoves:
            tempgame = Game(self.game.state)
            temptigerplayer = greedyTiger(tempgame)
            tempgoatplayer = valueGoat(tempgame)
            tempgame.addplayers(temptigerplayer, tempgoatplayer)

            movefunc, piece, dest = firstmove
            temppiece = tempgame.state.goats[piece.number]
            if movefunc == piece.place:
                movefunc = temppiece.place
            if movefunc == piece.move:
                movefunc = temppiece.move                

            tempgame.makemove((movefunc, temppiece, dest))
            nextmatrix = tempgoatplayer.move2matrix(currentmatrix, firstmove)
            tigermove = temptigerplayer.predict()
            tempgame.makemove(tigermove)
            
            nextmatrix = tempgoatplayer.move2matrix(nextmatrix, tigermove)
            goatmoves = tempgoatplayer.considermove()
            tempvalues = []
            for secondmove in goatmoves:
                tempmatrix =  tempgoatplayer.move2matrix(nextmatrix, secondmove)
                tempvalues.append(tempgoatplayer.valuefunc(tempmatrix))
            values.append(tempvalues)
  
        allvalues = list(itertools.chain.from_iterable(values)) 
        
        if verbose:
            print(allvalues)
        probs = self.softmax(allvalues)
        moveprobs = []
        i = 0
        for lists in values:
            moveprobs.append(np.sum(probs[i:i+len(lists)]))
            i = i+len(lists)

        for moves in range(len(allmoves)):
            print(allmoves[moves], moveprobs[moves])
            
        moveind = npr.choice(np.arange(len(values)), p = moveprobs)
        movefun = allmoves[moveind]
        if self.game.state.phase == 'place':
            print('Chosen move: ', movefun[1].position, movefun[2])
        else:
            print('Chosen move: ', movefun[1].position.address, movefun[2])
        return movefun

class Game():
    '''This class performs the logic associated with the game: determining whose turn, who won (if any), passing on inputs (if any), making the moves, and keeping track of the state of the game.'''
    class State():
        def __init__(self, State = None):
            if State == None:
                '''
                Start at the beginning. (default behavior)
                '''
                self.matrix = np.zeros((23,3))
                self.movecount = 0
                self.players = ['Tiger', 'Goat']
                self.goatturn = False
                self.phase = 'place'
                self.goatEaten = 0
                self.goatCount = 0
                self.tigers =[]
                self.goats = []
                self.winner = None
            else:
                self.matrix = copy.deepcopy(State.matrix)
                self.movecount = copy.deepcopy(State.movecount)
                self.players = ['Tiger', 'Goat']
                self.goatturn = copy.deepcopy(State.goatturn)
                self.phase = copy.deepcopy(State.phase)
                self.goatEaten = copy.deepcopy(State.goatEaten)
                self.goatCount = copy.deepcopy(State.goatCount)
                self.tigers = []
                self.goats = []
                self.winner = copy.deepcopy(State.winner)
        def booleanturn(self):
            return self.goatturn

        def playerturn(self):
            if verbose:
                print('game.state.whoseturn:',self.players[self.goatturn])
            if self.players[self.goatturn]:
                return self.players[self.goatturn]
            else:
                return None
        
        def board2matrix(self,gameBoard):
            matrix = np.zeros((23,3))
            row = 0
            for key in gameBoard.addresslist:
                if gameBoard.positions[key].content == None:
                    matrix[row,:] = np.array([1,0,0])
                else:
                    if gameBoard.positions[key].content.identity() == 'Goat' :
                        matrix[row,:] = np.array([0,1,0])
                    else:
                        matrix[row,:] = np.array([0,0,1])
                row = row +1
            return matrix
                    
        def update(self,game):
            self.matrix = self.board2matrix(game.gameBoard)
            self.winner = game.gameover()
            if not self.winner:
                if self.goatturn:
                    self.movecount = self.movecount +1
                    if self.movecount < 15:
                        self.phase = 'place'
                        game.state.goatCount = self.movecount
                    else:
                        game.state.goatCount = 15
                        self.phase = 'move'
                self.goatturn = not self.goatturn
                game.gameBoard.turnDisplay()
            else:
                if game.gameBoard.graphics:
                    winner = self.winner.identity()
                    messagebox.showinfo("Game Over", winner+" wins!")
                else:
                    winner = self.winner.identity()

        def resetchange(self):
            self.changed = False

        def changeindicator(self):
            return self.changed

        def getphase(self):
            return self.phase

        def getmovecount(self):
            return self.movecount
        
    def __init__(self, State = None):
        '''
        When initialize a game from a state different from the start,
        do not use the attachboard method. Pass the current board directly.

        TODO: initialize from another game object instead of state and board
        '''
        self.playersname = ['Tiger', 'Goat']
        if State == None:
            self.state = self.State(None)
            self.gameBoard = None
            self.numTigers = 3
            self.numGoats = 15
            self.winner = None
            self.liftedpiece = None
        else:
            self.state = self.State(State)
            self.gameBoard = Board(graphics = False)
            self.gameBoard.attachgame(self)
            self.numTigers = 3
            self.numGoats = 15
            self.winner = None
            for number in range(self.numTigers):
                self.state.tigers.append(Tiger(self.gameBoard, number))
            for number in range(self.numGoats):
                self.state.goats.append(Goat(self.gameBoard, number))
            for numtiger in range(self.numTigers):
                self.state.tigers[numtiger].place(State.tigers[numtiger].position.address)
            for numgoat in range(self.numGoats):
                self.state.goats[numgoat].state = State.goats[numgoat].state
                if self.state.goats[numgoat].state == 'Playing':
                    self.state.goats[numgoat].place(State.goats[numgoat].position.address)            
            self.liftedpiece = None

            
    def attachboard(self, board):
        self.gameBoard = board
        self.gameBoard.attachgame(self)
        for number in range(self.numTigers):
            self.state.tigers.append(Tiger(self.gameBoard, number))
        for number in range(self.numGoats):
            self.state.goats.append(Goat(self.gameBoard, number))
        self.state.tigers[0].place('b0')
        self.state.tigers[1].place('c1')
        self.state.tigers[2].place('d1')
        self.state.update(self)

    def addplayers(self,tigerplayer, goatplayer):
        self.players = [tigerplayer, goatplayer]
        for player in self.players:
            player.reset()

    def checkmove(self, returnedtuple):
        ''' Checks if player returned a move in the valid format. '''
        if returnedtuple != None:
            if len(returnedtuple) ==3:
                return returnedtuple
        return None, None, None

    def makemove(self, returnedtuple):
        player = self.players[self.state.booleanturn()]
        movefunc, piece, destaddress = self.checkmove(returnedtuple)
        if piece != None:
            if verbose:
                print('Returned move: ', piece.position, destaddress)
        else:
            if verbose:
                print('Returned move: None, ', destaddress)
                
        if movefunc != None and piece.identity() == self.state.playerturn():
            if movefunc == piece.move:
                if verbose:
                    print('Lifting piece:', piece)
                piece.lift()
                piece = piece.move(destaddress)
            if movefunc == piece.place:
                piece = piece.place(destaddress)
                        
            if verbose:
                print(piece)
            if not piece:
                '''Move function did not return anything'''
                if verbose:
                    print('Move', movefunc,' unsuccessful, please make a move manually!')
                player.waitingoninput = True
            else:
                if movefunc != piece.lift:
                    if verbose:
                        print(' Moved: ',movefunc, piece)
                    player.reset()
                    self.state.update(self)
                else:
                    self.liftedpiece = piece
                    if verbose:
                        print('Lifting piece...', piece)
        else:
            '''
            Invalid/null move function or wrong type piece chosen
            '''
            print('Null/invalid move: ', movefunc ,' please make a move manually!')
            player.waitingoninput = True
                    
    

    def gamelogic(self):
        '''
        If an automatic player needs input, must return 'None' on
        call to predict.  Else, Player returns a move.

        '''
        while not self.winner:
            player = self.players[self.state.booleanturn()]
            if verbose:
                print('Turn: ', player)
                print('Phase: ', self.state.getphase())
            if player.waitingoninput:
                print('Waiting on input...')
                while player.waitingoninput:
                    self.gameBoard.window.update()
                if verbose:
                    print('Exiting input loop.')
                player.reset()
                self.state.update(self)
            else:
                if verbose:
                    print('Not waiting on input...')
                returnedtuple = player.predict()
                self.makemove(returnedtuple)
            print(player.identity(), ': done with move ', self.state.getmovecount())
            self.gameBoard.window.update()
            time.sleep(1)
        
        
    def input(self,*,position=None):
        print('Pressed button: '+position.address)
        if verbose:
            print('Adjacent: '+' '.join(position.neighbors_address))
            print('Captures: '+' '.join(position.captures_address))
        if position != None:
            # function called with an input
            # pass the input to the appropriate player
            # need stronger checks to make sure function returns what is expected
            if self.liftedpiece == None:
                if self.state.phase == 'move':
                    if position.content == None:
                        self.players[self.state.booleanturn()].reset()
                        print('Game: clicked on an empty square!')
                        return None
                    elif position.content.identity() != self.state.playerturn():
                        self.players[self.state.booleanturn()].reset()
                        print('Game: clicked on ',position.content.identity(),', but Turn is ',self.state.playerturn())
                        return None
                
            returnedtuple = self.players[self.state.booleanturn()].input(position=position)
            movefunc, piece, destaddress = self.checkmove(returnedtuple)
            if not piece:
                self.players[self.state.booleanturn()].reset()
            else:
                if movefunc != None and piece.identity() == self.state.playerturn():
                    if verbose:
                        print('Making move...')
                    if movefunc == piece.lift:
                        piece = movefunc()
                        if verbose:
                            print('Piece making move:', piece)
                    else:
                        piece = movefunc(destaddress)
                        if verbose:
                            print('Piece making move:', piece)

                    if not piece:
                        print('Invalid move, try again!')
                        self.players[self.state.booleanturn()].reset()
                    else:
                        if movefunc == piece.lift:
                            if verbose:
                                print('Lifting piece')
                            self.liftedpiece = piece
                else:
                    '''Function did
                    not return anything
                    '''
                    return None
            print('Exiting game:input')
            

    def stalemate(self):
        for tiger in self.state.tigers:
            for position in tiger.position.neighbors:
                if position.content == None:
                    ''' Move possible '''
                    if verbose:
                        print('Move possible for '+tiger.position.address)
                    return False
                elif position.content.identity() == 'Goat':
                    captures = position.neighbors_address.intersection(tiger.position.captures_address)
                    if captures:
                        for p in captures:
                            pass
                        if self.gameBoard.positions[p].content == None:
                            ''' Capture possible '''
                            if verbose:
                                print('Capture possible for '+tiger.position.address)
                            return False
        return True

    def gameover(self):
        '''When the 6th goat is eaten, the tigers cannot be in stalemate.
        The below conditional is only used when exactly one of the clauses
        is True or both are false

        '''
        if self.state.goatEaten == 6 or self.stalemate():
            if self.state.goatEaten == 6:
                self.winner = self.players[0]
                self.state.phase = 'over'
                return self.winner
            else:
                self.winner = self.players[1]
                self.state.phase = 'over'                
                return self.winner
        else:
            return None
        
        
class Board():

    numPosition = 23 
    boardSize = 500

    addresslist = ['a1', 'a2', 'a3', 
             'b0', 'b1', 'b2', 'b3', 'b4',
                   'c1', 'c2', 'c3', 'c4',
                   'd1', 'd2', 'd3', 'd4',
                   'e1', 'e2', 'e3', 'e4',
                   'f1', 'f2', 'f3']
    ''' 
    The following are the adjacency and capture matrices
    They are hardcoded from the geometry of the board, and
    must not be changed or modified in any way during runtime.
    A: Adjacency matrix for the board
       A[i,j] = 1 if i,j adjacent in the board
    C: The capture matrix for the board 
       C[i,j] = 1 if i and j are positions on the same line with one gap
    Do not use define_geometry() except to define A and C (and hence the board)
    '''

    A = np.zeros((numPosition,numPosition))
    def define_geometry(addresslist):
        G = np.zeros((23,23))
        T = np.zeros((23,23))
        # for a1:
        G[0,[x for x in range(23) if addresslist[x] in ['a2']]] = 1
        T[0,[x for x in range(23) if addresslist[x] in ['b1']]] = 1
        # for a2:
        G[1,[x for x in range(23) if addresslist[x] in ['a1','a3']]] =1
        T[1,[x for x in range(23) if addresslist[x] in ['b2']]] =1
        # for a3:
        G[2,[x for x in range(23) if addresslist[x] in ['a2']]] = 1
        T[2,[x for x in range(23) if addresslist[x] in ['b3']]] = 1
        # for b0:
        # Adding G[0,list]=1 will create paths of length 2 between
        # b1 and c1, for example. To avoid this, we define the
        # captures and adjacents for b0 separately
        
        # for b1:
        G[4,[x for x in range(23) if addresslist[x] in ['b0','b2']]] =1
        T[4,[x for x in range(23) if addresslist[x] in ['a1','c1']]] =1
        # for b2:
        G[5,[x for x in range(23) if addresslist[x] in ['b1','b3']]] =1
        T[5,[x for x in range(23) if addresslist[x] in ['a2','c2']]] =1
        # for b3:
        G[6,[x for x in range(23) if addresslist[x] in ['b2','b4']]] =1
        T[6,[x for x in range(23) if addresslist[x] in ['a3','c3']]] =1
        # for b4:
        G[7,[x for x in range(23) if addresslist[x] in ['b3']]] =1
        T[7,[x for x in range(23) if addresslist[x] in ['c4']]] =1
        # for c1:
        G[8,[x for x in range(23) if addresslist[x] in ['b0','c2']]] =1
        T[8,[x for x in range(23) if addresslist[x] in ['b1','d1']]] =1
        # for c2:
        G[9,[x for x in range(23) if addresslist[x] in ['c1','c3']]] =1
        T[9,[x for x in range(23) if addresslist[x] in ['b2','d2']]] =1
        # for c3:
        G[10,[x for x in range(23) if addresslist[x] in ['c2','c4']]] =1
        T[10,[x for x in range(23) if addresslist[x] in ['b3','d3']]] =1
        # for c4:
        G[11,[x for x in range(23) if addresslist[x] in ['c3']]] =1
        T[11,[x for x in range(23) if addresslist[x] in ['b4','d4']]] =1
        # for d1:
        G[12,[x for x in range(23) if addresslist[x] in ['b0','d2']]] =1
        T[12,[x for x in range(23) if addresslist[x] in ['c1','e1']]] =1
        # for d2:
        G[13,[x for x in range(23) if addresslist[x] in ['d1','d3']]] =1
        T[13,[x for x in range(23) if addresslist[x] in ['c2','e2']]] =1
        # for d3:
        G[14,[x for x in range(23) if addresslist[x] in ['d2','d4']]] =1
        T[14,[x for x in range(23) if addresslist[x] in ['c3','e3']]] =1
        # for d4:
        G[15,[x for x in range(23) if addresslist[x] in ['d3']]] =1
        T[15,[x for x in range(23) if addresslist[x] in ['c4','e4']]] =1
        # for e1:
        G[16,[x for x in range(23) if addresslist[x] in ['b0','e2']]] =1
        T[16,[x for x in range(23) if addresslist[x] in ['d1','f1']]] =1
        # for e2:
        G[17,[x for x in range(23) if addresslist[x] in ['e1','e3']]] =1
        T[17,[x for x in range(23) if addresslist[x] in ['d2','f2']]] =1
        # for e3:
        G[18,[x for x in range(23) if addresslist[x] in ['e2','e4']]] =1
        T[18,[x for x in range(23) if addresslist[x] in ['d3','f3']]] =1
        # for e4:
        G[19,[x for x in range(23) if addresslist[x] in ['e3']]] =1
        T[19,[x for x in range(23) if addresslist[x] in ['d4']]] =1
        # for f1:
        G[20,[x for x in range(23) if addresslist[x] in ['f2']]] =1
        T[20,[x for x in range(23) if addresslist[x] in ['e1']]] =1
        # for f2:
        G[21,[x for x in range(23) if addresslist[x] in ['f1','f3']]] =1
        T[21,[x for x in range(23) if addresslist[x] in ['e2']]] =1
        # for f3:
        G[22,[x for x in range(23) if addresslist[x] in ['f2']]] =1
        T[22,[x for x in range(23) if addresslist[x] in ['e3']]] =1
        A = G+T
        A[3,[x for x in range(23) if addresslist[x] in ['b1','c1','d1', 'e1']]] = 1

        '''V: Vertical direction capture positions These are obtained
        as paths of length 2 along one vertical line (hence G@ G).
        Since G omits outgoing adjacencies of b0, no path can pass
        through b0 (but can terminate in b0).  As a consequence we
        avoid paths of form b1->b0->c1 in G@G.  But we need to
        explicitly add paths starting from b0, which is done in the
        line V[3,..] = 1  '''
        V = G @ G
        V = V -np.diag(np.diag(V))
        V[3,[x for x in range(23) if addresslist[x] in ['b2','c2','d2', 'e2']]] = 1
        '''
        H: Vertical direction capture positions. No complications.
        '''
        H = T @ T
        H = H- np.diag(np.diag(H))
        C = V+H
        return A,C 
    A, C = define_geometry(addresslist)

    
    def __init__(self, graphics=False):
        self.boardPosition = {'b0':'X',
                              'a1':(), 'b1':(), 'c1':'X', 'd1':'X', 'e1':(), 'f1':(),
                              'a2':(), 'b2':(), 'c2':(), 'd2':(), 'e2':(), 'f2':(),
                              'a3':(), 'b3':(), 'c3':(), 'd3':(), 'e3':(), 'f3':(),
                              'b4':(), 'c4':(), 'd4':(), 'e4':(), 'out':()}
        
        self.game = None
        self.pieceinmove = None
        self.positions ={}
        self.graphics = graphics
        if self.graphics:
            self.window = Tk()
        self.initialize_board()
        for address in self.addresslist:
            self.positions[address] = Position(self,address)
        for address in self.addresslist:
            self.positions[address].initiate_geometry()
        
        
    def tomove(self, piece=None):
        self.pieceinmove = piece
        self.selectedToggle()

    def inmove(self):
        return self.pieceinmove
    
    def attachgame(self, game):
        self.game = game
        if self.graphics:
            self.turnDisplay()
            self.selectedToggle()
            self.turnDisp = Label(self.window,
                              font=(self.font, self.fontsize),
                              textvariable=self.turntext)
            self.turnDisp.place(x=self.boardSize-100,y= 30)

            self.selectedDisp = Label(self.window,
                                  font=(self.font, self.fontsize),
                                  textvariable=self.selectedBtn)
            self.selectedDisp.place(x=self.boardSize/2,y= self.boardSize - 30)

            self.numGoats.set("Number of goats: " + str(game.state.goatCount+1))        
            self.goatDisp = Label(self.window,
                              font=(self.font, self.fontsize),
                              textvariable=self.numGoats)
            self.goatDisp.place(x=10,y= 30)

            self.goatsEatentext.set("Goats eaten: " + str(self.game.state.goatEaten))        
            self.goatEatenDisp = Label(self.window,
                                   font=(self.font, self.fontsize),
                                   textvariable=self.goatsEatentext)
            self.goatEatenDisp.place(x= 10,y= 60)
            self.drawlines()
            self.placebuttons()
        
    def neighbors_address(self, address):
        index = self.addresslist.index(address)
        return {self.addresslist[x] for x in range(23) if self.A[index,x]}

    def neighbors(self, position):
        index = self.addresslist.index(position.address)
        return {self.positions[x] for x in self.addresslist if self.A[index,self.addresslist.index(x)]}

    def captures_address(self, address):
        index = self.addresslist.index(address)
        return {self.addresslist[x] for x in range(23) if self.C[index,x]}

    def captures(self, position):
        index = self.addresslist.index(position.address)
        return {self.positions[x] for x in self.addresslist if self.C[index,self.addresslist.index(x)]}
    
    def initialize_board(self):
        ''' Sets various parameters needed to draw the board '''
        if self.graphics:
            self.font = "Helvetica"
            self.fontsize = 16

            self.window.title('Huligutta (Goats & Tigers)')
            self.window.geometry('500x500')
            self.window.resizable(0,0)

            self.canvas = Canvas(self.window,width=self.boardSize,height=self.boardSize)
            self.canvas.pack()
            self.turntext = StringVar()
            self.selectedBtn = StringVar()
            self.numGoats = StringVar()
            self.goatsEatentext = StringVar()

    def drawlines(self):
        self.canvas.create_rectangle(self.boardSize/10,self.boardSize/2 - 70,
                                     self.boardSize-self.boardSize/10,self.boardSize/2+70)    
        self.canvas.create_line(self.boardSize/10,self.boardSize/2,
                                self.boardSize-self.boardSize/10,self.boardSize/2)
        self.canvas.create_line(self.boardSize/2,self.boardSize/10,
                                self.boardSize/10,self.boardSize - self.boardSize/10)
        self.canvas.create_line(self.boardSize/2,self.boardSize/10,
                                self.boardSize - self.boardSize/10,self.boardSize - self.boardSize/10)
        self.canvas.create_line(self.boardSize/10,self.boardSize - self.boardSize/10,
                                self.boardSize - self.boardSize/10,self.boardSize - self.boardSize/10)
        self.canvas.create_line(self.boardSize/2,self.boardSize/10,
                                self.boardSize/2 - 80,self.boardSize - self.boardSize/10)
        self.canvas.create_line(self.boardSize/2,self.boardSize/10,
                                self.boardSize/2 + 80,self.boardSize - self.boardSize/10)
        
    def selectedToggle(self):
        if self.graphics:
            if self.pieceinmove != None:
                self.selectedBtn.set(self.pieceinmove.identity()+': '+self.pieceinmove.position.address+'-> ?')
            else:
                self.selectedBtn.set('Select piece')
        else:
            pass
        
    def turnDisplay(self):
        # Displays turn as text in the window
        if self.graphics:
            self.turntext.set(self.game.state.playerturn())
            self.numGoats.set("Number of goats: " + str(self.game.state.goatCount))
            self.goatsEatentext.set("Goats eaten: " + str(self.game.state.goatEaten))
        else:
            pass
        
    def placebuttons(self):
        self.positions['a1'].button.place(x=self.boardSize/10,
                        y=self.boardSize/2-70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['a2'].button.place(x=self.boardSize/10,
                        y=self.boardSize/2,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['a3'].button.place(x=self.boardSize/10,
                        y=self.boardSize/2 + 70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['b0'].button.place(x=self.boardSize/2,
                        y=self.boardSize/10,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['b1'].button.place(x=self.boardSize/2 - 65,
                        y=self.boardSize/2 - 70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['b2'].button.place(x=self.boardSize/2- 100,
                        y=self.boardSize/2,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['b3'].button.place(x=self.boardSize/2-135,
                        y=self.boardSize/2+70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['b4'].button.place(x=self.boardSize/10,
                        y=self.boardSize - self.boardSize/10,
                        height=30,
                        width=30,
                        anchor=CENTER)

        self.positions['c1'].button.place(x=self.boardSize/2 - 25,
                       y=self.boardSize/2 - 70,
                       height=30,
                       width=30,
                       anchor=CENTER)
        self.positions['c2'].button.place(x=self.boardSize/2 - 38,
                         y=self.boardSize/2,
                         height=30,
                         width=30,
                         anchor=CENTER)
        self.positions['c3'].button.place(x=self.boardSize/2 - 53,
                         y=self.boardSize/2 + 70,
                         height=30,
                         width=30,
                         anchor=CENTER)
        self.positions['c4'].button.place(x= self.boardSize/2 - 80,
                         y=self.boardSize - self.boardSize/10,
                         height=30,
                         width=30,
                         anchor=CENTER)

        self.positions['d1'].button.place(x=self.boardSize/2 + 25,
                       y=self.boardSize/2 - 70,
                       height=30,
                       width=30,
                       anchor=CENTER)
        self.positions['d2'].button.place(x=self.boardSize/2 + 38,
                         y=self.boardSize/2,
                         height=30,
                         width=30,
                         anchor=CENTER)
        self.positions['d3'].button.place(x=self.boardSize/2 +53,
                         y=self.boardSize/2 + 70,
                         height=30,
                         width=30,
                         anchor=CENTER)
        self.positions['d4'].button.place(x= self.boardSize/2 + 80,
                         y=self.boardSize - self.boardSize/10,
                         height=30,
                         width=30,
                         anchor=CENTER)
        self.positions['e1'].button.place(x=self.boardSize/2 + 65,
                        y=self.boardSize/2 - 70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['e2'].button.place(x=self.boardSize/2 + 100,
                        y=self.boardSize/2,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['e3'].button.place(x=self.boardSize/2 + 135,
                        y=self.boardSize/2+70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['e4'].button.place(x=self.boardSize -self.boardSize/10,
                        y=self.boardSize - self.boardSize/10,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['f1'].button.place(x=self.boardSize -self.boardSize/10,
                        y=self.boardSize/2-70,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['f2'].button.place(x=self.boardSize - self.boardSize/10,
                        y=self.boardSize/2,
                        height=30,
                        width=30,
                        anchor=CENTER)
        self.positions['f3'].button.place(x=self.boardSize - self.boardSize/10,
                        y=self.boardSize/2 + 70,
                        height=30,
                        width=30,
                        anchor=CENTER)

class Position(Board):
    def __init__(self, board, address):
        if address not in super().addresslist:
            return -1
        self.board = board
        self.address = address
        self.history = []
        self.content = None
        self.neighbors = None
        self.neighbors_address = None
        self.captures = None
        self.captures_address = None
        self.graphics = board.graphics
        if self.graphics:
            self.emptyimage = ImageTk.PhotoImage(file='./images/empty.png',master=self.board.window) 
            self.button = Button(self.board.window,
                             image=self.emptyimage,
                             command=lambda : self.buttonpress())
        
    def initiate_geometry(self):
        self.positions = self.board.positions
        self.neighbors = super().neighbors(self)
        self.neighbors_address = super().neighbors_address(self.address)
        self.captures = super().captures(self)
        self.captures_address = super().captures_address(self.address)        

    def buttonpress(self):
        self.board.game.input(position = self)
        time.sleep(.25)
        
    def addpiece(self, piece):
        if self.content == None:
            if piece != None:
                self.history.append(self.content)
                self.content = piece
                if self.graphics:
                    newimage = piece.returnimage()
                    self.button.config(image=newimage)
                return piece
            else:
                print('Tried placing an empty piece')
                return None
        else:
            print('addpiece: Contents of '+self.address+' not empty!')
            return None

    def removepiece(self):
        if self.content == None:
            print('Nothing to remove')
            return None
        else:
            piece = self.content
            self.content = None
            self.history.append(self.content)
            if self.graphics:
                self.button.config(image=self.emptyimage)
            return piece

    def checkcontent(self):
        if self.content == None:
            return None
        else:
            return self.content.identity()
        
class Piece(Board):
    def __init__(self, board, number):
        self.board = board
        self.position = None
        self.number = number
        self.graphics = board.graphics
        if self.graphics:
            self.image = None
        self.state = None
        
    def place(self,address):
        if self.board.positions[address].addpiece(self):
            self.position = self.board.positions[address]
            self.board.tomove(None)
            self.state = 'Playing'
            return self
        else:
            print('There is already a piece at '+address)
            return None

    def step(self, address):
        ''' Step to address from self.position, returns 
        self if valid step. '''
        if self.position == None:
            ValueError('Place piece before move')

        if address in self.position.neighbors_address:
            if self.place(address) == None:
                # already piece in target address, put back
                self.place(self.position.address)
                return None
            else:
                # success!
                return self
        else:
            return None
                       
    def returnimage(self):
        return self.image

    def lift(self):
        ''' Returns self if no other piece is in move, None else.
         Uses the second argument for compatability with other functions,
         the address argument is ignored'''
        if self.board.inmove() == None:
            self.position.removepiece()
            self.board.tomove(self)
            return self
        else:
            print('Piece', self.board.inmove(), 'already in move, place it first!')

    def identity(self):
        return type(self).__name__


    def allmoves(self):
        allmoves = []
        for position in self.position.neighbors:
            if position.content == None:
                allmoves.append([self.move, self, position.address])
        return allmoves

class Goat(Piece):
    def __init__(self, Board, number):
        super().__init__(Board, number)
        self.state = 'Unused'
        if self.graphics:
            goat_img = Image.open('./images/goat.png')
            goat_img = goat_img.resize((30,30),Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(goat_img,master=self.board.window)

    def place(self,address):
        self.state = 'Playing'
        return super().place(address)

    def capture(self):
        self.position = None
        self.state = 'Captured'
        self.board.game.state.goatEaten = self.board.game.state.goatEaten +1

    def getstate(self):
        return self.state

    def move(self, address):
        if self.position == None:
            if self.place(address):
                return self
            else:
                print('Cannot place in ', address)
                return None
        else:
            if self.step(address):
                return self
            else:
                print('Cannot step to ', address)
                self.place(self.position.address)
                return None

    def allmoves(self):
        if self.getstate() == 'Playing':
            return super().allmoves()
        if self.getstate() == 'Captured':
            return []
        if self.getstate() == 'Unused':
            ''' We are in the place phase '''
            allmoves = []
            for address in self.board.addresslist:
                if self.board.positions[address].content == None:
                    allmoves.append([self.place, self, address])
            return allmoves
            


            
class Tiger(Piece):
    def __init__(self, Board, number):
        super().__init__(Board, number)
        if self.graphics:
            tiger_img = Image.open('./images/tiger-512.png')
            tiger_img = tiger_img.resize((30,30),Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(tiger_img,master=self.board.window)
        
    def capture(self, address):
        ''' Returns goat if capture successful, else returns None'''
        if address in self.position.captures_address:
            # Calculate middle position
            middleposition = self.board.positions[address].neighbors_address.intersection(
                             self.position.neighbors_address)
            if len(middleposition) >= 1:
                for m in middleposition:
                    ''' b0 can show up in this list, but 
                    b0 can never be the middle position.'''
                    if m != 'b0':
                        break
        else:
            print('Cannot capture to position: '+address+' from '+self.position.address)
            return None
        if self.board.positions[m].checkcontent() == 'Goat':
            # valid capture if capture address is empty
            # Why was this here? self.lift()
            if self.place(address) != None:
                # address is empty
                goat = self.board.positions[m].removepiece()
                goat.capture()
                return goat
            else:
                # address was not empty, put piece back
                print('Position '+address+' must be empty for capture!')
                self.place(self.position.address) # put piece back
                return None
        else:
            print('No goat in position '+m)
            return None

    def move(self, address):
        if self.step(address):
            return self
        else:
            if verbose:
                print('Cannot step to ', address)
                print('Trying a capture...')
            piece = self.capture(address)
            if piece:
                return piece
            else:
                self.place(self.position.address)
                return None

    def allmoves(self):
        allmoves = super().allmoves()
        allcaptures = []
        goats = []
        for position in self.position.neighbors:
            if position.content != None:
                if position.content.identity() == 'Goat':
                    if position.address!= 'b0':
                        '''
                        b0 can never be the middle position of a capture.
                        '''
                        captures = position.neighbors_address.intersection(self.position.captures_address)
                        if captures:
                            for p in captures:
                                ''' captures contains 1 element in this case 
                                the loop assigns that element to p.  '''
                                pass
                            if self.board.positions[p].content == None:
                                allcaptures.append([self.move, self, p])
                                goats.append(position.content)
        return allmoves, allcaptures, goats
 
        
if __name__ == '__main__':
    gameone = Game()
    boardone = Board(graphics = True)
    gameone.attachboard(boardone)
    tiger = greedyTiger(gameone)
    goat = twostepGoat(gameone)
    gameone.addplayers(tiger, goat)
    gameone.gamelogic()
