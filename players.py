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
import sys
from PIL import ImageTk, Image
from random import randint, choice
import itertools

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
        print('Entering Player:input')
        if position == None:
            return self.fit()
        else:
            return self.handle_input(position)

    def handle_input(self, position):
        print('Entering Player:handle_input')
        if self.positionone == None:
            print('Assign positionone: ', position.address)
            self.positionone = position
            return self.fit()
        else:
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
        print('Entering Player:fit ', self.positionone, self.positiontwo)
        if self.positiontwo == None:
            '''Only positionone is assigned, we are waiting on another input. '''
            piece = self.positionone.content
            if piece != None:
                print('Clicked on a ',piece.identity())
            else:
                print('Clicked on an empty square')
            if piece.identity() == self.identity():
                print('Lifting piece..')
                func = piece.lift
                self.waitingoninput = True
                return func, piece, piece.position
            else:
                print('Choose a', self.playeridentity, ' to move!')
                self.positionone = None
                self.positiontwo = None
                self.waitingoninput = True
                return None, None, None
        else:
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
        self.pieces = self.game.tigers
        self.waitingoninput = False

    def predict(self):
        print('Entering greedyTiger:predict')
        randomorder = choice(list(itertools.permutations(self.pieces)))
        
        for tiger in randomorder:
            moves, captures = tiger.allmoves()
            for move in moves:
                print('Possible moves: ', move[1].position.address,move[2].address)

            for capture in captures:
                print('Possible captures: ', capture[1].position.address, capture[2].address)
            if captures:
                onecapture = choice(captures)
                onecapture[1].lift(onecapture[1].position.address)
                return onecapture[0], onecapture[1], onecapture[2].address

        for tiger in randomorder:
            moves, captures = tiger.allmoves()
            if moves:
                onemove = choice(moves)
                onemove[1].lift(onemove[1].position.address)
                return onemove[0], onemove[1], onemove[2].address

    def reset(self):
        self.positionone = None
        self.positiontwo = None
        self.waitingoninput = False
        
class TigerPlayer(Player):
    ''' Human input interface '''
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Tiger"
        self.pieces = self.game.tigers

    def predict(self):
        pass

        
class GoatPlayer(Player):
    '''Human input interface'''
    def __init__(self,game):
        super().__init__(game)
        self.playeridentity = "Goat"
        self.pieces = self.game.goats
        
    def fit(self):
        ''' Called only after self.positionone assigned. '''
        print('Entering Goat.fit()')
        if self.game.state.getphase() == 'place':
            print('In place phase')
            print('Movecount: ', self.game.state.getmovecount())
            piece = self.pieces[self.game.state.getmovecount()] 
            func = piece.place
            dest = self.positionone.address
            self.waitingoninput = False
            return func, piece, dest
        else:
            print('In move phase')
            print('Movecount: ', self.game.state.getmovecount())
            return super().fit()

    def needinput(self):
        return True
    
    def predict(game):
        pass

    
