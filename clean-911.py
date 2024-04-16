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

    
class Game():
    '''This class performs the logic associated with the game: determining whose turn, who won (if any), passing on inputs (if any), making the moves, and keeping track of the state of the game.'''
    class State():
        def __init__(self):
            self.matrix = np.zeros((23,3))
            self.movecount = 0
            self.players = ['Tiger', 'Goat']
            self.goatturn = False
            self.phase = 'place'

        def booleanturn(self):
            return self.goatturn

        def playerturn(self):
            print('game.state.whoseturn:',self.players[self.goatturn])
            if self.players[self.goatturn]:
                return self.players[self.goatturn]
            else:
                return None
        
        def board2matrix(self,gameBoard):
            pass
        
        def update(self,game):
            self.matrix = self.board2matrix(game.gameBoard)
            self.winner = game.gameover()
            if not self.winner:
                if self.goatturn:
                    self.movecount = self.movecount +1
                    if self.movecount < 15:
                        self.phase = 'place'
                        game.goatCount = self.movecount
                    else:
                        self.phase = 'move'
                self.goatturn = not self.goatturn
                game.gameBoard.turnDisplay()
            else:
                winner = self.winner.identity()
                messagebox.showinfo("Game Over", winner+" wins!") 
        def resetchange(self):
            self.changed = False

        def changeindicator(self):
            return self.changed

        def getphase(self):
            return self.phase

        def getmovecount(self):
            return self.movecount
        
    def __init__(self):
        self.playersname = ['Tiger', 'Goat']
        self.state = self.State()
        
        self.goatEaten = 0
        self.goatCount = 0
        self.moveCount = 0
        self.gameBoard = None

        self.numTigers = 3
        self.numGoats = 15
        self.winner = None
        self.liftedpiece = None
        
    def attachboard(self, board):
        self.gameBoard = board
        self.gameBoard.attachgame(self)
        self.tigers =[]
        self.goats = []
        for number in range(self.numTigers):
            self.tigers.append(Tiger(self.gameBoard, number))
        for number in range(self.numGoats):
            self.goats.append(Goat(self.gameBoard, number))
        self.tigers[0].place('b0')
        self.tigers[1].place('c1')
        self.tigers[2].place('d1')
        self.state.update(self)

    def addplayers(self,tigerplayer, goatplayer):
        self.players = [tigerplayer, goatplayer]
        for player in self.players:
            player.reset()

    def gamelogic(self):
        ''' If Player needs input, must return 'wait' on call to predict.
        Else, Player returns a move.'''
        while not self.winner:
            player = self.players[self.state.booleanturn()]
            print('Turn: ', player)
            print('Phase: ', self.state.getphase())
            if player.waitingoninput:
                print('Waiting on input...')
                while player.waitingoninput:
                    self.gameBoard.window.update()
                print('Exiting input loop.')
                player.reset()
                self.state.update(self)
            else:
                print('Not waiting on input...')
                returnedtuple = player.predict()
                if returnedtuple != None:
                    if len(returnedtuple) == 3:
                        movefunc, piece, dest = returnedtuple
                        if movefunc != None and piece.identity() == self.state.playerturn():
                            piece = movefunc(dest)
                            print(piece)
                            if not piece:
                                '''Function did not return anything or
                                wrong type piece chosen'''
                                print('Move error: ',player.identity(),' please make a move manually!')
                                player.waitingoninput = True
                            else:
                                if movefunc != piece.lift:
                                    print('Updating?...')
                                    player.reset()
                                    self.state.update(self)
                                else:
                                    self.liftedpiece = piece
                                    print('Lifting piece...', piece)
                else:
                    '''Function did not return anything or
                    wrong type piece chosen'''
                    print('Move error: ',player.identity(),' please make a move manually!')
                    player.waitingoninput = True
            print('Done with one turn')        
        
    def input(self,*,position=None):
        print('Pressed button: '+position.address)
        print('Adjacent: '+' '.join(position.neighbors_address))
        print('Captures: '+' '.join(position.captures_address))
        if position != None:
            # function called with an input
            # pass the input to the appropriate player
            # need stronger checks to make sure function returns what is expected
            movefunc, piece, dest = self.players[self.state.booleanturn()].input(position=position)
            if not piece:
                self.players[self.state.booleanturn()].reset()
            else:
                if movefunc != None and piece.identity() == self.state.playerturn():
                    print('Making move...')
                    piece = movefunc(dest)
                    print('Piece making move:', piece)
                    if not piece:
                        print('Invalid move, try again!')
                        self.players[self.state.booleanturn()].reset()
                    else:
                        if movefunc != piece.lift:
                            print('Updating?...')
                        else:
                            print('Lifting piece')
                            self.liftedpiece = piece
                else:
                    '''Function did not return anything because wrong type piece chosen'''
                    return None
            print('Exiting game:input')
            

    def stalemate(self):
        for tiger in self.tigers:
            for position in tiger.position.neighbors:
                if position.content == None:
                    ''' Move possible '''
                    print('Move possible for '+tiger.position.address)
                    return False
                elif position.content.identity() == 'Goat':
                    captures = position.neighbors_address.intersection(tiger.position.captures_address)
                    if captures:
                        for p in captures:
                            pass
                        if self.gameBoard.positions[p].content == None:
                            ''' Capture possible '''
                            print('Capture possible for '+tiger.position.address)
                            return False
        return True

    def gameover(self):
        '''When the 6th goat is eaten, the tigers cannot be in stalemate.
        The below conditional is only used when exactly one of the clauses
        is True or both are false

        '''
        if self.goatEaten == 6 or self.stalemate():
            if self.goatEaten == 6:
                self.winner = self.players[0]
                return self.winner
            else:
                self.winner = self.players[1]
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

    
    def __init__(self):
        self.boardPosition = {'b0':'X',
                              'a1':(), 'b1':(), 'c1':'X', 'd1':'X', 'e1':(), 'f1':(),
                              'a2':(), 'b2':(), 'c2':(), 'd2':(), 'e2':(), 'f2':(),
                              'a3':(), 'b3':(), 'c3':(), 'd3':(), 'e3':(), 'f3':(),
                              'b4':(), 'c4':(), 'd4':(), 'e4':(), 'out':()}
        
        self.game = None
        self.pieceinmove = None
        self.positions ={}
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

        self.numGoats.set("Number of goats: " + str(game.goatCount+1))        
        self.goatDisp = Label(self.window,
                              font=(self.font, self.fontsize),
                              textvariable=self.numGoats)
        self.goatDisp.place(x=10,y= 30)

        self.goatsEatentext.set("Goats eaten: " + str(self.game.goatEaten))        
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
        if self.pieceinmove != None:
            self.selectedBtn.set(self.pieceinmove.identity()+': '+self.pieceinmove.position.address+'-> ?')
        else:
            self.selectedBtn.set('Select piece')

    def turnDisplay(self):
        # Displays turn as text in the window
        self.turntext.set(self.game.state.playerturn())
        self.numGoats.set("Number of goats: " + str(self.game.goatCount))
        self.goatsEatentext.set("Goats eaten: " + str(self.game.goatEaten))        
        
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
        self.content = None
        self.neighbors = None
        self.neighbors_address = None
        self.captures = None
        self.captures_address = None
        self.emptyimage = ImageTk.PhotoImage(file='./images/empty.png') 
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
                self.content = piece
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
        self.number = 0
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

    def lift(self, address):
        ''' Returns self if no other piece is in move, None else.
         Uses the second argument for compatability with other functions,
         the address argument is ignored'''
        if self.board.inmove() == None:
            self.position.removepiece()
            self.board.tomove(self)
            return self
        else:
            print('Piece already in move, place it first!')

    def identity(self):
        return type(self).__name__
        
class Goat(Piece):
    def __init__(self, Board, number):
        super().__init__(Board, number)
        goat_img = Image.open('./images/goat.png')
        goat_img = goat_img.resize((30,30),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(goat_img)

        
    def capture(self):
        self.position = None
        self.state = 'Captured'
        self.board.game.goatEaten = self.board.game.goatEaten +1

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
                

            
class Tiger(Piece):
    def __init__(self, Board, number):
        super().__init__(Board, number)
        tiger_img = Image.open('./images/tiger-512.png')
        tiger_img = tiger_img.resize((30,30),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(tiger_img)
        
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
            print('Cannot step to ', address)
            print('Trying a capture...')
            piece = self.capture(address)
            if piece:
                return piece
            else:
                self.place(self.position.address)
                return None

    def allmoves(self):
        allmoves = []
        allcaptures = []
        for position in self.position.neighbors:
            if position.content == None:
                allmoves.append([self.move, self, position])
            elif position.content.identity() == 'Goat':
                if position.address != 'b0':
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
                            allcaptures.append([self.move, self, self.board.positions[p]])
        return allmoves, allcaptures
 
        
if __name__ == '__main__':
    gameone = Game()
    boardone = Board()
    gameone.attachboard(boardone)
    tiger = greedyTiger(gameone)
    goat = GoatPlayer(gameone)
    gameone.addplayers(tiger, goat)
    gameone.gamelogic()
