###Reinforcement Learning 

import pygame 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

#window size

WIDTH = 360
HEIGHT = 240
FPS = 144

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

class Player(pygame.sprite.Sprite):
    #sprite for the player 
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20)) #Agent
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        self.rect.centerx = WIDTH/2
        self.rect.centery = HEIGHT - 10
        self.speedx = 0
        self.speedy = 0
        
        
    def update(self, action):
        self.speedx = 0
        self.speedy = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -7
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 7
            '''
        elif keystate[pygame.K_DOWN] or action == 2:
            self.speedy = -1
        elif keystate[pygame.K_UP] or action == 3:
            self.speedy = 1
            '''

        else:
            self.speedx == 0
            
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            
            
    def getCoordinates(self):
        return(self.rect.x, self.rect.y)




class Enemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 8
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.x = random.randrange(0,WIDTH - self.rect.width)
        self.rect.y = random.randrange(2,6)
        
        self.speedx = 0
        self.speedy = 8
        
        
    def update(self):
        
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
    
        if self.rect.top > HEIGHT + 1:
             self.rect.x = random.randrange(0,WIDTH - self.rect.width) # Merkezden aldığımızdan ekrana sığdırma yöntemidir. Çerçevenin dışından gelemeyecek.
             self.rect.y = random.randrange(2,6)
             self.speedy = 8
             self.speedx = 0




    def getCoordinates(self):
        return(self.rect.x, self.rect.y)       

'''
class SecEnemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, BLUE, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, 360)
        self.rect.y = random.randrange(0,HEIGHT)
        
        self.speedx = 3
        self.speedy = 0
        
        
    def update(self):
        
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
    
        if self.rect.right > WIDTH + 1:
             self.rect.x = random.randrange(0,HEIGHT - self.rect.height) # Merkezden aldığımızdan ekrana sığdırma yöntemidir. Çerçevenin dışından gelemeyecek.
             self.rect.y = random.randrange(2,6)
             self.speedx = 3
             self.speedy = 0

             

    def getCoordinates(self):
        return(self.rect.x, self.rect.y)       

    '''            


        
# class Point(pygame.sprite.Sprite):
    
#     def __init__(self):
#         pygame.sprite.Sprite.__init__(self)
#         self.image = pygame.Surface((10,10))
#         self.image.fill(GREEN)
#         self.rect = self.image.get_rect()
#         self.radius = 10
#         pygame.draw.circle(self.image, GREEN, self.rect.center, self.radius)
#         self.rect.x = random.randrange(0, WIDTH - self.rect.width)
#         self.rect.y = random.randrange(0, HEIGHT - self.rect.height)
#         self.speedx = 0
#         self.speedy = 0
        
#     def update(self):
        
#         self.rect.x += self.speedx
#         self.rect.y += self.speedy
        
            
                        
#         # Sağ duvara çarptığında hızını tersine çevir            
#         if self.rect.right > WIDTH:
#             self.rect.right = WIDTH
#             self.speedx = -self.speedx
            
#         # Sol duvara çarptığında hızını tersine çevir            
#         if self.rect.left < 0:
#             self.rect.left = 0
#             self.speedx = -self.speedx 
        
#         # Üst duvara çarptığında hızını tersine çevir
#         if self.rect.top < 0:
#             self.rect.top = 0
#             self.speedy = -self.speedy 
            
#         # Alt duvara çarptığında hızını tersine çevir
#         if self.rect.bottom > HEIGHT/2:
#             self.rect.bottom = HEIGHT/2
#             self.speedy = -self.speedy 
            
#     def getCoordinates(self):
#         return(self.rect.x, self.rect.y)         




            
class DQLAgent:
    def __init__(self):
        #parameter
        self.state_size = 10 #distance
        self.action_size = 3 #right, left ,up, down, none
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        
        

    def build_model(self):
        #neural network for deep q learning
        
        model = Sequential()
        
        model.add(Dense(64,input_dim = self.state_size, activation ="relu"))
        model.add(Dense(64, activation ="relu"))
        model.add(Dense(64, activation ="relu"))
        model.add(Dense(64, activation ="relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer= Adam(learning_rate = self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        
        #storage
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        
        #act exploer or exploit
        
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state,verbose = 0)
            return np.argmax(act_values[0])
        

                
    def replay(self, batch_size):
            if len(self.memory) < batch_size:
                return
            minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    
    def adapativeEGreddy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
            
            
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.points = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        '''
        self.m6 = SecEnemy()
        self.m7 = SecEnemy()
        self.m8 = SecEnemy()
        self.m9 = SecEnemy()
        self.m10 = SecEnemy()
        '''
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        '''
        self.all_sprite.add(self.m6)
        self.all_sprite.add(self.m7)
        self.all_sprite.add(self.m8)
        self.all_sprite.add(self.m9)
        self.all_sprite.add(self.m10)
        '''
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        '''
        self.enemy.add(self.m6)
        self.enemy.add(self.m7)
        self.enemy.add(self.m8)
        self.enemy.add(self.m9)
        self.enemy.add(self.m10)
        '''
        '''
        self.point = Point()
        self.point2 = Point()
        self.point3 = Point()
        self.all_sprite.add(self.point)
        self.all_sprite.add(self.point2)
        self.all_sprite.add(self.point3)
        self.points.add(self.point)
        self.points.add(self.point2)
        self.points.add(self.point3)
        '''

        
        
        self.reward = 0 
        self.total_reward = 0 
        self.done = False
        self.agent = DQLAgent()
        
        
    def findDistance(self, a, b):
        d = a-b
        return d
    

    def step(self,action):
        state_list = []
        
        #updates
        self.player.update(action)
        self.enemy.update()
        self.points.update()
        
        #get cordinates
        next_player_state = self.player.getCoordinates()
        
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        next_m3_state = self.m3.getCoordinates()
        next_m4_state = self.m4.getCoordinates()
        next_m5_state = self.m5.getCoordinates()
        '''
        next_m6_state = self.m6.getCoordinates()
        next_m7_state = self.m7.getCoordinates()
        next_m8_state = self.m8.getCoordinates()
        next_m9_state = self.m9.getCoordinates()
        next_m10_state = self.m10.getCoordinates()
        '''
        
        '''
        next_point_state = self.point.getCoordinates()
        next_point2_state = self.point2.getCoordinates()
        next_point3_state = self.point3.getCoordinates()
        '''
        


        #find distance 
        
        state_list.append(self.findDistance(next_player_state[0], next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m2_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m3_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m3_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m4_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m4_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m5_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m5_state[1]))
        '''
        state_list.append(self.findDistance(next_player_state[0], next_m6_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m6_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m7_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m7_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m8_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m8_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m9_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m9_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m10_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m10_state[1]))
        '''
        '''
        state_list.append(self.findDistance(next_player_state[0], next_point_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_point_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_point2_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_point2_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_point3_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_point3_state[1]))
        '''


        
        return [state_list]


    #reset
    def initialState(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.points = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        '''
        self.m6 = SecEnemy()
        self.m7 = SecEnemy()
        self.m8 = SecEnemy()
        self.m9 = SecEnemy()
        self.m10 = SecEnemy()
        '''
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        '''
        self.all_sprite.add(self.m6)
        self.all_sprite.add(self.m7)
        self.all_sprite.add(self.m8)
        self.all_sprite.add(self.m9)
        self.all_sprite.add(self.m10)
        '''
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        '''
        self.enemy.add(self.m6)
        self.enemy.add(self.m7)
        self.enemy.add(self.m8)
        self.enemy.add(self.m9)
        self.enemy.add(self.m10)
        '''
        '''
        self.point = Point()
        self.point2 = Point()
        self.point3 = Point()
        self.all_sprite.add(self.point)
        self.all_sprite.add(self.point2)
        self.all_sprite.add(self.point3)
        self.points.add(self.point)
        self.points.add(self.point2)
        self.points.add(self.point3)
        '''

        

        
        self.reward = 0 
        self.total_reward = 0 
        self.done = False
        
        state_list = []
        
        #get coordinate 
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()
        m3_state = self.m3.getCoordinates()
        m4_state = self.m4.getCoordinates()
        m5_state = self.m5.getCoordinates()
        '''
        m6_state = self.m6.getCoordinates()
        m7_state = self.m7.getCoordinates()
        m8_state = self.m8.getCoordinates()
        m9_state = self.m9.getCoordinates()
        m10_state = self.m10.getCoordinates()
        '''
        '''
        point_state = self.point.getCoordinates()
        point2_state = self.point2.getCoordinates()
        point3_state = self.point3.getCoordinates()
        '''

        

        
        
        # find distance                
        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))
        state_list.append(self.findDistance(player_state[0], m2_state[0]))
        state_list.append(self.findDistance(player_state[1], m2_state[1]))
        state_list.append(self.findDistance(player_state[0], m3_state[0]))
        state_list.append(self.findDistance(player_state[1], m3_state[1]))
        state_list.append(self.findDistance(player_state[0], m4_state[0]))
        state_list.append(self.findDistance(player_state[1], m4_state[1]))
        state_list.append(self.findDistance(player_state[0], m5_state[0]))
        state_list.append(self.findDistance(player_state[1], m5_state[1]))
        '''
        state_list.append(self.findDistance(player_state[0], m6_state[0]))
        state_list.append(self.findDistance(player_state[1], m6_state[1]))
        state_list.append(self.findDistance(player_state[0], m7_state[0]))
        state_list.append(self.findDistance(player_state[1], m7_state[1]))
        state_list.append(self.findDistance(player_state[0], m8_state[0]))
        state_list.append(self.findDistance(player_state[1], m8_state[1]))
        state_list.append(self.findDistance(player_state[0], m9_state[0]))
        state_list.append(self.findDistance(player_state[1], m9_state[1]))
        state_list.append(self.findDistance(player_state[0], m10_state[0]))
        state_list.append(self.findDistance(player_state[1], m10_state[1]))
        '''
        '''
        state_list.append(self.findDistance(player_state[0], point_state[0]))
        state_list.append(self.findDistance(player_state[1], point_state[1]))
        state_list.append(self.findDistance(player_state[0], point2_state[0]))
        state_list.append(self.findDistance(player_state[1], point2_state[1]))
        state_list.append(self.findDistance(player_state[0], point3_state[0]))
        state_list.append(self.findDistance(player_state[1], point3_state[1]))
        '''

        return [state_list]
    
    

    
    def run(self):
        #game loop
        
        state = self.initialState()
        running = True
        batch_size = 1
        while running:
            self.reward = 1
            self.total_reward += self.reward                

            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            #update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            
            hits = pygame.sprite.spritecollide(self.player,self.enemy,False , pygame.sprite.collide_circle)
            if hits:
                self.reward = -500
                self.total_reward += self.reward                
                self.done = True
                running = False
                print("Total reward: ",self.total_reward)

            '''
            touch = pygame.sprite.spritecollide(self.player,self.points, False , pygame.sprite.collide_circle)
            if touch:
                self.reward = 500
                self.total_reward += self.reward
                self.done = False
                running = False
                print("Total reward: ",self.total_reward)
            '''
            #storage
            self.agent.remember(state, action, self.reward, next_state, self.done)
                        
            if len(self.agent.memory) > batch_size:
                self.agent.replay(batch_size)

            #update
            state = next_state 
                        
            #epsilon
            self.agent.adapativeEGreddy()
            
            #draw
            screen.fill(BLACK)
            self.all_sprite.draw(screen)
            #after drawing flip display
            pygame.display.flip()


            
            
                
    pygame.quit()

if __name__ =="__main__":
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1 
        print("Episode: ",t)
        liste.append(env.total_reward)
        
        #initilaize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        
        env.run()

    
        
        
        

        
        
        
        
        
        