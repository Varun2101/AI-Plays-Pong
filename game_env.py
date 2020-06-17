###############################################################################################
#Use this file to play the pong game yourself.
#Up and down arrow keys for P1, W and S for P2
#This file has nothing to do with the AI, it is just the game environment used.
###############################################################################################
import pygame
import random
pygame.init()
w, h = 720, 400

class Paddle:
    def __init__(self, x):
        self.width = 16
        self.height = 80
        self.x = x
        self.y = (h - self.height)//2
        self.vel = 4
        self.currvel = 0  # for applying spin to ball
        self.keys = {'up': False, 'down': False}

    def move(self):
        if self.keys['up'] and self.y > 0:
            self.y -= self.vel
            self.currvel = -2
            if self.y < 0:
                self.y = 0
                self.currvel = 0
        if self.keys['down'] and self.y < h - self.height:
            self.y += self.vel
            self.currvel = 2
            if self.y > h - self.height:
                self.y = h - self.height
                self.currvel = 0

class Ball:
    def __init__(self):
        self.radius = 8
        self.x = w // 2
        self.y = h // 2
        self.xvel = random.choice([-6, 6])
        self.yvel = random.randrange(-4, 5, 2)
        self.softcap = 8

    def move(self, box_obs):
        self.x += self.xvel
        self.y += self.yvel
        # make sure you actually collide for a single frame before changing direction
        if self.y <= self.radius:
            self.y = self.radius
            self.yvel = -self.yvel
        if self.y >= h - self.radius:
            self.y = h - self.radius
            self.yvel = -self.yvel

        x_cooldown, y_cooldown = 0, 0  # to prevent multiple collisions in a single axis in a single frame
        for box in box_obs:
            if y_cooldown == 1 and x_cooldown == 1:
                break  # exits early to save computation
            #collision from left
            if x_cooldown == 0 and 0 < box.x - self.x <= self.radius and box.y - self.radius / 2 <= self.y <= box.y + box.height + self.radius / 2:
                self.x = box.x - self.radius
                self.xvel = -self.xvel
                x_cooldown = 1
                self.yvel += box.currvel
                if self.yvel > 0 and self.yvel > self.softcap:
                    self.yvel = self.softcap
                elif self.yvel < 0 and self.yvel < -self.softcap:
                    self.yvel = -self.softcap
            #collision from right
            elif x_cooldown == 0 and 0 < self.x - box.x - box.width <= self.radius and box.y - self.radius / 2 <= self.y <= box.y + box.height + self.radius / 2:
                self.x = box.x + box.width + self.radius
                self.xvel = -self.xvel
                x_cooldown = 1
                self.yvel += box.currvel
                if self.yvel > 0 and self.yvel > self.softcap:
                    self.yvel = self.softcap
                elif self.yvel < 0 and self.yvel < -self.softcap:
                    self.yvel = -self.softcap
            # collision from above
            elif y_cooldown == 0 and 0 < box.y - self.y <= self.radius and box.x - self.radius <= self.x <= box.x + box.width + self.radius:
                self.y = box.y - self.radius
                self.yvel = -self.yvel
                y_cooldown = 1  # so that it doesn't register multiple collisions in the same frame
            # collision from below
            elif y_cooldown == 0 and 0 < self.y - box.y - box.height <= self.radius and box.x - self.radius <= self.x <= box.x + box.width + self.radius:
                self.y = box.y + box.height + self.radius
                self.yvel = -self.yvel
                y_cooldown = 1

        if self.x > w - 4 or self.x < 4:  # kill ball
            return 1
        return 0

def draw_window(win, paddles, ball, score2, score1, indicator=False, font=pygame.font.SysFont("comicsans", 30), game_num=0):
    win.fill((0, 0, 0))
    for i in range(h//20):
        pygame.draw.rect(win, (127, 127, 127), ((w - 10)//2, (i * h//20) + 3, 10, h//20 - 6))
    for paddle in paddles:
        pygame.draw.rect(win, (255, 255, 255), (paddle.x, paddle.y, paddle.width, paddle.height))
    pygame.draw.circle(win, (255, 255, 255), (ball.x, ball.y), ball.radius)
    score2_label = font.render(str(score2), 1, (255, 255, 255))
    win.blit(score2_label, (10, 10))
    score1_label = font.render(str(score1), 1, (255, 255, 255))
    win.blit(score1_label, (w - score1_label.get_width() - 10, 10))
    if indicator:
        for i in range(2, 50, 2):
            pygame.draw.circle(win, (255, 255, 255), (ball.x + i * ball.xvel, ball.y + i * ball.yvel), 2)
    if game_num:
        game_label = font.render("Game:" + str(game_num), 1, (255, 255, 255))
        win.blit(game_label, ((w - game_label.get_width()) // 2, 10))
        pygame.display.update()
    pygame.display.update()

def main():
    win = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Pong")
    font = pygame.font.SysFont("comicsans", 30)
    win_font = pygame.font.SysFont("comicsans", 50)
    clock = pygame.time.Clock()
    running = True
    paddles = [Paddle(8), Paddle(w - 24)]
    ball = Ball()
    dead_ball = 0
    score2, score1 = 0, 0
    draw_window(win, paddles, ball, score2, score1, True)
    pygame.time.delay(1000)
    while running:
        clock.tick(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if score1 == 10:
            win_label = win_font.render("P1 WINS!", 1, (255, 255, 255))
            win.blit(win_label, ((w - win_label.get_width()) // 2, (h - win_label.get_height()) // 2))
            pygame.display.update()
            pygame.time.delay(1000)
            break
        if score2 == 10:
            win_label = win_font.render("P2 WINS!", 1, (255, 255, 255))
            win.blit(win_label, ((w - win_label.get_width()) // 2, (h - win_label.get_height()) // 2))
            pygame.display.update()
            pygame.time.delay(1000)
            break

        for paddle in paddles:
            paddle.currvel = 0
            paddle.keys['up'] = False
            paddle.keys['down'] = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            paddles[1].keys['up'] = True
        if keys[pygame.K_DOWN]:
            paddles[1].keys['down'] = True
        if keys[pygame.K_w]:
            paddles[0].keys['up'] = True
        if keys[pygame.K_s]:
            paddles[0].keys['down'] = True
        for paddle in paddles:
            paddle.move()
        dead_ball = ball.move(paddles)

        if dead_ball:
            if ball.x > w//2:
                score2 += 1
            else:
                score1 += 1
            draw_window(win, paddles, ball, score2, score1)
            pygame.time.delay(1000)
            if score1 != 10 and score2 != 10:
                del ball
                del paddles[0:2]
                paddles.extend([Paddle(8), Paddle(w - 24)])
                ball = Ball()
                draw_window(win, paddles, ball, score2, score1, True)
                pygame.time.delay(1000)
        else:
            draw_window(win, paddles, ball, score2, score1)

    pygame.quit()

if __name__ == "__main__":
    main()
