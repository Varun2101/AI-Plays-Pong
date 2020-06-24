###############################################################################################
# Use this file to train the Deep Q-Learning network to play pong against itself.
# Heuristic is defined to make it learn to defend first, then be as aggressive as possible.
# IMPORTANT NOTE ABOUT CODE: the convention used is that "agent 1" is the one on the right
#                           and "agent 2" is on the left. This means that reward1 and score1
#                           are for the right paddle and reward2 and score2 for the left.
#                           However, the paddles list has them in the order [left, right].
###############################################################################################
import pygame
from game_env import Paddle, Ball, draw_window
from agents import DQN
import pickle

pygame.init()
w, h = 720, 400

filepath = None  # add the name of the model here to load it in (without extension as config file will use the same name)


def get_states(paddles, ball):
    """
    Returns the state of the game from each paddle's perspective to be used as inputs for the DQN
    Inputs for DQN (i.e. state, all inputs will be normalized):
    >> recommended direction
    >> y of opponent's middle
    >> predicted y of ball when it reaches opponent
    >> relative x distance of ball from agent
    >> yvel of ball
    """
    relx_left = ball.x - ball.radius - (paddles[0].x + paddles[0].width)
    relx_right = paddles[1].x - (ball.x + ball.radius)
    # predicting where the ball would hit on left and right sides, assuming no spin is applied
    if ball.xvel < 0:
        pred_left = ball.y + ball.yvel * relx_left // abs(ball.xvel)
        mod = 1
        if pred_left < 0:
            pred_left = -pred_left
            mod = -1
        elif pred_left > h:
            pred_left = -pred_left + 2 * h
            mod = -1
        pred_right = pred_left + mod * ball.yvel * (w - 2 * (paddles[0].x + paddles[0].width + ball.radius)) // abs(
            ball.xvel)
        if pred_right < 0:
            pred_right = -pred_right
        elif pred_right > h:
            pred_right = -pred_right + 2 * h
    else:
        pred_right = ball.y + ball.yvel * relx_right // abs(ball.xvel)
        mod = 1
        if pred_right < 0:
            pred_right = -pred_right
            mod = -1
        elif pred_right > h:
            pred_right = -pred_right + 2 * h
            mod = -1
        pred_left = pred_right + mod * ball.yvel * (w - 2 * (paddles[0].x + paddles[0].width + ball.radius)) // abs(
            ball.xvel)
        if pred_left < 0:
            pred_left = -pred_left
        elif pred_left > h:
            pred_left = -pred_left + 2 * h
    if pred_left > (paddles[0].y + paddles[0].height // 2) and paddles[0].y + paddles[0].height != h:
        rec_left = 1  # recommended to move down
    elif pred_left < (paddles[0].y + paddles[0].height // 2) and paddles[0].y != 0:
        rec_left = -1  # recommeded to move up
    else:
        rec_left = 0
    if pred_right > (paddles[1].y + paddles[1].height // 2) and paddles[1].y + paddles[1].height != h:
        rec_right = 1  # recommended to move down
    elif pred_right < (paddles[1].y + paddles[1].height // 2) and paddles[1].y != 0:
        rec_right = -1  # recommeded to move up
    else:
        rec_right = 0
    # input states for left and right agents
    state_left = [rec_left, (paddles[1].y + paddles[1].height // 2) / h,
                  pred_right / h, relx_left / w, ball.yvel / ball.softcap]
    state_right = [rec_right, (paddles[0].y + paddles[0].height // 2) / h,
                   pred_left / h, relx_right / w, ball.yvel / ball.softcap]
    return state_left, state_right


def main():
    win = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Pong")
    font = pygame.font.SysFont("comicsans", 30)
    win_font = pygame.font.SysFont("comicsans", 50)
    clock = pygame.time.Clock()
    running = True
    paddles = [Paddle(8), Paddle(w - 24)]
    ball = Ball()
    game_num = 1
    if filepath:
        with open((filepath + '_config.pkl'), 'rb') as f:
            exp_rate, iters, game_num = pickle.load(f)
        agent = DQN(exploration_rate=exp_rate, iterations=iters, filepath=(filepath + '.ai'))
    else:
        agent = DQN()
    dead_ball = 0
    score2, score1 = 0, 0
    reward1, reward2 = 0, 0
    draw_window(win, paddles, ball, score2, score1, True, game_num=game_num)
    pygame.time.delay(1000)
    tapped = 0  # for detecting when a paddle was not able to block the initial random ball
    start_ticks = pygame.time.get_ticks()  # to prevent infinite loop, break after 20 seconds of no scoring
    save_counter = 0
    while running:
        clock.tick(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        reward1, reward2 = 0, 0  # resetting rewards at the beginning of each "experience"
        state_left, state_right = get_states(paddles, ball)  # gets the state of the game from each paddle's perspective
        # defining movement
        for paddle in paddles:
            paddle.currvel = 0
            paddle.keys['up'] = False
            paddle.keys['down'] = False
        left_choice = agent.get_next_action(state_left)
        right_choice = agent.get_next_action(state_right)
        if right_choice == 0:
            paddles[1].keys['up'] = True
        elif right_choice == 2:
            paddles[1].keys['down'] = True
        if left_choice == 0:
            paddles[0].keys['up'] = True
        elif left_choice == 2:
            paddles[0].keys['down'] = True
        for paddle in paddles:
            paddle.move()
        dead_ball = ball.move(paddles)
        new_state_left, new_state_right = get_states(paddles, ball)  # gets the new state after actions are taken

        # figuring out if the ball just hit one of the paddles
        if paddles[1].x - ball.radius - abs(ball.xvel) - 1 < ball.x < paddles[1].x - ball.radius - 1 and ball.xvel < 0:
            # hit the right paddle, above is true for exactly one frame each time
            tapped += 1
            # for the first 40 games, reward saving the ball to improve defence
            if game_num <= 40:
                reward1 = 50
            if new_state_right[2] * h > paddles[0].y + paddles[0].height or new_state_right[2] * h < paddles[0].y:
                #ball is headed away from opponent
                reward1 += 100
        elif paddles[0].x + paddles[0].width + ball.radius + 1 < ball.x < paddles[0].x + paddles[0].width + ball.radius + abs(ball.xvel) + 1 and ball.xvel > 0:
            # hit the left paddle, above is true for exactly one frame each time
            tapped += 1
            # for the first 20 games, reward saving the ball to improve defence
            if game_num <= 20:
                reward2 = 50
            if new_state_left[2] * h > paddles[1].y + paddles[1].height or new_state_left[2] * h < paddles[1].y:
                #ball is headed away from opponent
                reward2 += 100

        # for when ball is beyond saving i.e. point is scored and ball is "dead"
        if dead_ball:
            # assigns point
            if ball.x > w // 2:
                score2 += 1
                reward1 = -(abs(ball.y - (paddles[1].y + paddles[1].height // 2)))  # a1 can't save the ball
                # the negative feedback depends on how far away the ball was from the paddle, but is at least -50
                if tapped:  # reward scorer only if they actually played a role
                    if game_num <= 40:
                        reward2 = 100
                    else:  # increases reward for being offensive later, after it learns to defend
                        reward2 = 200
            else:
                score1 += 1
                reward2 = -(abs(ball.y - (paddles[0].y + paddles[0].height // 2)))  # a2 can't save the ball
                if tapped:  # reward scorer only if they actually played a role
                    if game_num <= 40:
                        reward1 = 100
                    else:  # increases reward for being offensive later, after it learns to defend
                        reward1 = 200
            # resets score and updates game number if 10 is reached by either side
            if score1 == 10:
                score1 = 0
                score2 = 0
                game_num += 1
            elif score2 == 10:
                score1 = 0
                score2 = 0
                game_num += 1
            dead_ball = 0  # resetting dead_ball just in case, to prevent double counting from occuring
            tapped = 0
            # below line reset the ball and paddles regardless of game situation
            # DEV NOTE FOR THE FUTURE: THIS IS THE ONLY BALL AND PADDLE RESET NORMALLY USED, COME HERE TO FIX BUGS
            del ball
            del paddles[0:2]
            paddles.extend([Paddle(8), Paddle(w - 24)])
            ball = Ball()
            draw_window(win, paddles, ball, score2, score1, True, game_num=game_num)
            # resets start_ticks, which is used to detect infinite loops
            # print((pygame.time.get_ticks() - start_ticks)/1000)  # to check how fast the game is actually running
            start_ticks = pygame.time.get_ticks()
        # in case of an infinite loop
        elif (pygame.time.get_ticks() - start_ticks) / 1000 > 150:
            # update the above number (150) accordingly based on how fast/slow your GPU can run this and how long to wait
            #before declaring a rally an infinite loop
            reward1, reward2 = -200, -200  # for getting into an infinite loop/taking too long to score
            print("Infinite loop occurred after", tapped, "bounces")
            dead_ball = 0
            tapped = 0
            # reset ball and paddles
            del ball
            del paddles[0:2]
            paddles.extend([Paddle(8), Paddle(w - 24)])
            ball = Ball()
            draw_window(win, paddles, ball, score2, score1, True, game_num=game_num)
            start_ticks = pygame.time.get_ticks()  # reset start_ticks
        # normal, uneventful frame - movements of the entire loop iteration are finally rendered
        else:
            draw_window(win, paddles, ball, score2, score1, game_num=game_num)

        print(reward2, reward1, tapped)  # for debugging
        # update the agent for both paddle's experiences at the end of each iteration
        agent.update(state_left, new_state_left, left_choice, reward2)
        agent.update(state_right, new_state_right, right_choice, reward1)

        if game_num % 10 == 0 and save_counter == 0:  # saves the model every 10 games
            agent.model.save(filepath=('game_' + str(game_num) + '_model.ai'))
            with open(('game_' + str(game_num) + '_model_config.pkl'), 'wb') as f:
                pickle.dump([agent.exploration_rate, agent.iterations, game_num], f)
            print("Successfully saved after", game_num, "games")
            save_counter = 1
        elif game_num % 10 != 0 and save_counter != 0:  # resets the counter allowing the game to be saved again
            save_counter = 0

    pygame.quit()  # ends pygame instance before quitting the program


if __name__ == "__main__":
    main()
