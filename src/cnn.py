# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, choice

import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

OBS_SIZE = 10
TRACK_WIDTH = 3
TRACK_LENGTH = 10
HEIGHT = 1
LOG_FREQUENCY = 10
REPLAY_SIZE = 10000
BATCH_SIZE = 30
GAMMA = 0.99
PATTERNS = []


def get_mission_xml():
    my_xml = ''
    for x in range(3):
        for z in range(TRACK_LENGTH):
            my_xml += "<DrawBlock x='{}' y='1' z='{}' type='gold_block'/>\n".format(x, z)
    my_xml = my_xml[:-1]

    pattern = PATTERNS[np.random.randint(0, 80)][:10]

    for z in range(TRACK_LENGTH):
        for x in range(3):
            if pattern[z][x]:
                my_xml += "<DrawBlock x='{}' y='2' z='{}' type='emerald_block'/>\n".format(x, z)
                my_xml += "<DrawBlock x='{}' y='3' z='{}' type='emerald_block'/>\n".format(x, z)

    my_xml = my_xml[:-1]

    my_rewards = ''
    for z in range(len(pattern)):
        for x in range(3):
            if not pattern[z][x]:
                my_rewards += '''<Marker x='{}' y='{}' z='{}' reward='2' tolerance='1'/>\n'''.format(x+0.5, 2, z+0.5)

    for x in range(3):
        my_rewards += '''<Marker x='{}' y='{}' z='{}' reward='30' tolerance='1'/>\n'''.format(x+0.5, 2, TRACK_LENGTH+0.5)


    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>Minecraft Surfer</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>
                        <DrawingDecorator>''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-100, 100, -100, 100) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='3' y2='2' z1='{}' z2='{}' type='air'/>".format(-100, 100, -100, 100) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='grass'/>".format(-100, 100, -100, 100) + \
                            my_xml + \
                            '''<DrawBlock x='0'  y='2' z='0' type='air' />
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>CS175MinecraftSurfer</Name>
                    <AgentStart>
                        <Placement x="1.5" y="2" z="0.5" pitch="0" yaw="0"/>
                    </AgentStart>
                    <AgentHandlers>
                        <RewardForTouchingBlockType>
                            <Block type='emerald_block' reward='-15' />
                        </RewardForTouchingBlockType>
                        <RewardForReachingPosition>''' + \
                        my_rewards + \
                        '''</RewardForReachingPosition>
                        <ContinuousMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <ObservationFromGrid>
                            <Grid name="floorAll" absoluteCoords="true">
                                <min x="0" y="2" z="0"/>
                                <max x="2" y="2" z="20"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total='30'/>
                        <AgentQuitFromTouchingBlockType>
                            <Block type="emerald_block" />
                        </AgentQuitFromTouchingBlockType>
                        <AgentQuitFromReachingPosition>''' + \
                            '''<Marker x="0.5" y="2" z="{}" tolerance="1"/>
                            <Marker x="1.5" y="2" z="{}" tolerance="1"/>
                            <Marker x="2.5" y="2" z="{}" tolerance="1"/>'''.format(TRACK_LENGTH + 0.5, TRACK_LENGTH + 0.5, TRACK_LENGTH + 0.5) + \
                        '''</AgentQuitFromReachingPosition>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''

def init_malmo(agent_host):
    max_retries = 3
    for retry in range(max_retries):
        try:
            my_mission = MalmoPython.MissionSpec(get_mission_xml(), True)
            my_mission_record = MalmoPython.MissionRecordSpec()
            my_mission.requestVideo(800, 500)
            my_mission.setViewpoint(1)

            my_clients = MalmoPython.ClientPool()
            my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'MinecraftSurfer' )
            break
        except RuntimeError as e:
            print(retry)
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("\nError:", error.text)

    return world_state

def build_model(action_dict):
    model = Sequential()
    model.add(Input(shape=(OBS_SIZE * TRACK_WIDTH + 1)))
    model.add(Dense(20))
    model.add(Dense(10))
    model.add(Dense(len(action_dict)))
    print(model.summary())

    optimizer = Adam(learning_rate=0.001)
    loss_function = MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss_function)
    return model, optimizer, loss_function

def get_observation(agent_host, world_state):
    obs = np.zeros((OBS_SIZE * TRACK_WIDTH * HEIGHT + 1, ))

    allow_left = False
    allow_right = False
    ZPos = None

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            # Get observation
            
            try:
                grid = observations['floorAll']
            except:
                print("FLOORALL ERROR FUCK OUR LIVES")
                continue

            zPos = int(observations['ZPos'])
            for i, z in enumerate(range(zPos * 3, (zPos + 10) * 3)):
                obs[i] = grid[z] == 'emerald_block'
            obs[-1] = observations['XPos']

            allow_left = observations['XPos'] < 2
            allow_right = observations['XPos'] > 1
            ZPos = observations["ZPos"]
            
            break

    return obs, allow_left, allow_right, ZPos

def log_returns():
    plt.clf()
    plt.plot([i for i in range(len(zPositions))], zPositions)
    plt.title('Z Positions')
    plt.ylabel('Z Position')
    plt.xlabel('Steps')
    plt.savefig('zPositions.png')


    box = np.ones(LOG_FREQUENCY) / LOG_FREQUENCY
    returns_smooth = np.convolve(returns[1:], box, mode='same')
    plt.clf()
    plt.plot(steps[1:], returns_smooth)
    plt.title('Rewards')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for step, value in zip(steps[1:], returns[1:]):
            f.write("{}\t{}\n".format(step, value))

def train(episode_states, model, optimizer, loss_function):

    #Change below
    print("***********TRAINING************\n")
    episode_states = np.array(episode_states)
    count = 0
    while count < 30:
        sample = choice(list(range(len(episode_states))), BATCH_SIZE)
        sample = episode_states[sample]

        state_t = []
        action_t = []
        reward_t = []
        state_t1 = []

        for s_t, a_t, r_t, s_t1 in sample:
            state_t.append(s_t)
            action_t.append(a_t)
            reward_t.append(r_t)
            state_t1.append(s_t1)

        state_t = np.array(state_t).flatten()
        action_t = np.array(action_t).flatten()
        reward_t = np.array(reward_t).flatten()
        state_t1 = np.array(state_t1).flatten()

        Q_sa = model.predict(state_t1, batch_size=BATCH_SIZE, steps=1)
        targets = model.predict(state_t, batch_size=BATCH_SIZE, steps=1)
        targets[range(BATCH_SIZE), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)

        count += 1

        model.train_on_batch(state_t, targets)


def run():
    action_dict = {
        0: 'strafe -1.5',
        1: 'strafe 0',
        2: 'strafe 1.5',
        # 3: 'crouch 1',
        # 4: 'jump 1'
    }

    # Malmo Parameters
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)

    episode_steps = []
    episode_states = []
    returns = []

    model, optimizer, loss_function = build_model(action_dict)

    episode_step = 0
    epsilon = 1.0
    while episode_step < 999999:
        allow_left = True
        allow_right = True
        curZPos = 0
        moving = False
        index = 1
        
        zPositions = []
        # zPositions.append()

        episode_return = 0

        world_state = init_malmo(agent_host)
        obs, allow_left, allow_right, zPos = get_observation(agent_host, world_state)
        
        step = 0
        done = False
        while not done:
            if not moving:
                agent_host.sendCommand("move 0.5")
                time.sleep(.2)


            # Saving each state, reward and next state
            if len(episode_states) >= REPLAY_SIZE:
                episode_states.pop(0)
            episode_states.append([obs, index, sum([r.getValue() for r in world_state.rewards])])

            if np.random.rand() < epsilon:
                index = np.random.randint(0,3)
                command = action_dict[index]
            else:
                tf_obs = tf.convert_to_tensor(obs)
                expanded_obs = tf.expand_dims(tf_obs, -1)
                expanded_obs = tf.expand_dims(expanded_obs, 0)
                q = model(expanded_obs, training=False)
                index = np.argmax(q[0].numpy())
                command = action_dict[index]

            if ((command == 'strafe -1' and allow_left) or (command == 'strafe 1' and allow_right)):
                agent_host.sendCommand(command)
                time.sleep(.2)
                agent_host.sendCommand("strafe 0")
                time.sleep(.2)

            # if (command == "jump 1"):
            #     self.agent_host.sendCommand(command)
            #     time.sleep(.3)
            #     self.agent_host.sendCommand("jump 0")
            #     time.sleep(.2)

            # if (command == "crouch 1"):
            #     self.agent_host.sendCommand(command)
            #     time.sleep(.3)
            #     self.agent_host.sendCommand("crouch 0")
            #     time.sleep(.2)

            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            obs, allow_left, allow_right, zPos = get_observation(agent_host, world_state)
            episode_states[-1].append(obs)

            if zPos:
                curZPos = zPos

            done = not world_state.is_mission_running

            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            step += 1
        

        if len(episode_states) > BATCH_SIZE :
            train(episode_states, model, optimizer, loss_function)
            epsilon *= 0.99

        zPositions.append(curZPos)
        if len(returns) > LOG_FREQUENCY + 1 and \
            len(returns) % LOG_FREQUENCY == 0:
            log_returns()

        episode_steps.append(episode_step)
        obs, allow_left, allow_right, curZPos = get_observation(agent_host, world_state)


if __name__ == '__main__':
    ray.init()
    tf.compat.v1.enable_eager_execution()
    with open('patterns.txt', 'r') as file:
        for line in file:
            PATTERNS.append(eval(line))
    run()
