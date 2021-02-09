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


class MinecraftSurfer(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 10
        self.track_width = 3
        self.track_length = 10
        self.height = 1
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'strafe -1',
            1: 'strafe 0',
            2: 'strafe 1',
            3: 'crouch 1',
            4: 'jump 1'
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, (self.obs_size * self.track_width * self.height, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # MinecraftSurfer Parameters
        self.obs = None
        self.allow_left = False
        self.allow_right = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.zPositions = []
        self.curZPos = 0
        self.moving = False

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """

        self.zPositions.append(self.curZPos)

        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.allow_left, self.allow_right, self.curZPos = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        if not self.moving:
            self.agent_host.sendCommand("move 0.5")
            time.sleep(.2)

        # Get Action
        command = self.action_dict[action]
        if ((command == 'strafe -1' and self.allow_left) or (command == 'strafe 1' and self.allow_right)):
            self.agent_host.sendCommand(command)
            time.sleep(.2)
            self.agent_host.sendCommand("strafe 0")
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

        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_left, self.allow_right, curZPos = self.get_observation(world_state)
        if (curZPos):
            self.curZPos = curZPos

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
            print(reward)
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def get_mission_xml(self):
        

        my_xml = ''
        for x in range(3):
            for z in range(0, self.track_length):
                my_xml += "<DrawBlock x='{}' y='1' z='{}' type='gold_block'/>\n".format(x, z)
        my_xml = my_xml[:-1]

        patterns = []
        with open('patterns.txt', 'r') as file:
            for line in file:
                patterns.append(eval(line))

        pattern = []
        for ind in choice([i for i in range(80)], size=4):
            pattern += patterns[ind]

        for z in range(self.track_length):
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
            my_rewards += '''<Marker x='{}' y='{}' z='{}' reward='30' tolerance='1'/>\n'''.format(x+0.5, 2, self.track_length+0.5)


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
                                <Grid name="floorAll">
                                    <min x="0" y="2" z="0"/>
                                    <max x="2" y="2" z="9"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps * 3)+'''" />
                            <AgentQuitFromTouchingBlockType>
                                <Block type="emerald_block" />
                            </AgentQuitFromTouchingBlockType>
                            <AgentQuitFromReachingPosition>''' + \
                                '''<Marker x="0.5" y="2" z="{}" tolerance="1"/>
                                <Marker x="1.5" y="2" z="{}" tolerance="1"/>
                                <Marker x="2.5" y="2" z="{}" tolerance="1"/>'''.format(self.track_length + 0.5, self.track_length + 0.5, self.track_length + 0.5) + \
                            '''</AgentQuitFromReachingPosition>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'MinecraftSurfer' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        obs = np.zeros((self.obs_size * self.track_width * self.height, ))

        allow_left = False
        allow_right = False
        ZPos = None

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                # Get observation
                grid = observations['floorAll']

                for i, x in enumerate(grid):
                    obs[i] = x == 'emerald_block'

                allow_left = observations['XPos'] < 2
                allow_right = observations['XPos'] > 1
                ZPos = observations["ZPos"]
                
                break

        return obs, allow_left, allow_right, ZPos

    def log_returns(self):
        plt.clf()
        plt.plot([i for i in range(len(self.zPositions))], self.zPositions)
        plt.title('Z Positions')
        plt.ylabel('Z Position')
        plt.xlabel('Steps')
        plt.savefig('zPositions.png')


        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Rewards')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=MinecraftSurfer, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
