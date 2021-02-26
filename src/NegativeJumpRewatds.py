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

patterns = []
with open('patterns.txt', 'r') as file:
    for line in file:
        patterns.append(eval(line))

class MinecraftSurfer(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 10
        self.track_width = 3
        self.track_length = 10
        self.height = 4
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'strafe -1',
            1: 'strafe 0',
            2: 'strafe 1'
        }

        # Rllib Parameters
        self.action_space = Box(-1, 1, shape=(2, ), dtype=np.float32)
        self.observation_space = Box(0, 3, (self.obs_size * self.track_width * self.height + 3, ), dtype=np.float32)

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

        self.moving = False

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

        action[1] = 0 if action[1] < 0 else 1

        if not self.moving:
            self.agent_host.sendCommand("move 0.5")
            time.sleep(.2)
            self.moving = True

        # Get Action
        command = "strafe " + str(action[0])
        if ((action[0] < 0 and self.allow_left) or (action[0] > 0 and self.allow_right)):
            self.agent_host.sendCommand(command)
            time.sleep(.2)
        self.agent_host.sendCommand("strafe 0")
        time.sleep(.1)

        if action[1]:
            self.agent_host.sendCommand("jump 1")
            time.sleep(.2)
            self.agent_host.sendCommand("jump 0")

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

        pattern = []
        for ind in choice([i for i in range(len(patterns))], size=4):
            pattern += patterns[ind]

        my_rewards = ''
        for z in range(self.track_length):
            for x in range(3):
                if pattern[z][x] == 1:
                    my_xml += "<DrawBlock x='{}' y='2' z='{}' type='emerald_block'/>\n".format(x, z)
                    my_xml += "<DrawBlock x='{}' y='3' z='{}' type='emerald_block'/>\n".format(x, z)
                elif pattern[z][x] == 2:
                    my_xml += "<DrawBlock x='{}' y='1' z='{}' type='air'/>\n".format(x, z)
                    my_rewards += '''<Marker x='{}' y='2' z='{}' reward='10' tolerance='.5'/>\n'''.format(x+0.5, z+.5)
                # elif pattern[z][x] == 3:
                #     my_xml += "<DrawBlock x='{}' y='4' z='{}' type='emerald_block'/>\n".format(x, z)

        my_xml = my_xml[:-1]

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
                                "<DrawCuboid x1='{}' x2='{}' y1='4' y2='1' z1='{}' z2='{}' type='air'/>".format(-100, 100, -100, 100) + \
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
                                <Block type='bedrock' reward='-30' />
                            </RewardForTouchingBlockType>
                            <RewardForReachingPosition>''' + \
                            my_rewards + \
                            '''</RewardForReachingPosition>
                            <RewardForSendingCommand reward='-2' distribution='jump'/>
                            <ContinuousMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromGrid>
                                <Grid name="floorAll" absoluteCoords="true">
                                    <min x="0" y="1" z="0"/>
                                    <max x="2" y="4" z="99"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps * 3)+'''" />
                            <AgentQuitFromTouchingBlockType>
                                <Block type="emerald_block" />
                                <Block type="bedrock" />
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
        max_retries = 10
        for retry in range(max_retries):
            try:
                my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
                my_mission_record = MalmoPython.MissionRecordSpec()
                my_mission.requestVideo(800, 500)
                my_mission.setViewpoint(1)

                my_clients = MalmoPython.ClientPool()
                my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

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
        obs = np.zeros((self.obs_size * self.track_width * self.height + 3, ))

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
                try:
                    grid = observations['floorAll']
                except Exception:
                    continue
                
                ground = grid[:300]
                middle1 = grid[300:600]
                middle2 = grid[600:900]
                top = grid[900:]

                grid_index = int(observations['ZPos']) * 3
                ground = ground[grid_index:grid_index+30]
                top = top[grid_index:grid_index+30]

                middle1 = middle1[grid_index:grid_index+30]
                middle2 = middle2[grid_index:grid_index+30]

                for i, x in enumerate(ground):
                    if x == 'air':
                        obs[i] = 1

                for i, x in enumerate(middle1, i):
                    obs[i] = x == 'emerald_block'

                for i, x in enumerate(middle2, i):
                    obs[i] = x == 'emerald_block'
                
                for i, x in enumerate(top, i):
                    obs[i] = x == 'emerald_block'
                

                allow_left = observations['XPos'] < 2
                allow_right = observations['XPos'] > 1
                ZPos = observations["ZPos"]

                if observations["XPos"] > 3:
                    observations["XPos"] = 3
                if observations["XPos"] < 0:
                    observations["XPos"] = 0

                obs[-3] = observations["XPos"]
                obs[-2] = observations["XPos"] - int(observations["XPos"])
                obs[-1] = ZPos - int(ZPos)
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
        'framework': 'torch',         # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    if len(sys.argv) == 2:
        trainer.load_checkpoint('./model/checkpoint-' + sys.argv[1])

    try:
        while True:
            print(trainer.train())
    except KeyboardInterrupt:
        trainer.save_checkpoint('./model')
