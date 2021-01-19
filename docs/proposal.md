---
layout: default
title:  Proposal
---

## Summary of the Project
The goal of our project is to have an agent successfully reach the end of a track while avoiding obstacles and collecting gold nuggets to maximize its score. This idea is based on the game Subway Surfers. The agent will constantly be moving forward at a speed of 1 block/second. The agent will also encounter three types of obstacles which can be avoided by either moving 1 block left or right (3 block wide path), crouching, or jumping. The agent will use its view to detect these obstacles and choose the correct action. The agent has two lives. If the agent runs into an obstacle, it will lose a life and after two hits, it will die :(

## AI/ML Algorithms
We will use object detection and deep reinforcement learning with neural networks for this project.

## Evaluation Plan
For the quantitative evaluation, we will be using the distance traveled, the number of gold nuggets collected, and the number of obstacles avoided as our metrics. The agent will receive a higher score for traveling farther, collecting more gold nuggets, and avoiding more obstacles. For example, an agent would receive 1 point for every block traveled, 2 points for every gold nugget collected, and 4 points for every obstacle avoided. However, for every obstacle the agent hits, the agent will lose a life and will lose 30 points. Our baseline for success is that the agent should be able to avoid at least one of every type of obstacle in the track. On the other hand, our baseline for failure is that the agent is unable to avoid a single obstacle in the track. 

To verify that our project works, we will provide a function which will be able to take an input image and label all obstacles with their types as well as all gold nuggets. For our approach, the sanity cases will be if the agent can detect an obstacle or gold nugget and avoid obstacles and/or collect gold nuggets. Our moonshot case is if the agent is able to successfully make it to the end of the track while maximizing its score by collecting gold nuggets.

## Appointment with the Instructor
3:00pm - 3:15pm, Wednesday, January 20, 2021
