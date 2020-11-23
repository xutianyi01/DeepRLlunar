# Lab2: lunar lander-v2
## by Tianyi Xu
I use Monte-Carlo policy gradient: REINFORCE to solve this problem. Baseline is n-step Sarsa with linear function approximation. The Google Colab link is https://colab.research.google.com/drive/1ur7xOx6DeNWrhOkDGYhg5oDUlNu9zQL5?usp=sharing , you can find the final result figure of both algorithms in the end of the output module. 
The detailed descriptions of both algorithms are as follows:
### a.
For n-step Sarsa with linear function approximation, we can discretize states and set feature vectors by printing and observing the environment's state values. Details are as follows:

For horizontal and vertical position, horizontal and vertical velocity, angle and angular velocity, I discretize [-1, 1] into 5 intervals of each one. For each component of states, I give each interval of this component a bucket_index s'
