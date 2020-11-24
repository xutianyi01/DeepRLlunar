# Lab2: lunar lander-v2
## by Tianyi Xu
I use Monte-Carlo policy gradient: REINFORCE to solve this problem. Baseline is n-step Sarsa with linear function approximation. The Google Colab link is https://colab.research.google.com/drive/1ur7xOx6DeNWrhOkDGYhg5oDUlNu9zQL5?usp=sharing , you can find the final result figure of both algorithms in the end of the output module. 
The detailed descriptions of both algorithms are as follows:
### a.
For n-step Sarsa with linear function approximation, I can discretize states and set feature vectors by printing and observing the environment's state values. Details are as follows:

For horizontal and vertical position, horizontal and vertical velocity, angle and angular velocity, I discretize [-1, 1] into 4 intervals of each one of states. For each component of states, I give boundary value of intervals of this component as a bucket_index <a href="https://www.codecogs.com/eqnedit.php?latex=s'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s'" title="s'" /></a> that is respectively 0, 1, 2, 3, 4.

For left and right leg contact, I keep same as before, because only two possible values 0 and 1.

--------

Linear model is used to estimate the optimum Q-value(<a href="https://www.codecogs.com/eqnedit.php?latex=Q_{opt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{opt}" title="Q_{opt}" /></a>).

<a href="https://www.codecogs.com/eqnedit.php?latex=Q_{opt}(s,a)=w\phi(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{opt}(s,a)=w\phi(s,a)" title="Q_{opt}(s,a)=w\phi(s,a)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w" title="w" /></a> is weight vector and <a href="https://www.codecogs.com/eqnedit.php?latex=\phi(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(s,a)" title="\phi(s,a)" /></a> is a feature vector. Here, I set the original <a href="https://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w" title="w" /></a> is a full zero vector. And we set original  <a href="https://www.codecogs.com/eqnedit.php?latex=\phi(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(s,a)" title="\phi(s,a)" /></a> is a full zero vector for each a new generated state contains 8 components and then based on this state I modified <a href="https://www.codecogs.com/eqnedit.php?latex=\phi(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(s,a)" title="\phi(s,a)" /></a> as this:

For each of horizontal(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=0) and vertical position(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=1), horizontal(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=2) and vertical velocity(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=3), angle(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=4) and angular velocity(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=5),

if <a href="https://www.codecogs.com/eqnedit.php?latex=s\in" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in" title="s\in" /></a> [-1, 1],

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_k=1(round(2(s&plus;1))=s_i',&space;a=a_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_k=1(round(2(s&plus;1))=s_i',&space;a=a_j)" title="\phi_k=1(round(2(s+1))=s_i', a=a_j)" /></a> 

where  <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> is the state value of each component of these 6 states.  <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> is the index of features that depend on the setting of <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> and the index of state component <a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>. For example, in my setting I set <a href="https://www.codecogs.com/eqnedit.php?latex=k=ij&plus;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k=ij&plus;j" title="k=ij+j" /></a>+ 20<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>. Here, <a href="https://www.codecogs.com/eqnedit.php?latex=round()" target="_blank"><img src="https://latex.codecogs.com/gif.latex?round()" title="round()" /></a> is a rounding function that rounds the number to be an integer. <a href="https://www.codecogs.com/eqnedit.php?latex=2(s&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2(s&plus;1)" title="2(s+1)" /></a> makes the state <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> from [-1, 1] scaled to be in [0, 4]. <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> is the index of bucket_index <a href="https://www.codecogs.com/eqnedit.php?latex=s'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s'" title="s'" /></a>. <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> is the index of action.

if <a href="https://www.codecogs.com/eqnedit.php?latex=s\in" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in" title="s\in" /></a> (-inf, 1),

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_k&space;=&space;1(a=a_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_k&space;=&space;1(a=a_j)" title="\phi_k = 1(a=a_j)" /></a> 

where <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a>+ 20<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>, which means the bucket_index of this type of the state is 0.

if <a href="https://www.codecogs.com/eqnedit.php?latex=s\in" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s\in" title="s\in" /></a> (1, -inf),

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_k&space;=&space;1(a=a_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_k&space;=&space;1(a=a_j)" title="\phi_k = 1(a=a_j)" /></a> 

where <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=4i&plus;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?4i&plus;j" title="4i+j" /></a>+ 20<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>, which means the bucket_index of this type of the state is 4.

For each of left(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=6) and right leg contact(<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>=7), 

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_k=1(s=0,a=a_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_k=1(s=0,a=a_j)" title="\phi_k=1(s=0,a=a_j)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\phi_k=1(s=1,&space;a=a_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi_k=1(s=1,&space;a=a_j)" title="\phi_k=1(s=1, a=a_j)" /></a>

where  <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> is equal to <a href="https://www.codecogs.com/eqnedit.php?latex=4i&plus;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?4i&plus;j" title="4i+j" /></a>+ 120 + 8<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>, which means the bucket_index of this type of the state is 4.

Therefore, the dimension of feature vector is 5<a href="https://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\times" title="\times" /></a>*4*<a href="https://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\times" title="\times" /></a>6 + 2<a href="https://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\times" title="\times" /></a>4<a href="https://www.codecogs.com/eqnedit.php?latex=\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\times" title="\times" /></a>2 = 136.

### b.

For Monte-Carlo policy gradient: REINFORCE, 3 layer neural network is chosen.  For the first layer, input is the states, output is 256 dimension neurons with fully connection. Activation function is "relu". For the second layer, output is 256 dimension neurons with fully connection. Activation function is "relu". For the third layer, output is 4 dimension neurons (4 actions). Activation function is "Softmax". Optimizer is Adam. Loss function is  <a href="https://www.codecogs.com/eqnedit.php?latex=-G_tlog\pi(A_t|S_t,\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-G_tlog\pi(A_t|S_t,\theta)" title="-G_tlog\pi(A_t|S_t,\theta)" /></a>, Note that <a href="https://www.codecogs.com/eqnedit.php?latex=log\pi(A_t|S_t,\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log\pi(A_t|S_t,\theta)" title="log\pi(A_t|S_t,\theta)" /></a> is cross entropy of softmaxed action prediction and labeled action.

We don't need to use a memory class because we just keep track of the each episode by using a list and then convert to the states. So the batch size of each episode just this episode we generated.

Step size is 0.005. discount factor is 0.99.

### c.

--------
Hyperparameters for both algorithms
Algorithm|Monte-Carlo policy gradient: REINFORCE  | n-step Sarsa with linear function approximation(n=1000)|
---------|--------- | --------|
Running episode|3000  | 3000 |
Step size|0.005  | 0.03 |
Exploration rate|None  | 0.1 |
Discount factor|0.99  | 0.99 |
Neural network|See __b.__  | None. Only linear function approximation can be seen __a.__|

The results of both algorithms with 3000 episodes can be seen as follows:

![pgvssarsa](https://github.com/xutianyi01/lunarlanderv2/blob/main/pgvssarsa.png)

xlabel is episode number. ylabel is the average reward considers previous 100 episodes at episode _t_. We can find REINFORCE totally win Sarsa after 1000 episodes. After 2500 episodes, REINFORCE can reach 200 average reward, and stably solve the  "lunar lander-v2" problem.
