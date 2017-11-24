## Survey of Reinforcement Learning Techniques Using OpenAI Gym

#### Motivation

In recent years, the research interest of reinforcement learning (RL) has increased significantly, partly due to the shocking breakthrough of AlphaGo by DeepMind.  The first generation of AlphaGo was able to compete and win a best-of-five series against
the best human players in 2016 – A feat previously thought impossible for at least another decade.

In late 2017, AlphaGo Zero was published.  A notable difference is that it was trained with deep RL without any kind of human guidance. Comparing the two generations of AlphaGo head-to-head, Zero dominated the previous generation 100-0 – An accomplishment that speaks to the ability of deep reinforcement learning.


#### Setting Up Roboschool

I'm using Ubuntu 17.10 and Anaconda Python 3.6.  Instructions for setting up Roboschool can be found at https://github.com/openai/roboschool.  I've installed (using `pip`) `gym`, `agents`, `pybullet`, and `PyOpenGL` previously.  The only deviation from the Roboschool instruction is that I needed to run the Anaconda specific steps (listed under "Mac, Anaconda with Python 3") before the last step.

    > conda install qt
    > export PKG_CONFIG_PATH=$(dirname $(dirname $(which python)))/lib/pkgconfig


#### Setting Up TensorFlow with GPU

Tensorflow's setup instruction is currently (at the time of this writing, TensorFlow v1.4.0) using an older version of CUDA and cuDNN.  By default, Ubuntu 17.10 will download CUDA 9 and cuDNN 7.

> All our prebuilt binaries have been built with CUDA 8 and cuDNN 6. We anticipate releasing TensorFlow 1.5 with CUDA 9 and cuDNN 7.

In order to use these specific versions, I needed to compile Tensorflow from source, the instructions can be found at https://www.tensorflow.org/install/install_sources.  Make sure to install all the dependencies first, then compiling from source should be straight-forward.


