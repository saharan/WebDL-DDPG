WebDL-DDPG
---

A library for deep reinforcement learning. The article about this library [is here](http://el-ement.com/blog/2018/06/30/full-ddpg/) (Japanese).

## Features
* Supports GPU (WebGL for JavaScript platform and CUDA for Python platform)
* DDPG (Deep Deterministic Policy Gradient) is implemented

## Main classes
* [src/RobotDDPGTest.hx](./src/RobotDDPGTest.hx) for playing a learned model of bipedal walking robot (JS)
* [src/RobotDDPGCudaTest.hx](./src/RobotDDPGCudaTest.hx) for learning and exporting models of bipedal walking robot (Python)
* [src/PendulumTest.hx](./src/PendulumTest.hx) for learning and playing an inverted pendulum swingup problem (JS)

## License
The MIT License
