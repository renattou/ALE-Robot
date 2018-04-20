# ALE-Robot

## Description

ALE-Robot is an architecture that uses a DDQN (Double Deep Q-Network) in order for an agent to learn how to play Atari games through a simulated robot.
ALE (Arcade Learning Environment) is used as the emulator and V-REP as the robot simulator.
A robot-free implementation is also available.

The agent uses the robot's camera in order to see the game screen, which is being projected on the simulator.
For executing an action, the agent positions the robot's joints so that it interacts with a game controller.
This controller then sends signals back to the agent when its being pressed and then the actions from the controller are  performed on the emulator.

Initial results were unsatisfactory, but more testing needs to done.
More details about the implementation, experiments and results can be seen on the [report](https://www.ic.unicamp.br/~reltech/PFG/2017/PFG-17-18.pdf).

## Screenshots

<table>
  <tr>
    <td><img src="figures/controller.png?raw=true" width="200"></td>
    <td><img src="figures/nao.png?raw=true" width="200"></td>
    <td><img src="figures/scene.png?raw=true" width="400"></td>
  </tr>
</table>

## Requirements
Note the previous version of these softwares may work, but weren't tested.
* Python 2.7 or 3.5
* [Keras](https://keras.io/) >= 2.1.1
* [Tensorflow](https://www.tensorflow.org/) >= 1.4.0
* [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment/) >= 0.5
* [V-REP](http://www.coppeliarobotics.com/) >= 3.4.0
* [OpenCV](https://pypi.python.org/pypi/opencv-python) >= 3.3.0

For increased performance on communication with V-REP, it's strongly suggested to use shared memory.
It can be installed from:

* [vrep_remote_api_shm](https://github.com/BenjaminNavarro/vrep_remote_api_shm)

If you want to save or load network weights, you also need:
* [h5py](http://www.h5py.org/)

There's also a 2-player version of Pong (agent vs user) which is not fully tested, but you can use.
If you want to do so, you also need:
* [PyGame](http://www.pygame.org/)
* [Custom version of ALE](https://github.com/renattou/Arcade-Learning-Environment)

## Running

You can use the .sh files to run both training and playing.

Train without robot: `./train_ale.sh game args`

Play without robot: `./play_ale.sh game args`

Train with robot: `./train.sh game vrep_path args`

Play with robot: `./play.sh game vrep_path args`

Play without robot on 2-player mode: `./play_2p.sh args`

`game` should be a game ROM placed on the `roms` folder.
`vrep_path` should be the path to where `vrep.sh` is localized.
`args` can be multiple arguments.
All possible arguments can be seen on `src/utils.py`.

## Acknowledgments

A lot of the code related to the network and structure of the code were inspired by two other projects: [Simple DQN](https://github.com/tambetm/simple_dqn) and [keras-rl](https://github.com/matthiasplappert/keras-rl).

This is a project I developed for my undergraduate thesis on Computer Science at the [Institute of Computing](https://www.ic.unicamp.br/) at [Unicamp](http://www.unicamp.br) (University of Campinas) on 2017.

## Citing
If you use ALE-Robot, the architecture or the report, on your research, you can cite it as ([bib file](https://www.ic.unicamp.br/~reltech/PFG/2017/PFG-17-18.bib)):
```
@techreport{TR-IC-PFG-17-18,
   number = {IC-PFG-17-18},
   author = {Renato Landim Vargas and Esther Luna Colombini},
   title = {{Atari-playing Robot}},
   month = {December},
   year = {2017},
   institution = {Institute of Computing, University of Campinas}
}
```
