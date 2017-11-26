from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'action_bound'])

games = {}

robo_reacher = Game(env_name='RoboschoolReacher-v1',
    input_size=9,
    output_size=2,
    time_factor=1000,
    action_bound=[-1., 1.]
)
games['robo_reacher'] = robo_reacher

robo_double_pendulum = Game(env_name='RoboschoolInvertedDoublePendulum-v1',
    input_size=9,
    output_size=1,
    time_factor=0,
    action_bound=[-1., 1.]
)
games['robo_double_pendulum'] = robo_double_pendulum

robo_pendulum = Game(env_name='RoboschoolInvertedPendulumSwingup-v1',
    input_size=5,
    output_size=1,
    time_factor=1000,
    action_bound=[-1., 1.]
)
games['robo_pendulum'] = robo_pendulum

robo_humanoid = Game(env_name='RoboschoolHumanoid-v1',
    input_size=44,
    output_size=17,
    time_factor=1000,
    action_bound=[-1., 1.]
)
games['robo_humanoid'] = robo_humanoid

robo_walker= Game(env_name='RoboschoolWalker2d-v1',
    input_size=22,
    output_size=6,
    time_factor=1000,
    action_bound=[-1., 1.]
)
games['robo_walker'] = robo_walker