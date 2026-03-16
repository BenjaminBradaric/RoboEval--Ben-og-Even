from roboeval.action_modes import JointPositionActionMode
from roboeval.demonstrations.demo_player import DemoPlayer
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata
from roboeval.envs.stack_books import PickSingleBookFromTablePositionAndOrientation
from roboeval.robots.configs.panda import BimanualPanda, SinglePanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.stack_books import StackSingleBookShelf
#BANE 
#~/.roboeval/roboeval_demos/1.0.0/BimanualPanda/LiftPot/JointPositionActionMode_floating_absolute_joint

# Create environment with camera observations
env = StackSingleBookShelf(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[]),
    render_mode="human", #rgb_array
    control_frequency=20,
    robot_cls=BimanualPanda,
    observation_config= ObservationConfig(
    cameras=[
        CameraConfig(
            name="head",
            rgb=True,
            depth=False,
            resolution=(96, 96)
        ),
        CameraConfig(
            name="left_wrist",
            rgb=True,
            depth=False,
            resolution=(96, 96)
        ),
        CameraConfig(
            name="right_wrist",
            rgb=True,
            depth=False,
            resolution=(96, 96)
        )
    ],
    proprioception=True,
    privileged_information=False
)
)

#print(env.config.camera.keys())


#Get demonstrations from DemoStore
metadata = Metadata.from_env(env)
demos = DemoStore().get_demos(metadata, amount=100, frequency=20)

# Replay demonstrations
for demo in demos:
    DemoPlayer().replay_in_env(demo, env, demo_frequency=20)