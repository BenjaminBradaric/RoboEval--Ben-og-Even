"""Collect new demonstrations for PackBoxPosition and add them to the DemoStore.

Controls:
    Left Arm:
        A/D  - Left/Right (X-axis)
        Z/C  - Forward/Backward (Y-axis)
        W/S  - Up/Down (Z-axis)
        V    - Toggle left gripper

    Right Arm:
        J/L  - Left/Right (X-axis)
        U/O  - Forward/Backward (Y-axis)
        I/K  - Up/Down (Z-axis)
        B    - Toggle right gripper

    Session:
        R    - Start new recording (resets env)
        X    - Save current recording
        T    - Toggle position / orientation control
        G    - Toggle gripper mode (autoclose vs hold)
        ESC  - Exit and cache all saved demos
"""

import tempfile
from pathlib import Path

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.pack_objects import PackBoxPosition
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata
from roboeval.data_collection.keyboard_input import KeyboardTeleop


ACTION_MODE = JointPositionActionMode(
    floating_base=True,
    absolute=True,
    floating_dofs=[],
)


def collect_demos(demo_dir: Path) -> int:
    """Run keyboard teleoperation and return number of saved demos."""
    print("\n" + "=" * 60)
    print("PackBoxPosition  —  Keyboard Demo Collection")
    print("=" * 60)
    print("R  : Start recording  (resets the environment)")
    print("X  : Save demo")
    print("T  : Toggle position / orientation mode")
    print("G  : Toggle gripper mode")
    print("ESC: Exit")
    print("=" * 60 + "\n")

    teleop = KeyboardTeleop(
        env_cls=PackBoxPosition,
        action_mode=ACTION_MODE,
        resolution=(900, 1000),
        demo_directory=demo_dir,
        robot_cls=BimanualPanda,
        config={"env": "PackBoxPosition", "robot": "Bimanual Panda"},
    )

    try:
        teleop.run()
    except KeyboardInterrupt:
        print("\nSession interrupted.")

    saved = list(demo_dir.glob("*.safetensors"))
    print(f"\nSaved {len(saved)} demo(s) to {demo_dir}")
    return len(saved)


def cache_new_demos(demo_dir: Path):
    """Load every .safetensors file from demo_dir and add it to the DemoStore."""
    files = list(demo_dir.glob("*.safetensors"))
    if not files:
        print("No demos to cache.")
        return

    store = DemoStore()
    cached = 0
    for path in files:
        try:
            demo = Demo.from_safetensors(path)
            store.cache_demo(demo)
            cached += 1
            print(f"  Cached: {path.name}")
        except Exception as exc:
            print(f"  Failed to cache {path.name}: {exc}")

    print(f"\nAdded {cached}/{len(files)} new demo(s) to the DemoStore.")


def report_total_demos():
    """Print how many PackBoxPosition lightweight demos are now in the store."""
    env = PackBoxPosition(
        action_mode=ACTION_MODE,
        render_mode=None,
        control_frequency=20,
        robot_cls=BimanualPanda,
    )
    metadata = Metadata.from_env(env, is_lightweight=True)
    env.close()

    store = DemoStore()
    paths = store.list_demo_paths(metadata)
    print(f"Total PackBoxPosition lightweight demos in store: {len(paths)}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        demo_dir = Path(tmp)
        n = collect_demos(demo_dir)

        if n > 0:
            print("\nCaching new demos into DemoStore...")
            cache_new_demos(demo_dir)

        report_total_demos()
