import sys

from cartpole_dqn.agent import CartPoleDQNAgent
from cartpole_dqn.trainer import CartPoleDQNTrainer
from cartpole_dqn.runner import CartPoleRunner


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
IMAGE_SIZE = 80


def print_usage():
    print("Usage: python cartpole_v1.py [train | run] [episodes (train) | {pretrained model file} (run)]")


def main(argv):
    if len(argv) < 2:
        print_usage()
        exit()

    mode = argv[0]
    if mode not in ['train', 'run']:
        print_usage()

    # TODO Separate agent and environment?
    # Also, move device and rng generator out of agent
    agent = CartPoleDQNAgent(IMAGE_SIZE)

    if mode == 'train':
        episodes = int(argv[1])
        print(f"Training on {agent.device}")
        trainer = CartPoleDQNTrainer(
            agent,
            agent.device,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
            target_update=TARGET_UPDATE,
            w=IMAGE_SIZE,
            h=IMAGE_SIZE,
            action_space=agent.action_space,
            memory=MEMORY_SIZE
        )
        trainer.set_state_file(f'CartPoleDQN_{episodes}ep.pt')
        trainer.run(episodes, plot=True)

    elif mode == 'run':
        model_file = argv[1]
        agent.load_net(model_file)

        print(f"Running on {agent.device}")
        runner = CartPoleRunner(agent)
        runner.run()

    agent.render()
    agent.close()


if __name__ == "__main__":
    main(sys.argv[1:])
