import sys
import click

@click.group()
def main():
    pass

@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def train(args):
    from base_net.train_model import main as train_main
    sys.argv = ['train'] + list(args)
    train_main()

@main.command(context_settings=dict(ignore_unknown_options=True), add_help_option=False)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def calculate_ground_truth(args):
    from base_net.scripts.calculate_ground_truth import main as ground_truth_main
    sys.argv = ['calculate_ground_truth'] + list(args)
    ground_truth_main()

@main.command(context_settings=dict(ignore_unknown_options=True), add_help_option=False)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def generate_task_poses(args):
    from base_net.scripts.generate_task_poses import main as generate_task_poses_main
    sys.argv = ['generate_task_poses'] + list(args)
    generate_task_poses_main()

if __name__ == '__main__':
    main()