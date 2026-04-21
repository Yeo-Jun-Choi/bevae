from loguru import logger


def print_args(args):
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
