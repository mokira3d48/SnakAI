import logging
import logging.config
from .agent import train


logging.config.fileConfig('logging.conf')
LOG = logging.getLogger('package_name')


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    main()
