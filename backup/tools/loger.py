import logging
from logging import handlers
import os


def logFactor(filename):
    logger = logging.getLogger('/home/xiaonan/factor_wxn/Log/{}.log'.format(filename))
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(funcName)s')

    handler = handlers.TimedRotatingFileHandler('/home/xiaonan/factor_wxn/Log/{}.log'.format(filename), when = 'D', backupCount = 30, encoding = 'utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)


    return(logger)

def logggggg(a):
    a.info('asdf.')

if __name__ == "__main__":
    # make sure the folder log exists.
    '''
    if os.path.exists(r'./Log/'):
        pass
    else:
        os.mkdir(r'./Log/')
    
    filename = 'run.py'
    logger = logging.getLogger('xiaonan-{}'.format(filename))
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler('./Log/log.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger = logFactor('asdfgagadsggafdhg.py')
    logger.info('sadfgdfsg.')
    logggggg(logger)
    # logger.info('Finished.')