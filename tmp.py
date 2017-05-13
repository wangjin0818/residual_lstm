import os
import sys
import logging

import re

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    log_file = os.path.join('log', '1-layer lstm.txt')
    data = []
    with open(log_file, 'r') as my_file:
        for line in my_file.readlines():
            if not line.startswith('Epoch') and not line.startswith('2017'):
                line_data = re.findall(r'\d+\.?\d*', line)
                print(line_data)