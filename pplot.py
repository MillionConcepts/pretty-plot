"""user-facing noninteractive pplot utility"""

from clize import run

import pplot.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(pplot.cli.do_pplot)