"""user-facing noninteractive pretty_plot utility"""

from clize import run

import pretty_plot.cli


# tell clize to handle command line call
if __name__ == '__main__':
    run(pretty_plot.cli.do_pplot)