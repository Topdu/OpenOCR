from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml


class ArgsParser(ArgumentParser):

    def __init__(self):
        super(ArgsParser,
              self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument('-c', '--config', help='configuration file to use')
        self.add_argument('-o',
                          '--opt',
                          nargs='*',
                          help='set configuration options')
        self.add_argument('--local_rank')
        self.add_argument('--local-rank')

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, 'Please specify --config=configure_file_path.'
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config
