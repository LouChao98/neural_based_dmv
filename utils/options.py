import os
import json
import argparse
from dataclasses import dataclass

# do NOT name any options with '_' as the first char
@dataclass
class Options:
    load_option: str = ''

    def __post_init__(self):
        if self.load_option:
            self.load(self.load_option)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        for key, value in d.items():
            setattr(self, key, value)

    def print(self):
        print("============== OPTIONS =============")
        for key, value in self.get_iterator():
            print(f"{key.ljust(20, ' ')} = {value}")
        print("====================================")

    def to_dict(self):
        out = {}
        for key, value in self.get_iterator():
            if isinstance(value, Options):
                out[key] = value.get_iterator()
            else:
                out[key] = value
        return out

    def save(self, path):
        to_save = self.to_dict()
        with open(path, 'w') as f:
            json.dump(to_save, f, ensure_ascii=False, allow_nan=True, indent=4)

    def get_iterator(self):
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                yield key, value

    def set(self, name, value):
        setattr(self, name, value)

    def parse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.get_iterator():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', type=boolean_string, required=False)
            elif isinstance(value, Options):
                raise NotImplementedError
                value.parse()
            else:
                parser.add_argument(f'--{key}', type=type(value), required=False)
        options, unkown = parser.parse_known_args()
        if unkown:
            print(f'[WARNING] ignoring invalid arguments: {",".join(unkown)}')
        if options.load_option:
            self.load(options.load_option)
        for key, _ in self.get_iterator():
            if getattr(options, key) is not None:
                setattr(self, key, getattr(options, key))


def boolean_string(s):
    s = s.lower()
    if s not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
