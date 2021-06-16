import neptune.new as neptune

class Logger():
    def __init__(self, use_neptune):
        self.use_neptune = use_neptune
        if self.use_neptune:
            self.neptune_run = neptune.init(project="sungsahn0215/mol-hrl", source_files=["*.py", "**/*.py"])
    
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("logger")
        group.add_argument("--logger_use_neptune", action="store_true")
        return parser

    def set(self, key, value):
        print(key)
        print(value)
        if self.use_neptune:
            self.neptune_run[key]=value

    def log(self, statistics, prefix=""):
        prefixed_statistics = {f"{prefix}/{key}": val for key, val in statistics.items()}
        print(prefixed_statistics)
        if self.use_neptune:
            for key in prefixed_statistics:
                self.neptune_run[key].log(prefixed_statistics[key])
