from colorama import Fore
import sys
import torch


def get_tqdm_config(total, leave=True, color='white'):
    fore_colors = {
        'red': Fore.LIGHTRED_EX,
        'green': Fore.LIGHTGREEN_EX,
        'yellow': Fore.LIGHTYELLOW_EX,
        'blue': Fore.LIGHTBLUE_EX,
        'magenta': Fore.LIGHTMAGENTA_EX,
        'cyan': Fore.LIGHTCYAN_EX,
        'white': Fore.LIGHTWHITE_EX,
    }
    return {
        'file': sys.stdout,
        'total': total,
        'desc': " ",
        'dynamic_ncols': True,
        'bar_format':
            "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]" % (fore_colors[color], Fore.RESET),
        'leave': leave
    }
    
class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
        
class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
    
def get_wd_param_list(model: torch.nn.Module):
    """
    Get list of model parameters to which weight decay should be applied. The function basically filters out
    all BatchNorm-related parameters to which weight decay should not be applied.
    Parameters
    ----------
    model: torch.nn.Module
        torch model which is trained using weight decay.
    Returns
    -------
    wd_param_list: List
        List containing two dictionaries containing parameters for which weight decay should be applied and parameters
        to which weight decay should not be applied.
    """
    wd_params, no_wd_params = [], []
    for name, param in model.named_parameters():
        # Filter BatchNorm parameters from weight decay parameters
        if "bn" in name:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return [{"params": wd_params}, {"params": no_wd_params, "weight_decay": 0}]