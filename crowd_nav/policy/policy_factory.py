policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.pas_rnn import PASRNN
from crowd_nav.policy.diffstack import PaS_DiffStack
from crowd_nav.policy.mppi import PaS_MPPI

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['pas_rnn'] = PASRNN
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['pas_diffstack'] = PaS_DiffStack
policy_factory['pas_mppi'] = PaS_MPPI

