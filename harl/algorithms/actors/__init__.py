"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.adaptive_hatrpo import Adaptive_HATRPO  #WIP - Added; Please check this
#from harl.algorithms.actors.adaptive_wasserstein import ?          #WIP - To be Added; Nothing is defined there for now; Please check this
#from harl.algorithms.actors.nonadaptive_wasserstein import ?       #WIP - To be Added; Nothing is defined there for now; Please check this
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "adaptive_hatrpo": Adaptive_HATRPO,         #WIP - Added; Please check this
#    "adaptive_wasserstein": ?,                 #WIP - To be Added; Nothing is defined there for now; Please check this
#    "nonadaptive_wasserstein": ?,              #WIP - To be Added; Nothing is defined there for now; Please check this
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
}
