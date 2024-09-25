from toolbox.data.hub.hub_ids.cchit_hub_id import CCHITHubId
from toolbox.data.hub.hub_ids.cm1_hub_id import CM1HubId
from toolbox.data.hub.hub_ids.drone_hub_id import DroneHubId
from toolbox.data.hub.hub_ids.git_hub_id import GitHubId
from toolbox.data.hub.hub_ids.iceoryx_hub_id import IceoryxHubId
from toolbox.data.hub.hub_ids.itrust_hub_id import ITrustHubId
from toolbox.data.hub.hub_ids.jpl_hub_id import JplHubId
from toolbox.data.hub.hub_ids.mip_hub_id import MipHubId
from toolbox.data.hub.hub_ids.train_controller_hub_id import TrainControllerHubId
from toolbox.util.supported_enum import SupportedEnum


class SupportedDatasets(SupportedEnum):
    DRONE = DroneHubId
    ITRUST = ITrustHubId
    CM1 = CM1HubId
    MIP = MipHubId
    CCHIT = CCHITHubId
    TRAINCONTROLLER = TrainControllerHubId
    ICEORYX = IceoryxHubId
    GIT = GitHubId
    JPL = JplHubId
