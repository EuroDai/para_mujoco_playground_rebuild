"""Base classes for para nontendon fr3."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.parahand.para_nontendon_fr3 import para_nontendon_fr3_constants as consts

def get_assets() -> Dict[str, bytes]:
  """Loads XML assets for the para_nontendon_fr3 environments.

  The XML at xmls/para_nontendon_fr3.xml is self-contained (no external mesh
  or texture references), so we only need to expose XML files in the assets
  dict for `mujoco.MjModel.from_xml_string`.
  """
  assets: Dict[str, bytes] = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  return assets

class ParaNontendonFR3Env(mjx_env.MjxEnv):
  def __init__(
    self,
    xml_path: str,
    config: config_dict.ConfigDict,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 10

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model