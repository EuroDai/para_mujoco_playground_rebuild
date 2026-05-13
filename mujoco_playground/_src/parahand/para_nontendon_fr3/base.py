"""Base classes for para nontendon fr3."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp

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

  def get_fingertip_cube_contact(self, data: mjx.Data) -> jax.Array:
    """
    返回每个指尖与 cube 接触的法向力大小。
    返回[thumb,index,middle,ring,little]
    """
    if hasattr(data._impl, "contact__geom"):
      geom = data._impl.contact__geom
      geom1 = geom[:, 0]
      geom2 = geom[:, 1]
      efc_address = data._impl.contact__efc_address[:, 0]
      ncon = jp.asarray(data._impl.nacon).reshape(-1)[0]
      valid = (jp.arange(geom.shape[0]) < ncon) & (efc_address >= 0)
      efc_force = data._impl.efc__force
    else:
      contact = data._impl.contact
      geom1 = contact.geom1
      geom2 = contact.geom2
      efc_address = contact.efc_address
      valid = (geom1 >= 0) & (geom2 >= 0) & (efc_address >= 0)
      efc_force = data._impl.efc_force

    cube_geom_ids = jp.array([self.mj_model.geom(name).id for name in consts.CUBE_GEOMS])
    geom1_is_cube = jp.any(geom1[:, None] == cube_geom_ids[None, :], axis=1)
    geom2_is_cube = jp.any(geom2[:, None] == cube_geom_ids[None, :], axis=1)

    safe_efc_address = jp.where(valid, efc_address, 0)
    contact_normal_force = jp.where(valid, jp.abs(efc_force[safe_efc_address]), 0.0)

    contacts = []
    for name in consts.FINGERTIP_TACS:
      tip_geom_id = self.mj_model.geom(name).id
      tip_on_geom1 = (geom1 == tip_geom_id) & geom2_is_cube
      tip_on_geom2 = (geom2 == tip_geom_id) & geom1_is_cube
      tip_cube_contact = valid & (tip_on_geom1 | tip_on_geom2)
      contacts.append(jp.sum(jp.where(tip_cube_contact, contact_normal_force, 0.0)))

    return jp.stack(contacts)

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

def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  theta = jax.random.uniform(rng, (), minval=0.0, maxval=2 * jp.pi)
  return jp.array([
      jp.cos(theta / 2),
      0.0,
      0.0,
      jp.sin(theta / 2),
  ])