"""Base classes for para nontendon fr3."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp
import numpy as np

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
    返回每个指尖与 cube 接触的三维世界系接触合力。
    返回形状为 [5, 3]，对应 [thumb,index,middle,ring,little]。
    """
    if hasattr(data._impl, "contact__geom"):
      geom = data._impl.contact__geom
      geom1 = geom[:, 0]
      geom2 = geom[:, 1]
      efc_address = data._impl.contact__efc_address[:, 0]
      ncon = jp.asarray(data._impl.nacon).reshape(-1)[0]
      valid = (jp.arange(geom.shape[0]) < ncon) & (efc_address >= 0)
      frame = data._impl.contact__frame.reshape(-1, 3, 3)
      dim = data._impl.contact__dim[:, 0]
      friction = data._impl.contact__friction
      efc_force = data._impl.efc__force
    else:
      contact = data._impl.contact
      geom1 = contact.geom1
      geom2 = contact.geom2
      efc_address = contact.efc_address
      valid = (geom1 >= 0) & (geom2 >= 0) & (efc_address >= 0)
      frame = contact.frame.reshape(-1, 3, 3)
      dim = contact.dim
      friction = contact.friction
      efc_force = data._impl.efc_force

    cube_geom_ids = jp.array([self.mj_model.geom(name).id for name in consts.CUBE_GEOMS])
    geom1_is_cube = jp.any(geom1[:, None] == cube_geom_ids[None, :], axis=1)
    geom2_is_cube = jp.any(geom2[:, None] == cube_geom_ids[None, :], axis=1)

    def _decode_contact_force(cid: int) -> jax.Array:
      addr = efc_address[cid]
      condim = dim[cid]

      def _invalid():
        return jp.zeros(3, dtype=efc_force.dtype)

      def _valid():
        if self.mj_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
          if condim == 1:
            force_contact = jp.array([efc_force[addr], 0.0, 0.0])
          else:
            pyr = efc_force[addr : addr + 2 * (condim - 1)]
            fri = friction[cid]
            force_contact = jp.array([
                pyr[0::2].sum() + pyr[1::2].sum(),
                (pyr[0::2] - pyr[1::2]) @ fri[: condim - 1],
                jp.zeros((), dtype=efc_force.dtype),
            ])
        else:
          force_contact = jp.zeros(3, dtype=efc_force.dtype)
          force_contact = force_contact.at[:condim].set(efc_force[addr : addr + condim])

        return force_contact @ frame[cid]

      return jax.lax.cond((addr >= 0) & valid[cid], _valid, _invalid)

    contact_force_vec = jax.vmap(_decode_contact_force)(jp.arange(geom1.shape[0]))

    contacts = []
    for name in consts.FINGERTIP_TACS:
      tip_geom_id = self.mj_model.geom(name).id
      tip_on_geom1 = (geom1 == tip_geom_id) & geom2_is_cube
      tip_on_geom2 = (geom2 == tip_geom_id) & geom1_is_cube
      tip_cube_contact = valid & (tip_on_geom1 | tip_on_geom2)
      contacts.append(
          jp.sum(jp.where(tip_cube_contact[:, None], contact_force_vec, 0.0), axis=0)
      )

    return jp.stack(contacts)

  def _mat_to_quat(mat: jax.Array) -> jax.Array:
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]

    def case_trace_positive():
      s = 2.0 * jp.sqrt(trace + 1.0)
      qw = 0.25 * s
      qx = (mat[2, 1] - mat[1, 2]) / s
      qy = (mat[0, 2] - mat[2, 0]) / s
      qz = (mat[1, 0] - mat[0, 1]) / s
      return jp.array([qw, qx, qy, qz])

    def case_x():
      s = 2.0 * jp.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
      qw = (mat[2, 1] - mat[1, 2]) / s
      qx = 0.25 * s
      qy = (mat[0, 1] + mat[1, 0]) / s
      qz = (mat[0, 2] + mat[2, 0]) / s
      return jp.array([qw, qx, qy, qz])

    def case_y():
      s = 2.0 * jp.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
      qw = (mat[0, 2] - mat[2, 0]) / s
      qx = (mat[0, 1] + mat[1, 0]) / s
      qy = 0.25 * s
      qz = (mat[1, 2] + mat[2, 1]) / s
      return jp.array([qw, qx, qy, qz])

    def case_z():
      s = 2.0 * jp.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
      qw = (mat[1, 0] - mat[0, 1]) / s
      qx = (mat[0, 2] + mat[2, 0]) / s
      qy = (mat[1, 2] + mat[2, 1]) / s
      qz = 0.25 * s
      return jp.array([qw, qx, qy, qz])

    return jp.where(
      trace > 0,
      case_trace_positive(),
      jp.where(
          (mat[0, 0] > mat[1, 1]) & (mat[0, 0] > mat[2, 2]),
          case_x(),
          jp.where(mat[1, 1] > mat[2, 2], case_y(), case_z()),
      ),
    )

  def get_box_pointcloud(
    self,
    data: mjx.Data,
    num_points: int,
    box_geom_name: str = "cube",
  ) -> jax.Array:
    geom_id = self.mj_model.geom(box_geom_name).id
    if self.mj_model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX.value:
      raise ValueError(f"Geom {box_geom_name!r} is not a box.")

    hx, hy, hz = self.mj_model.geom_size[geom_id]
    face_areas = np.array([
        4.0 * hy * hz,
        4.0 * hy * hz,
        4.0 * hx * hz,
        4.0 * hx * hz,
        4.0 * hx * hy,
        4.0 * hx * hy,
    ])
    face_counts = np.floor(num_points * face_areas / np.sum(face_areas)).astype(int)
    remainder = num_points - int(face_counts.sum())
    if remainder > 0:
      face_fracs = num_points * face_areas / np.sum(face_areas) - face_counts
      face_counts[np.argsort(-face_fracs)[:remainder]] += 1

    local_points = []
    face_extents = [
        (hy, hz),
        (hy, hz),
        (hx, hz),
        (hx, hz),
        (hx, hy),
        (hx, hy),
    ]
    for face_id, face_count in enumerate(face_counts):
      if face_count == 0:
        continue
      u_extent, v_extent = face_extents[face_id]
      cols = max(1, int(np.ceil(np.sqrt(face_count * u_extent / v_extent))))
      rows = int(np.ceil(face_count / cols))
      u = (np.arange(cols) + 0.5) / cols * 2.0 - 1.0
      v = (np.arange(rows) + 0.5) / rows * 2.0 - 1.0
      uu, vv = np.meshgrid(u, v, indexing="xy")
      uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)[:face_count]

      if face_id == 0:
        face_points = np.stack([
            np.full(face_count, hx), uv[:, 0] * hy, uv[:, 1] * hz
        ], axis=-1)
      elif face_id == 1:
        face_points = np.stack([
            np.full(face_count, -hx), uv[:, 0] * hy, uv[:, 1] * hz
        ], axis=-1)
      elif face_id == 2:
        face_points = np.stack([
            uv[:, 0] * hx, np.full(face_count, hy), uv[:, 1] * hz
        ], axis=-1)
      elif face_id == 3:
        face_points = np.stack([
            uv[:, 0] * hx, np.full(face_count, -hy), uv[:, 1] * hz
        ], axis=-1)
      elif face_id == 4:
        face_points = np.stack([
            uv[:, 0] * hx, uv[:, 1] * hy, np.full(face_count, hz)
        ], axis=-1)
      else:
        face_points = np.stack([
            uv[:, 0] * hx, uv[:, 1] * hy, np.full(face_count, -hz)
        ], axis=-1)
      local_points.append(face_points)

    if local_points:
      local_points = jp.array(np.concatenate(local_points, axis=0), dtype=data.qpos.dtype)
    else:
      local_points = jp.zeros((0, 3), dtype=data.qpos.dtype)

    geom_xmat = data.geom_xmat[geom_id]
    geom_xpos = data.geom_xpos[geom_id]
    return local_points @ geom_xmat.T + geom_xpos

  def render(
    self,
    trajectory,
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    scene_option=None,
    modify_scene_fns=None,
  ):
    return super().render(
      trajectory,
      height=height,
      width=width,
      camera="default" if camera is None else camera,
      scene_option=scene_option,
      modify_scene_fns=modify_scene_fns,
    )

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
  