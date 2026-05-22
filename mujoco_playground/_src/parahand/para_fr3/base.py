"""Base classes for para nontendon fr3."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support
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

class ParaFR3Env(mjx_env.MjxEnv):
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
      efc_address = data._impl.contact__efc_address
      ncon = jp.asarray(data._impl.nacon).reshape(-1)[0]
      valid = (jp.arange(geom.shape[0]) < ncon) & (efc_address[:, 0] >= 0)
      frame = data._impl.contact__frame
      condim = data._impl.contact__dim
      friction = data._impl.contact__friction
      efc_force = data._impl.efc__force
      is_pyramidal = self.mj_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL

      def _decode_contact_force(cid: jax.Array) -> jax.Array:
        addr_row = efc_address[cid]
        addr0 = addr_row[0]
        dim = condim[cid]
        fric = friction[cid]
        mat = frame[cid]
        dtype = efc_force.dtype

        def _invalid():
          return jp.zeros(3, dtype=dtype)

        def _valid():
          if is_pyramidal:
            offsets = jp.arange(10, dtype=addr0.dtype)
            indices = addr0 + offsets
            safe_indices = jp.clip(indices, 0, efc_force.shape[0] - 1)
            pyramid = jp.where(
                (indices >= 0) & (indices < efc_force.shape[0]),
                efc_force[safe_indices],
                0.0,
            )

            def _dim1():
              return jp.array([pyramid[0], 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)

            def _dimn():
              pairs = jp.arange(5)
              pair_count = 2 * (dim - 1)
              normal = jp.where(offsets < pair_count, pyramid, 0.0).sum()
              tangential = (pyramid[2 * pairs] - pyramid[2 * pairs + 1]) * fric[:5]
              wrench = jp.zeros(6, dtype=dtype)
              wrench = wrench.at[0].set(normal)
              wrench = wrench.at[1:].set(
                  jp.where(pairs < (dim - 1), tangential, 0.0)
              )
              return wrench

            wrench_contact = jax.lax.cond(dim == 1, _dim1, _dimn)
          else:
            safe_indices = jp.clip(addr_row, 0, efc_force.shape[0] - 1)
            values = jp.where(
                (addr_row >= 0) & (addr_row < efc_force.shape[0]),
                efc_force[safe_indices],
                0.0,
            )
            wrench_contact = jp.where(jp.arange(6) < dim, values, 0.0)

          wrench_world = (wrench_contact.reshape(2, 3) @ mat).reshape(-1)
          return wrench_world[:3]

        return jax.lax.cond(valid[cid], _valid, _invalid)

      contact_force_vec = jax.vmap(_decode_contact_force)(jp.arange(geom1.shape[0]))
    else:
      contact = data._impl.contact
      geom1 = contact.geom1
      geom2 = contact.geom2
      efc_address = contact.efc_address
      valid = (geom1 >= 0) & (geom2 >= 0) & (efc_address >= 0)

      contact_ids = jp.arange(geom1.shape[0])
      contact_wrench = jax.vmap(
          lambda cid: support.contact_force(
              self.mjx_model, data, cid, to_world_frame=True
          )
      )(contact_ids)
      contact_force_vec = contact_wrench[:, :3] * valid[:, None]

    cube_geom_ids = jp.array([self.mj_model.geom(name).id for name in consts.CUBE_GEOMS])
    geom1_is_cube = jp.any(geom1[:, None] == cube_geom_ids[None, :], axis=1)
    geom2_is_cube = jp.any(geom2[:, None] == cube_geom_ids[None, :], axis=1)

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

  @staticmethod
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

  def get_fingertip_kinematics(
    self,
    data: mjx.Data,
    site_ids: jax.Array,
    body_ids: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    fingertip_pos = data.site_xpos[site_ids]
    fingertip_quat = jax.vmap(self._mat_to_quat)(data.site_xmat[site_ids])

    def _get_vel(pos: jax.Array, body_id: jax.Array) -> tuple[jax.Array, jax.Array]:
      jacp, jacr = mjx.jac(self.mjx_model, data, pos, body_id)
      return jacp.T @ data.qvel, jacr.T @ data.qvel

    fingertip_linvel, fingertip_angvel = jax.vmap(_get_vel)(
        fingertip_pos, body_ids
    )
    fingertip_pose = jp.concatenate([fingertip_pos, fingertip_quat], axis=-1)
    fingertip_vel = jp.concatenate([fingertip_linvel, fingertip_angvel], axis=-1)
    return fingertip_pose, fingertip_vel

  def _sample_box_pointcloud(
    self,
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

    pointcloud = []
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
      pointcloud.append(face_points)

    if pointcloud:
      return jp.array(np.concatenate(pointcloud, axis=0), dtype=jp.float32)
    return jp.zeros((0, 3), dtype=jp.float32)

  def get_box_pointcloud(
    self,
    data: mjx.Data,
    num_points: int,
    box_geom_name: str = "cube",
    pointcloud: Optional[jax.Array] = None,
  ) -> jax.Array:
    geom_id = self.mj_model.geom(box_geom_name).id
    if pointcloud is None:
      pointcloud = self._sample_box_pointcloud(num_points, box_geom_name)
    geom_xmat = data.geom_xmat[geom_id]
    geom_xpos = data.geom_xpos[geom_id]
    return pointcloud.astype(data.qpos.dtype) @ geom_xmat.T + geom_xpos

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
  