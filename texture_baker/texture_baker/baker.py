import torch
import torch.nn as nn
from torch import Tensor


class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()

    def rasterize(
        self,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Rasterize the UV coordinates to a barycentric coordinates
        & Triangle idxs texture map

        Args:
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 4, float: Rasterized map
        """
        def empty_raster(device: torch.device) -> Tensor:
            out = torch.zeros(
                (bake_resolution, bake_resolution, 4), dtype=torch.float32
            )
            out[..., 3] = -1.0
            return out.to(device)

        def all_faces_degenerate(uv_cpu: Tensor, faces_cpu: Tensor) -> bool:
            if faces_cpu.numel() == 0:
                return True
            nv = uv_cpu.shape[0]
            if nv == 0:
                return True
            valid_idx = (
                (faces_cpu >= 0) & (faces_cpu < nv)
            ).all(dim=1)
            if not bool(valid_idx.any()):
                return True
            tri = uv_cpu[faces_cpu[valid_idx]]
            v0 = tri[:, 0, :]
            v1 = tri[:, 1, :]
            v2 = tri[:, 2, :]
            # 2D triangle signed area * 2
            area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (
                v1[:, 1] - v0[:, 1]
            ) * (v2[:, 0] - v0[:, 0])
            return not bool((area2.abs() > 1e-12).any())

        device = uv.device
        if device.type != "cpu":
            # Built as CPU-only extension: run op on CPU, return tensor on original device.
            uv_cpu = uv.cpu()
            faces_cpu = face_indices.cpu().to(torch.int32)
            if all_faces_degenerate(uv_cpu, faces_cpu):
                return empty_raster(device)
            out = torch.ops.texture_baker_cpp.rasterize(
                uv_cpu,
                faces_cpu,
                bake_resolution,
            )
            return out.to(device)
        faces_cpu = face_indices.to(torch.int32)
        if all_faces_degenerate(uv, faces_cpu):
            return empty_raster(device)
        return torch.ops.texture_baker_cpp.rasterize(
            uv, faces_cpu, bake_resolution
        )

    def get_mask(self, rast: Tensor) -> Tensor:
        """
        Get the occupancy mask from the rasterized map

        Args:
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map

        Returns:
            Tensor, bake_resolution bake_resolution, bool: Mask
        """
        return rast[..., -1] >= 0

    def interpolate(
        self,
        attr: Tensor,
        rast: Tensor,
        face_indices: Tensor,
    ) -> Tensor:
        """
        Interpolate the attributes using the rasterized map

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Interpolated attributes
        """
        device = attr.device
        if device.type != "cpu":
            out = torch.ops.texture_baker_cpp.interpolate(
                attr.cpu(),
                face_indices.cpu().to(torch.int32),
                rast.cpu(),
            )
            return out.to(device)
        return torch.ops.texture_baker_cpp.interpolate(
            attr, face_indices.to(torch.int32), rast
        )

    def forward(
        self,
        attr: Tensor,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Bake the texture

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Baked texture
        """
        rast = self.rasterize(uv, face_indices, bake_resolution)
        return self.interpolate(attr, rast, face_indices)
