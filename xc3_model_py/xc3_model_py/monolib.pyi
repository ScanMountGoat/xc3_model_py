from typing import Optional

from xc3_model_py.xc3_model_py import ImageTexture

class ShaderTextures:
    @staticmethod
    def from_folder(path: str) -> ShaderTextures: ...
    def global_texture(self, sampler_name: str) -> Optional[ImageTexture]: ...