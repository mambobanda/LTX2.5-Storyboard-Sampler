from comfy_api.latest import ComfyExtension

async def comfy_entrypoint() -> ComfyExtension:
    class LTX25Extension(ComfyExtension):
        async def get_node_list(self):
            from .nodes_ltx25_storyboard_strided import LTX25StoryboardStrided
            return [
                LTX25StoryboardStrided,
            ]

    return LTX25Extension()
