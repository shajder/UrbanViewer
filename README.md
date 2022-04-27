# UrbanViewer
This is an example of OpenStreetMap data viewer implemented based on the OpenSceneGraph library.

Main idea of this small project is to visualize cities' urban areas with a single instanced box.

As input data OpenStreetMap shapefiles and pbf were converted to tiled LODs with instanced textures attached to box geometry underneath.

Ground layer was also generated based on OpenStreetMap data with a separate off-screen tool.

Additionally two major effects were applied:

-PSSM shadowing which is an improved technique inherited from the OpenSceneGraph project. Instead of N-parallel split passes we can use a single Layered pass which is rendered through FBO into Texture2DArray.
-Ambient Occlusion adapted to OpenSceneGraph from this project: https://github.com/nvpro-samples/gl_ssao


