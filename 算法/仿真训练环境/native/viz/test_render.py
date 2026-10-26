"""Test headless MuJoCo rendering (EGL) inside WSL."""
import mujoco

print("mujoco", mujoco.__version__)
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
    <body pos="0 0 .5">
      <geom type="box" size=".2 .1 .05" rgba=".2 .5 .9 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 240, 320)
renderer.update_scene(data)
image = renderer.render()
print("render_ok", image.shape, float(image.mean()))
