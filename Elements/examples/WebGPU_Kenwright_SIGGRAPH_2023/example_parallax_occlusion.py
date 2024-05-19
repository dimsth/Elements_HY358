from random import random, randint
import wgpu   
import glm 
import numpy as np   
import Elements.definitions as definitions

from Elements.pyGLV.GUI.Viewer import GLFWWindow 
from Elements.pyGLV.GUI.Viewer import button_map
from Elements.pyGLV.GUI.fps_cammera import cammera
import Elements.pyECSS.math_utilities as util 
from Elements.definitions import TEXTURE_DIR, MODEL_DIR
from Elements.pyGLV.GL.wgpu_texture import ImportTexture 
import Elements.utils.normals as norm

from Elements.pyGLV.GL.wpgu_scene import Scene 
from Elements.pyGLV.GL.wgpu_material import Material
from Elements.pyGLV.GL.wgpu_object import Object, import_mesh
from Elements.pyGLV.GUI.wgpu_renderer import Renderer, RENDER_PASSES, PASS
from Elements.pyGLV.GL.Shader.wgpu_SimpleShader import SimpleShader 
from Elements.pyGLV.GL.Shader.wgpu_ParallaxOclusion import ParallaxOclusion


canvas = GLFWWindow(windowHeight=800, windowWidth=1280, wgpu=True, windowTitle="Wgpu Example")
canvas.init()
canvas.init_post() 

width = canvas._windowWidth 
height = canvas._windowHeight

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# Prepare present context
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)    

class objext(Object): 
    def __init__(self, *args, instance_count, **kwards):
        super(*args).__init__(*args, **kwards) 
        self.instance_count = instance_count

    def onInit(self):  
        self.attachedMaterial = Material(tag="simple", shader=ParallaxOclusion(device=device)) 

        texture = ImportTexture("colorTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1.png")
        texture.make(device=device) 
        self.attachedMaterial.shader.setDiffuseTexture(value=texture)   

        texture = ImportTexture("colorTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1normal.png")
        texture.make(device=device) 
        self.attachedMaterial.shader.setNormalTexture(value=texture)   

        texture = ImportTexture("colorTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1Height.png")
        texture.make(device=device) 
        self.attachedMaterial.shader.setSpecularTexture(value=texture)  

        # for i in range(0, self.instance_count - 1):
        model = glm.mat4x4(1)
        model = glm.transpose(glm.translate(model, glm.vec3(0.0, 0.0, 0.0))) 
        # model = glm.scale(model, glm.vec3(10, 10, 10))  
        # model = glm.rotate(model, np.deg2rad(-90), glm.vec3(1, 0, 0)) 
        model = glm.rotate(model, np.deg2rad(90), glm.vec3(0, 0, 1)) 

        self.attachedMaterial.shader.Models.append(np.array(model, dtype=np.float32))
        # self.attachedMaterial.shader.Models.append(
        #     np.array(
        #         glm.transpose(
        #             glm.translate(
        #                 # glm.mat4x4(1), glm.vec3(randint(0, 40), randint(0, 40), randint(0, 40))
        #                 glm.mat4x4(1), glm.vec3(0.0, 0.0, 0.0)
        #             )
        #         ),
        #         dtype=np.float32
        #     )
        # ) 
        self.load_mesh_from_obj(path=definitions.MODEL_DIR / "cube-sphere" / "plane.obj") 

    def onUpdate(self): 
        return; 

class talble(Object): 
    def __init__(self, *args, instance_count, **kwards):
        super(*args).__init__(*args, **kwards) 
        self.instance_count = instance_count

    def onInit(self):  
        self.attachedMaterial = Material(tag="simple", shader=ParallaxOclusion(device=device)) 

        texture = ImportTexture("colorTexture", path=definitions.MODEL_DIR / "stronghold" / "textures" / "texture_building.jpeg")
        texture.make(device=device) 
        self.attachedMaterial.shader.setDiffuseTexture(value=texture)   

        texture = ImportTexture("normal", path=definitions.MODEL_DIR / "stronghold" / "textures" / "texture_building_bumpmap.jpg")
        texture.make(device=device) 
        self.attachedMaterial.shader.setNormalTexture(value=texture)   

        texture = ImportTexture("spec", path=definitions.MODEL_DIR / "stronghold" / "textures" / "texture_building_specular.jpg")
        texture.make(device=device) 
        self.attachedMaterial.shader.setSpecularTexture(value=texture)  

        # for i in range(0, self.instance_count - 1):
        model = glm.mat4x4(1)
        model = glm.scale(model, glm.vec3(0.01, 0.01, 0.01))
        model = glm.transpose(glm.translate(model, glm.vec3(-400.0, 0.0, 0.0)))
        # model = glm.scale(model, glm.vec3(10, 10, 10))  
        # model = glm.rotate(model, np.deg2rad(-90), glm.vec3(1, 0, 0)) 
        # model = glm.rotate(model, np.deg2rad(90), glm.vec3(0, 0, 1)) 

        self.attachedMaterial.shader.Models.append(np.array(model, dtype=np.float32))
        # self.attachedMaterial.shader.Models.append(
        #     np.array(
        #         glm.transpose(
        #             glm.translate(
        #                 # glm.mat4x4(1), glm.vec3(randint(0, 40), randint(0, 40), randint(0, 40))
        #                 glm.mat4x4(1), glm.vec3(0.0, 0.0, 0.0)
        #             )
        #         ),
        #         dtype=np.float32
        #     )
        # ) 
        self.load_mesh_from_obj(path=definitions.MODEL_DIR / "stronghold" / "source" / "building.obj") 

    def onUpdate(self): 
        return; 


class talble2(Object): 
    def __init__(self, *args, instance_count, **kwards):
        super(*args).__init__(*args, **kwards) 
        self.instance_count = instance_count

    def onInit(self):  
        self.attachedMaterial = Material(tag="simple", shader=SimpleShader(device=device)) 

        # texture = ImprotTexture("colorTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1.png") 
        texture = ImportTexture("colorTexture", path=definitions.MODEL_DIR / "stronghold" / "textures" / "texture_building.jpeg")
        texture.make(device=device) 
        self.attachedMaterial.shader.setTexture(value=texture)   

        # texture = ImprotTexture("heightTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1Height.png")
        # texture.make(device=device) 
        # self.attachedMaterial.shader.setHeightTexture(value=texture)   

        # texture = ImprotTexture("normalTexture", path=definitions.TEXTURE_DIR / "BrickWall" / "wall1normal.png")
        # texture.make(device=device) 
        # self.attachedMaterial.shader.setNormalTexture(value=texture)  

        # for i in range(0, self.instance_count - 1):  
        model = glm.mat4x4(1)
        model = glm.scale(model, glm.vec3(0.01, 0.01, 0.01))
        # model = glm.scale(model, glm.vec3(10, 10, 10))  
        # model = glm.rotate(model, np.deg2rad(-90), glm.vec3(1, 0, 0)) 
        # model = glm.rotate(model, np.deg2rad(90), glm.vec3(0, 0, 1)) 

        self.attachedMaterial.shader.Models.append(np.array(model, dtype=np.float32))
        # self.attachedMaterial.shader.Models.append(
        #     np.array(
        #         glm.transpose(
        #             glm.translate(
        #                 # glm.mat4x4(1), glm.vec3(randint(0, 40), randint(0, 40), randint(0, 40))
        #                 glm.mat4x4(1), glm.vec3(0.0, 0.0, 0.0)
        #             )
        #         ),
        #         dtype=np.float32
        #     )
        # ) 

        self.load_mesh_from_obj(path=definitions.MODEL_DIR / "stronghold" / "source" / "building.obj") 

    def onUpdate(self): 
        return; 



# class mediumRare(Object): 
#     def __init__(self, *args, instance_count, **kwards):
#         super(*args).__init__(*args, **kwards) 
#         self.instance_count = instance_count

#     def onInit(self):  
#         self.attachedMaterial = Material(tag="simple", shader=SimpleShader(device=device))
#         texture = ImprotTexture("toolsTable", path=definitions.MODEL_DIR / "Anesthesia" / "Anaisthesia_UVS_02_Material__26_Albedo.png")
#         texture.make(device=device)
#         self.attachedMaterial.shader.setTexture(value=texture)  

#         for i in range(0, self.instance_count - 1):
#             self.attachedMaterial.shader.Models.append(
#                 np.array(
#                     glm.transpose(
#                         glm.translate(
#                             glm.mat4x4(1), glm.vec3(randint(0, 40), randint(0, 40), randint(0, 40))
#                         )
#                     ),
#                     dtype=np.float32
#                 )
#             ) 
            
#         self.load_mesh_from_obj(path=definitions.MODEL_DIR / "Anesthesia" / "Anesthesia.obj") 

#     def onUpdate(self): 
#         return;


scene = Scene()
obj1 = talble(instance_count=1); 
obj2 = talble2(instance_count=1); 
obj3 = objext(instance_count=2)
# obj2 = mediumRare(instance_count=256)
# scene.append_object(obj2)
# scene.append_object(obj1)   
# scene.append_object(obj2)  
scene.append_object(obj3)

cam = cammera([-5, -5, 5], 0, 0)
scene.set_cammera(cam=cam)

scene.set_light(glm.vec4(-10, 15, -15, 1.0)) 

renderer = Renderer() 

renderer.init(
    scene=scene,
    device=device, 
    present_context=present_context
) 

RENDER_PASSES.append(PASS.MODEL)

def draw_frame(): 
    renderer.render() 
    canvas.request_draw()

canvas.request_draw(draw_frame)

while canvas._running:
    event = canvas.event_input_process();   
    scene.update(canvas, event) 

    if canvas._need_draw:
        canvas.display()
        canvas.display_post()
        
canvas.shutdown()