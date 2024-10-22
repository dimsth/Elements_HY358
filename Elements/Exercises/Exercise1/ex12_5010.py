import numpy as np
import imgui
import Elements.pyECSS.math_utilities as util
from Elements.pyECSS.Entity import Entity
from Elements.pyECSS.Component import BasicTransform, Camera, RenderMesh
from Elements.pyECSS.System import  TransformSystem, CameraSystem
from Elements.pyGLV.GL.Scene import Scene
from Elements.pyGLV.GUI.Viewer import RenderGLStateSystem, ImGUIecssDecorator

from Elements.pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from Elements.pyGLV.GL.VertexArray import VertexArray

from Elements.utils.terrain import generateTerrain

from OpenGL.GL import GL_LINES

from Elements.utils.Shortcuts import displayGUI_text
example_description = \
"This is a scene with 2 cube, a trinagle and house." 

winWidth = 1024
winHeight = 768

scene = Scene()    

# Scenegraph with Entities, Components
rootEntity = scene.world.createEntity(Entity(name="RooT"))
entityCam1 = scene.world.createEntity(Entity(name="entityCam1"))
scene.world.addEntityChild(rootEntity, entityCam1)
trans1 = scene.world.addComponent(entityCam1, BasicTransform(name="trans1", trs=util.identity()))

eye = util.vec(1, 0.54, 1.0)
target = util.vec(0.02, 0.14, 0.217)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
projMat = util.perspective(50.0, 1.0, 1.0, 10.0)   
m = np.linalg.inv(projMat @ view)

entityCam2 = scene.world.createEntity(Entity(name="entityCam2"))
scene.world.addEntityChild(entityCam1, entityCam2)
trans2 = scene.world.addComponent(entityCam2, BasicTransform(name="trans2", trs=util.identity()))
# orthoCam = scene.world.addComponent(entityCam2, Camera(util.ortho(-100.0, 100.0, -100.0, 100.0, 1.0, 100.0), "orthoCam","Camera","500"))
orthoCam = scene.world.addComponent(entityCam2, Camera(m, "orthoCam","Camera","500"))

# a simple triangle
vertexData = np.array([
    [1.0, -0.5, 1.0, 1.0],
    [1.0, -0.5, 2.0, 1.0],
    [2.0, -0.5, 1.0, 1.0],
    [2.0, -0.5, 2.0, 1.0],
    [1.5, 1.0, 1.5, 1.0]
],dtype=np.float32) 
colorVertexData = np.array([
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0]
], dtype=np.float32)

#Simple Cube
vertexCube = np.array([
    [-0.5, -0.5, 0.5, 1.0],
    [-0.5, 0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0], 
    [-0.5, -0.5, -0.5, 1.0], 
    [-0.5, 0.5, -0.5, 1.0], 
    [0.5, 0.5, -0.5, 1.0], 
    [0.5, -0.5, -0.5, 1.0]
],dtype=np.float32) 
colorCube = np.array([
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0]
], dtype=np.float32)

vertexCube2 = np.array([
    [-1.5, -0.5, -2.5, 1.0],
    [-1.5, 0.5, -2.5, 1.0],
    [-2.5, 0.5, -2.5, 1.0],
    [-2.5, -0.5, -2.5, 1.0], 
    [-1.5, -0.5, -1.5, 1.0], 
    [-1.5, 0.5, -1.5, 1.0], 
    [-2.5, 0.5, -1.5, 1.0], 
    [-2.5, -0.5, -1.5, 1.0]
],dtype=np.float32) 

#index arrays for above vertex Arrays
indexData = np.array((0,1,4,  0,2,4,
                  1,3,4,  2,3,4,
                  0,1,2,  1,2,3), np.uint32)
indexCube = np.array((1,0,3, 1,3,2, 
                  2,3,7, 2,7,6,
                  3,0,4, 3,4,7,
                  6,5,1, 6,1,2,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1), np.uint32)

indexHouse = np.array((1,0,3, 1,3,2, 
                  2,3,7, 2,7,6,
                  3,0,4, 3,4,7,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1,
                  1,2,8, 1,5,8,
                  6,2,8, 6,5,8), np.uint32)

def addObjectToScene(name, position, index, colorArray):
    node = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, node)
    trans = scene.world.addComponent(node, BasicTransform(name=name + "_trans", trs=util.translate(0,0.5,0)))
    mesh = scene.world.addComponent(node, RenderMesh(name=name + "_mesh"))
    
    mesh.vertex_attributes.append(position)
    mesh.vertex_attributes.append(colorArray)
    mesh.vertex_index.append(index)
    
    vArray = scene.world.addComponent(node, VertexArray())
    shaderDec = scene.world.addComponent(node, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source = Shader.COLOR_FRAG)))
    
    return node, trans, vArray, shaderDec, mesh

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())


## ADD OBJECTS ##

node4, trans4, vArray4, shaderDec4, mesh4 = addObjectToScene("node4", vertexCube, indexCube, colorCube)
cube2, trans_cube2, vArray_cube2, shaderDec_cube2, mesh_cube2 = addObjectToScene("cube2", vertexCube2, indexCube, colorCube)
pyramid, trans_tri, vArray_tri, shaderDec_tri, mesh_tri = addObjectToScene("pyramid", vertexData, indexData, colorVertexData)

# Generate terrain

vertexTerrain, indexTerrain, colorTerrain= generateTerrain(size=4,N=20)
# Add terrain
terrain = scene.world.createEntity(Entity(name="terrain"))
scene.world.addEntityChild(rootEntity, terrain)
terrain_trans = scene.world.addComponent(terrain, BasicTransform(name="terrain_trans", trs=util.identity()))
terrain_mesh = scene.world.addComponent(terrain, RenderMesh(name="terrain_mesh"))
terrain_mesh.vertex_attributes.append(vertexTerrain) 
terrain_mesh.vertex_attributes.append(colorTerrain)
terrain_mesh.vertex_index.append(indexTerrain)
terrain_vArray = scene.world.addComponent(terrain, VertexArray(primitive=GL_LINES))
terrain_shader = scene.world.addComponent(terrain, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

# MAIN RENDERING LOOP

running = True
scene.init(imgui=True, windowWidth = winWidth, windowHeight = winHeight, windowTitle = "Elements: A Working Event Manager", customImGUIdecorator = ImGUIecssDecorator, openGLversion = 4)

scene.world.traverse_visit(initUpdate, scene.world.root)

################### EVENT MANAGER ###################

eManager = scene.world.eventManager
gWindow = scene.renderWindow
gGUI = scene.gContext

renderGLEventActuator = RenderGLStateSystem()


eManager._subscribers['OnUpdateWireframe'] = gWindow
eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
eManager._subscribers['OnUpdateCamera'] = gWindow 
eManager._actuators['OnUpdateCamera'] = renderGLEventActuator

cube1_translate = [0.0, 0.5, 0.0]
cube1_rotate = [0.0, 0.0, 0.0]
cube1_scale = [1.0, 1.0, 1.0]

cube2_translate = [0.0, 0.5, 0.0]
cube2_rotate = [0.0, 0.0, 0.0]
cube2_scale = [1.0, 1.0, 1.0]

pyramid_translate = [0.0, 0.5, 0.0]
pyramid_rotate = [0.0, 0.0, 0.0]
pyramid_scale = [1.0, 1.0, 1.0]

def TRSscreen():
    global cube1_translate
    global cube1_rotate
    global cube1_scale
    global cube2_translate
    global cube2_rotate
    global cube2_scale
    global pyramid_translate
    global pyramid_rotate
    global pyramid_scale
    imgui.begin("TRS")

    c1_changed = c2_changed = p_changed = False

    if imgui.begin_tab_bar("MyTabBar"):

        if imgui.begin_tab_item("Cube 1").selected:
            c1_changed_t, cube1_translate = imgui.drag_float3(
                "Translate", *cube1_translate
            )
            c1_changed_r, cube1_rotate = imgui.drag_float3(
                "Rotate", *cube1_rotate
            )
            c1_changed_s, cube1_scale = imgui.drag_float3(
                "Scale", *cube1_scale
            )
            c1_changed = c1_changed_t or c1_changed_r or c1_changed_s
            imgui.end_tab_item()

        if imgui.begin_tab_item("Cube 2").selected:
            c2_changed_t, cube2_translate = imgui.drag_float3(
                "Translate", *cube2_translate
            )
            c2_changed_r, cube2_rotate = imgui.drag_float3(
                "Rotate", *cube2_rotate
            )
            c2_changed_s, cube2_scale = imgui.drag_float3(
                "Scale", *cube2_scale
            )
            c2_changed = c2_changed_t or c2_changed_r or c2_changed_s
            imgui.end_tab_item()

        if imgui.begin_tab_item("Pyramid").selected:
            p_changed_t, pyramid_translate = imgui.drag_float3(
                "Translate", *pyramid_translate
            )
            p_changed_r, pyramid_rotate = imgui.drag_float3(
                "Rotate", *pyramid_rotate
            )
            p_changed_s, pyramid_scale = imgui.drag_float3(
                "Scale", *pyramid_scale
            )
            p_changed = p_changed_t or p_changed_r or p_changed_s
            imgui.end_tab_item()

    imgui.end_tab_bar()

    imgui.end()

    return c1_changed, c2_changed, p_changed



eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)

projMat = util.perspective(50.0, 1.0, 0.01, 10.0)
gWindow._myCamera = view
model_terrain = terrain.getChild(0).trs

model_cube = trans4.trs
model_tri = trans_tri.trs
model_cube2 = trans_cube2.trs

def trsChanges(model, translate, rotate, scale):
    model = util.translate(*translate)  
    model = model @ util.rotate([1, 0, 0], translate[0])  
    model = model @ util.rotate([0, 1, 0], translate[1])  
    model = model @ util.rotate([0, 0, 1], translate[2])  
    model = model @ util.scale(*scale)

while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
    scene.world.traverse_visit(camUpdate, scene.world.root)
    c1_bool, c2_bool, p_bool = TRSscreen()

    if c1_bool:
        trsChanges(model_cube, cube1_translate, cube1_rotate, cube1_scale)
    if c2_bool:
        model_cube2 = util.translate(*cube2_translate)  
        model_cube2 = model_cube2 @ util.rotate([1, 0, 0], cube2_rotate[0])  
        model_cube2 = model_cube2 @ util.rotate([0, 1, 0], cube2_rotate[1])  
        model_cube2 = model_cube2 @ util.rotate([0, 0, 1], cube2_rotate[2])  
        model_cube2 = model_cube2 @ util.scale(*cube2_scale)
    if p_bool:
        model_tri = util.translate(*pyramid_translate)  
        model_tri = model_tri @ util.rotate([1, 0, 0], pyramid_rotate[0])  
        model_tri = model_tri @ util.rotate([0, 1, 0], pyramid_rotate[1])  
        model_tri = model_tri @ util.rotate([0, 0, 1], pyramid_rotate[2])  
        model_tri = model_tri @ util.scale(*pyramid_scale)

    view =  gWindow._myCamera
    mvp_terrain = projMat @ view @ model_terrain
    terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain, mat4=True)

    mvp_cube = projMat @ view @ model_cube
    shaderDec4.setUniformVariable(key='modelViewProj', value=mvp_cube, mat4=True)

    mvp_tri = projMat @ view @ model_tri
    shaderDec_tri.setUniformVariable(key='modelViewProj', value=mvp_tri, mat4=True)

    mvp_cube2 = projMat @ view @ model_cube2
    shaderDec_cube2.setUniformVariable(key='modelViewProj', value=mvp_cube2, mat4=True)
    
    scene.render_post()
    
scene.shutdown()
