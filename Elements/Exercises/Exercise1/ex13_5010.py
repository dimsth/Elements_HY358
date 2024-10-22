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
"This is a scene with a trinagle, 2 lines and 3 points." 

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
    [2.0, -0.5, 1.0, 1.0],
    [2.0, -0.5, 2.0, 1.0],
    [1.5, 1.0, 1.5, 1.0]
],dtype=np.float32) 
colorVertexData = np.array([
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0]
], dtype=np.float32)

vertexLine1 = np.array([
    [0.5, 1.0, 0.5],
    [0.5, 2.0, 1.0]
], dtype=np.float32)
vertexLine2 = np.array([
    [1.0, 1.0, -0.5],
    [-0.5, 0.5, 0.75]
], dtype=np.float32)
colorLines = np.array([
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)


vertexPoint1 = np.array([
    [0.75, 2.0, 1.2]
], dtype=np.float32)
vertexPoint2 = np.array([
    [-1.5, 0.75, 0.5]
], dtype=np.float32)
vertexPoint3 = np.array([
    [-0.25, 0.5, 0.35]
], dtype=np.float32)
colorPoints = np.array([
    [0.0, 1.0, 1.0, 1.0]
], dtype=np.float32)

#index arrays for above vertex Arrays
indexTri = np.array((0,1,2), np.uint32)
indexLines = np.array((0,1,0), np.uint32)
indexPoints = np.array((0,0,0), np.uint32)

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

line1, trans_l1, vArray_l1, shaderDec_l1, mesh_l1 = addObjectToScene("line1", vertexLine1, indexLines, colorLines)
line2, trans_l2, vArray_l2, shaderDec_l2, mesh_l2 = addObjectToScene("line2", vertexLine2, indexLines, colorLines)
tringle, trans_tri, vArray_tri, shaderDec_tri, mesh_tri = addObjectToScene("triangle", vertexData, indexTri, colorVertexData)
point1, trans_p1, vArray_p1, shaderDec_p1, mesh_p1 = addObjectToScene("point1", vertexPoint1, indexPoints, colorPoints)
point2, trans_p2, vArray_p2, shaderDec_p2, mesh_p2 = addObjectToScene("point2", vertexPoint2, indexPoints, colorPoints)
point3, trans_p3, vArray_p3, shaderDec_p3, mesh_p3 = addObjectToScene("point3", vertexPoint3, indexPoints, colorPoints)


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

def Objectsscreen():
    imgui.begin("Objects")
    if imgui.begin_tab_bar("MyTabBar"):

        if imgui.begin_tab_item("Triangle").selected:
            imgui.text("Object Space\n\n")

            imgui.text("___________________________________\n\n\nWorld Space\n\n")

            imgui.text("___________________________________\n\n\nEye Space\n\n")

            imgui.text("___________________________________\n\n\nClip Space\n\n")

            imgui.text("___________________________________\n\n\nNormaliza Space\n\n")

            imgui.text("___________________________________\n\n\nWindow Space\n\n")

            imgui.end_tab_item()

        if imgui.begin_tab_item("Line1").selected:
            imgui.text("Object Space")
          
            imgui.end_tab_item()

        if imgui.begin_tab_item("Line2").selected:
          
            imgui.end_tab_item()

        if imgui.begin_tab_item("Point1").selected:
          
            imgui.end_tab_item()

        if imgui.begin_tab_item("Point2").selected:
          
            imgui.end_tab_item()

        if imgui.begin_tab_item("Point3").selected:
          
            imgui.end_tab_item()

    imgui.end_tab_bar()

    imgui.end()


eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)

projMat = util.perspective(50.0, 1.0, 0.01, 10.0)
gWindow._myCamera = view
model_terrain = terrain.getChild(0).trs

model_tri = trans_tri.trs
model_l1 = trans_l1.trs
model_l2 = trans_l2.trs
model_p1 = trans_p1.trs
model_p2 = trans_p2.trs
model_p3 = trans_p3.trs

while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
    scene.world.traverse_visit(camUpdate, scene.world.root)
    Objectsscreen()

    view =  gWindow._myCamera
    mvp_terrain = projMat @ view @ model_terrain
    terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain, mat4=True)

    mvp_tri = projMat @ view @ model_tri
    shaderDec_tri.setUniformVariable(key='modelViewProj', value=mvp_tri, mat4=True)

    mvp_l1 = projMat @ view @ model_l1
    shaderDec_l1.setUniformVariable(key='modelViewProj', value=mvp_l1, mat4=True)

    mvp_l2 = projMat @ view @ model_l1
    shaderDec_l2.setUniformVariable(key='modelViewProj', value=mvp_l2, mat4=True)
    
    mvp_p1 = projMat @ view @ model_p1
    shaderDec_p1.setUniformVariable(key='modelViewProj', value=mvp_p1, mat4=True)

    mvp_p2 = projMat @ view @ model_p2
    shaderDec_p2.setUniformVariable(key='modelViewProj', value=mvp_p2, mat4=True)

    mvp_p3 = projMat @ view @ model_p3
    shaderDec_p3.setUniformVariable(key='modelViewProj', value=mvp_p3, mat4=True)
    
    scene.render_post()
    
scene.shutdown()
