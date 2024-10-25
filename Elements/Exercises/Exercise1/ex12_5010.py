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

winWidth = 1920
winHeight = 1080

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
orthoCam = scene.world.addComponent(entityCam2, Camera(m, "orthoCam","Camera","500"))

# a simple triangle
vertexData = np.array([
    [-0.5, -0.5, -0.5, 1.0],
    [-0.5, -0.5, 0.5, 1.0],
    [0.5, -0.5, -0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0],
    [0.0, 1.0, 0.0, 1.0]
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

vertexHouse = np.array([
    [-0.5, -0.5, 0.5, 1.0],
    [-0.5, 0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0], 
    [-0.5, -0.5, -0.5, 1.0], 
    [-0.5, 0.5, -0.5, 1.0], 
    [0.5, 0.5, -0.5, 1.0], 
    [0.5, -0.5, -0.5, 1.0],
    [-0.5, 0.5, -0.5, 1.0],
    [-0.5, 0.5, 0.5, 1.0],
    [0.5, 0.5, -0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [0.0, 2.0, 0.0, 1.0]
],dtype=np.float32)
colorHouse = np.array([
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.5, 0, 1.0, 0.0]
], dtype=np.float32)

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
                  6,5,1, 6,1,2,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1,
                  8,9,12,  8,10,12,
                  9,11,12,  10,11,12,
                  8,9,10,  9,10,11), np.uint32)

def addObjectToScene(name, vertecies, index, colorArray, position, addAsChild=True):
    node = scene.world.createEntity(Entity(name=name))
    if addAsChild:
        scene.world.addEntityChild(rootEntity, node)
    trans = scene.world.addComponent(node, BasicTransform(name=name + "_trans", trs=util.translate(*position)))
    mesh = scene.world.addComponent(node, RenderMesh(name=name + "_mesh"))
    
    mesh.vertex_attributes.append(vertecies)
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

node4, trans4, vArray4, shaderDec4, mesh4 = addObjectToScene("node4", vertexCube, indexCube, colorCube, (0,0.5,0))
cube2, trans_cube2, vArray_cube2, shaderDec_cube2, mesh_cube2 = addObjectToScene("cube2", vertexCube, indexCube, colorCube, (-2,0.5,-2))
pyramid, trans_tri, vArray_tri, shaderDec_tri, mesh_tri = addObjectToScene("pyramid", vertexData, indexData, colorVertexData, (0,1.5,0))
house, trans_house, vArray_house, shaderDec_house, mesh_house = addObjectToScene("house", vertexHouse, indexHouse, colorHouse, (0,0.5,0))

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

vertexAxes = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
],dtype=np.float32) 
colorAxes = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)
indexAxes = np.array((0,1,2,3,4,5), np.uint32)
axes = scene.world.createEntity(Entity(name="axes"))
scene.world.addEntityChild(rootEntity, axes)
axes_trans = scene.world.addComponent(axes, BasicTransform(name="axes_trans", trs=util.translate(0.0, 0.001, 0.0)))
axes_mesh = scene.world.addComponent(axes, RenderMesh(name="axes_mesh"))
axes_mesh.vertex_attributes.append(vertexAxes) 
axes_mesh.vertex_attributes.append(colorAxes)
axes_mesh.vertex_index.append(indexAxes)
axes_vArray = scene.world.addComponent(axes, VertexArray(primitive=GL_LINES))
axes_shader = scene.world.addComponent(axes, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))


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

cube2_translate = [-2.0, 0.5, -2.0]
cube2_rotate = [0.0, 0.0, 0.0]
cube2_scale = [1.0, 1.0, 1.0]

pyramid_translate = [0.0, 1.5, 0.0]
pyramid_rotate = [0.0, 0.0, 0.0]
pyramid_scale = [1.0, 1.0, 1.0]

house_translate = [0.0, 0.0, 0.0]
house_rotate = [0.0, 0.0, 0.0]
house_scale = [0.0, 0.0, 0.0]

House = False
showAxis = True
showGrid = True
showCube1 = True
showCube2 = True
showPyramid = True
showHouse = True

def calcTRShouse():
    global cube1_translate
    global cube2_translate
    global pyramid_translate
    global house_translate
    global house_rotate 
    global house_scale

    house1, house2 = checkIfHouse()

    if house1:
        house_translate = cube1_translate
        house_rotate = cube1_rotate
        house_scale = cube1_scale

    if house2:
        house_translate = cube2_translate
        house_rotate = cube2_rotate
        house_scale = cube2_scale

def checkIfHouse():
    global cube1_translate
    global cube1_rotate
    global cube1_scale
    global cube2_translate
    global cube2_rotate
    global cube2_scale
    global pyramid_translate
    global pyramid_rotate
    global pyramid_scale
    

    if cube1_translate[0] == pyramid_translate[0] and cube1_translate[1] + 1 == pyramid_translate[1] and cube1_translate[2] == pyramid_translate[2] and cube1_rotate == pyramid_rotate and cube1_scale == pyramid_scale:
        return True, False
    
    if cube2_translate[0] == pyramid_translate[0] and cube2_translate[1] + 1 == pyramid_translate[1] and cube2_translate[2] == pyramid_translate[2] and cube2_rotate == pyramid_rotate and cube2_scale == pyramid_scale:
        return False, True

    return False, False

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
    global house_translate
    global house_rotate 
    global house_scale

    global House
    global showAxis
    global showGrid
    global showCube1
    global showCube2
    global showPyramid
    global showHouse

    global terrain
    global node4
    global cube2
    global pyramid
    global axes
    global house

    c1_changed, c2_changed, p_changed, h_changed = False, False, False, False

    imgui.begin("TRS")

    house1, house2 = checkIfHouse()

    if imgui.checkbox("Show Axis", showAxis)[0]:
        if showAxis:
            rootEntity.remove(axes)
        else:
            rootEntity.add(axes)
        showAxis = not showAxis

    if imgui.checkbox("Show Grid", showGrid)[0]:
        if showGrid:
            rootEntity.remove(terrain)
        else:
            rootEntity.add(terrain)
        showGrid = not showGrid

    if house1 or house2:
        if imgui.checkbox("Activate House", House)[0]:
            House = not House
            if House:
                calcTRShouse()
                rootEntity.add(house)
                rootEntity.remove(pyramid)
                if house1:
                    rootEntity.remove(node4)
                if house2:
                    rootEntity.remove(cube2)

                h_changed = True
            else:
                rootEntity.remove(house)
                rootEntity.add(pyramid)
                if house1:
                    rootEntity.add(node4)
                if house2:
                    rootEntity.add(cube2)

    c1_changed = c2_changed = p_changed = False

    if imgui.begin_tab_bar("MyTabBar"):

        if not House or not house1:
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

                if imgui.checkbox("Show Cube1", showCube1)[0]:
                    if showCube1:
                        rootEntity.remove(node4)
                    else:
                        rootEntity.add(node4)
                    showCube1 = not showCube1

                c1_changed = c1_changed_t or c1_changed_r or c1_changed_s
                imgui.end_tab_item()

        if not House or not house2:
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

                if imgui.checkbox("Show Cube2", showCube2)[0]:
                    if showCube2:
                        rootEntity.remove(cube2)
                    else:
                        rootEntity.add(cube2)
                    showCube2 = not showCube2

                c2_changed = c2_changed_t or c2_changed_r or c2_changed_s
                imgui.end_tab_item()

        if not House:
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

                if imgui.checkbox("Show Pyramid", showPyramid)[0]:
                    if showPyramid:
                        rootEntity.remove(pyramid)
                    else:
                        rootEntity.add(pyramid)
                    showPyramid = not showPyramid

                p_changed = p_changed_t or p_changed_r or p_changed_s
                imgui.end_tab_item()

        if House:
            if imgui.begin_tab_item("House").selected:
                h_changed_t, house_translate = imgui.drag_float3(
                    "Translate", *house_translate
                )
                h_changed_r, house_rotate = imgui.drag_float3(
                    "Rotate", *house_rotate
                )
                h_changed_s, house_scale = imgui.drag_float3(
                    "Scale", *house_scale
                )

                if imgui.checkbox("Show House", showHouse)[0]:
                    if showHouse:
                        rootEntity.remove(house)
                    else:
                        rootEntity.add(house)
                    showHouse = not showHouse

                h_changed = h_changed_t or h_changed_r or h_changed_s
                imgui.end_tab_item()

    imgui.end_tab_bar()

    imgui.end()

    return c1_changed, c2_changed, p_changed, h_changed



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
model_axes = axes_trans.trs
model_house = trans_house.trs

def trsChanges(translate, rotate, scale):
    model = util.translate(*translate)  
    model = model @ util.rotate([1, 0, 0], rotate[0])  
    model = model @ util.rotate([0, 1, 0], rotate[1])  
    model = model @ util.rotate([0, 0, 1], rotate[2])  
    model = model @ util.scale(*scale)

    return model

rootEntity.remove(house)
while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
    scene.world.traverse_visit(camUpdate, scene.world.root)
    c1_bool, c2_bool, p_bool, h_bool = TRSscreen()

    if c1_bool:
        model_cube = trsChanges(cube1_translate, cube1_rotate, cube1_scale)
    if c2_bool:
        model_cube2 = trsChanges(cube2_translate, cube2_rotate, cube2_scale)
    if p_bool:
        model_tri = trsChanges(pyramid_translate, pyramid_rotate, pyramid_scale)
    if h_bool:
        model_house = trsChanges(house_translate, house_rotate, house_scale)

    view =  gWindow._myCamera
    mvp_terrain = projMat @ view @ model_terrain
    terrain_shader.setUniformVariable(key='modelViewProj', value=mvp_terrain, mat4=True)
    mvp_axes = projMat @ view @ model_axes
    axes_shader.setUniformVariable(key='modelViewProj', value=mvp_axes, mat4=True)

    mvp_cube = projMat @ view @ model_cube
    shaderDec4.setUniformVariable(key='modelViewProj', value=mvp_cube, mat4=True)

    mvp_tri = projMat @ view @ model_tri
    shaderDec_tri.setUniformVariable(key='modelViewProj', value=mvp_tri, mat4=True)

    mvp_cube2 = projMat @ view @ model_cube2
    shaderDec_cube2.setUniformVariable(key='modelViewProj', value=mvp_cube2, mat4=True)

    mvp_house = projMat @ view @ model_house
    shaderDec_house.setUniformVariable(key='modelViewProj', value=mvp_house, mat4=True)
    
    scene.render_post()
    
scene.shutdown()
