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
"This is a scene with a sphere." 

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
orthoCam = scene.world.addComponent(entityCam2, Camera(m, "orthoCam","Camera","500"))

# Generate Sphere Vertices and Indices
def generateSphere(radius, latitudeBands, longitudeBands):
    vertices = []
    indices = []
    whiteColors = []
    ROColors = []
    normalColors = []

    for lat in range(latitudeBands + 1):
        theta = lat * np.pi / latitudeBands
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        for lon in range(longitudeBands + 1):
            phi = lon * 2 * np.pi / longitudeBands
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)

            x = cosPhi * sinTheta
            y = cosTheta
            z = sinPhi * sinTheta

            vertices.append([radius * x, radius * y, radius * z, 1.0])

            t = (y + 1) / 2  
            ROColors.append([1.0, 0.5 * (1 - t) , 0.0, 1.0])

            whiteColors.append([1.0, 1.0, 1.0, 1.0])

            normalColors.append([(x + 1) / 2, (y + 1) / 2, (z + 1) / 2, 1.0])

            if lat < latitudeBands and lon < longitudeBands:
                first = lat * (longitudeBands + 1) + lon
                second = first + longitudeBands + 1

                indices.append(first)
                indices.append(second)
                indices.append(first + 1)

                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    return np.array(vertices, dtype=np.float32), np.array(whiteColors, dtype=np.float32), np.array(ROColors, dtype=np.float32), np.array(normalColors, dtype=np.float32), np.array(indices, dtype=np.uint32)

vertexSphere, whiteColorSphere, ROcolorSphere, normalColorSphere, indexSphere = generateSphere(1.0, 20, 20)

def addSphereToScene(name, position, colorArray):
    node = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, node)
    trans = scene.world.addComponent(node, BasicTransform(name=name + "_trans", trs=util.translate(*position)))
    mesh = scene.world.addComponent(node, RenderMesh(name=name + "_mesh"))
    
    mesh.vertex_attributes.append(vertexSphere)
    mesh.vertex_attributes.append(colorArray)
    mesh.vertex_index.append(indexSphere)
    
    vArray = scene.world.addComponent(node, VertexArray())
    shaderDec = scene.world.addComponent(node, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source = Shader.COLOR_FRAG)))
    
    return node, trans, vArray, shaderDec, mesh

positions = [
    (0, 0, 0),  
    (3, 0, 0),   
    (0, 3, 0),  
    (0, -3, 0),  
    (0, 0, -3)
]

spheres = [
    addSphereToScene("sphere1", positions[0], whiteColorSphere),
    addSphereToScene("sphere2", positions[1], whiteColorSphere),
    addSphereToScene("sphere3", positions[2], ROcolorSphere),
    addSphereToScene("sphere4", positions[3], normalColorSphere),
    addSphereToScene("sphere5", positions[4], normalColorSphere)
]

# mesh4.vertex_attributes.append(vertexSphere)
# mesh4.vertex_attributes.append(whiteColorSphere)
# mesh4.vertex_index.append(indexSphere)
# vArray4 = scene.world.addComponent(node4, VertexArray())
# shaderDec4 = scene.world.addComponent(node4, ShaderGLDecorator(Shader(vertex_source = Shader.COLOR_VERT_MVP, fragment_source=Shader.COLOR_FRAG)))

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

# MAIN RENDERING LOOP

running = True
scene.init(imgui=True, windowWidth = winWidth, windowHeight = winHeight, windowTitle = "Elements: A Working Event Manager", customImGUIdecorator = ImGUIecssDecorator, openGLversion = 4)


scene.world.traverse_visit(initUpdate, scene.world.root)

################### EVENT MANAGER ###################

eManager = scene.world.eventManager
gWindow = scene.renderWindow
gGUI = scene.gContext

renderGLEventActuator = RenderGLStateSystem()

roColorCB = False
normalColorCB = False
fiveSpheresCB = False

eManager._subscribers['OnUpdateWireframe'] = gWindow
eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
eManager._subscribers['OnUpdateCamera'] = gWindow 
eManager._actuators['OnUpdateCamera'] = renderGLEventActuator


eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)

projMat = util.perspective(50.0, 1.0, 0.01, 10.0) 

gWindow._myCamera = view 

def ChangeScene():
    global roColorCB
    global normalColorCB
    global fiveSpheresCB

    print("Changing Scene...")

    if roColorCB: 
        spheres[0][4].vertex_attributes[1] = ROcolorSphere
    elif normalColorCB:
        spheres[0][4].vertex_attributes[1] = normalColorSphere
    else:
        spheres[0][4].vertex_attributes[1] = whiteColorSphere

    if fiveSpheresCB:
        pass

def CheckBoxGUI():
    global roColorCB
    global normalColorCB
    global fiveSpheresCB
    imgui.begin("Actions")

    if imgui.checkbox("RO Color", roColorCB)[0]:
        roColorCB = not roColorCB
        ChangeScene()

    if imgui.checkbox("Normal Color", normalColorCB)[0]:
        normalColorCB = not normalColorCB
        ChangeScene()

    if imgui.checkbox("Five Spheres", fiveSpheresCB)[0]:
        fiveSpheresCB = not fiveSpheresCB
        ChangeScene()

    imgui.end()

model_sphere = [
    spheres[0][1].trs,
    spheres[1][1].trs,
    spheres[2][1].trs,
    spheres[3][1].trs,
    spheres[4][1].trs
]

while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
    scene.world.traverse_visit(camUpdate, scene.world.root)

    CheckBoxGUI()
    view =  gWindow._myCamera

    spheres[0][2].update() 
    mvp_sphere = projMat @ view @ model_sphere[0]
    spheres[0][3].setUniformVariable(key='modelViewProj', value=mvp_sphere, mat4=True)

    for i in range(1, 5):
        spheres[i][2].init()
        mvp_sphere = projMat @ view @ model_sphere[i]
        spheres[i][3].setUniformVariable(key='modelViewProj', value=mvp_sphere, mat4=True)

    sphere_visibility = [True] * 5

    if fiveSpheresCB:
        for i in range(1, 5):
            if not sphere_visibility[i]:
                spheres[i][3].enableShader()
                sphere_visibility[i] = True

    else:
        for i in range(1, 5):
            if sphere_visibility[i]: 
                spheres[i][3].disableShader()  
                sphere_visibility[i] = False


    scene.render_post()
    
scene.shutdown()
