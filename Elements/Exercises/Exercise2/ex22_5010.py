import numpy as np
import os
import Elements.pyECSS.math_utilities as util
from Elements.pyECSS.Entity import Entity
from Elements.pyECSS.Component import BasicTransform, Camera, RenderMesh
from Elements.pyECSS.System import  TransformSystem, CameraSystem
from Elements.pyGLV.GL.Scene import Scene
from Elements.pyGLV.GUI.Viewer import RenderGLStateSystem, ImGUIecssDecorator

from Elements.pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from Elements.pyGLV.GL.VertexArray import VertexArray
import Elements.utils.normals as norm
from Elements.pyGLV.GL.Textures import Texture, get_texture_faces
from Elements.pyGLV.GL.Textures import get_single_texture_faces

from Elements.definitions import TEXTURE_DIR

from Elements.utils.Shortcuts import displayGUI_text
example_description = \
"This example demonstrates the cube map texture, i.e., \n\
we encapsulate the scene into a huge cube and apply texture to them\n\
creating the illusion of a scenery. \n\
You may move the camera using the mouse or the GUI. \n\
You may see the ECS Scenegraph showing Entities & Components of the scene and \n\
various information about them. Hit ESC OR Close the window to quit." 


winWidth = 1920
winHeight = 1080

Lposition = util.vec(5.0, 2.0, 2.0) #uniform lightpos
Lambientcolor = util.vec(1.0, 1.0, 1.0) #uniform ambient color
Lambientstr = 0.2 #uniform ambientStr
LviewPos = util.vec(2.5, 2.8, 5.0) #uniform viewpos
Lcolor = util.vec(1.0,1.0,1.0)
Lintensity = 0.8
#Material
Mshininess = 0.4 
Mcolor = util.vec(0.8, 0.0, 0.8)

eye = util.vec(1, 0.54, 1.0)
target = util.vec(0.02, 0.14, 0.217)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
projMat = util.perspective(50.0, 1.0, 1.0, 10.0)   

# Scenegraph with Entities, Components
scene = Scene()    
rootEntity = scene.world.createEntity(Entity(name="RooT"))

skybox = scene.world.createEntity(Entity(name="Skybox"))
scene.world.addEntityChild(rootEntity, skybox)
transSkybox = scene.world.addComponent(skybox, BasicTransform(name="transSkybox", trs=util.identity())) #util.identity()
meshSkybox = scene.world.addComponent(skybox, RenderMesh(name="meshSkybox"))



#Cube
minbox = -30
maxbox = 30
vertexSkybox = np.array([
    [minbox, minbox, maxbox, 1.0],
    [minbox, maxbox, maxbox, 1.0],
    [maxbox, maxbox, maxbox, 1.0],
    [maxbox, minbox, maxbox, 1.0], 
    [minbox, minbox, minbox, 1.0], 
    [minbox, maxbox, minbox, 1.0], 
    [maxbox, maxbox, minbox, 1.0], 
    [maxbox, minbox, minbox, 1.0]
],dtype=np.float32)

#index array for Skybox
indexSkybox = np.array((1,0,3, 1,3,2, 
                  2,3,7, 2,7,6,
                  3,0,4, 3,4,7,
                  6,5,1, 6,1,2,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1), np.uint32) 

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

#index Array for Cube
indexCube = np.array((1,0,3, 1,3,2, 
                  2,3,7, 2,7,6,
                  3,0,4, 3,4,7,
                  6,5,1, 6,1,2,
                  4,5,6, 4,6,7,
                  5,4,0, 5,0,1), np.uint32) 


# Generate Sphere Vertices and Indices
def generateSphere(radius, latitudeBands, longitudeBands):
    vertices = []
    indices = []
    uvs = []

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

            u = lon / longitudeBands
            v = 1 - (lat / latitudeBands)
            uvs.append([u, v])

            if lat < latitudeBands and lon < longitudeBands:
                first = lat * (longitudeBands + 1) + lon
                second = first + longitudeBands + 1

                indices.append(first)
                indices.append(second)
                indices.append(first + 1)

                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), np.array(uvs, dtype=np.float32)


vertexSphere, indexSphere, uvSphere = generateSphere(1.0, 20, 20)

def addSphereToScene(name, position):
    node = scene.world.createEntity(Entity(name=name))
    scene.world.addEntityChild(rootEntity, node)
    trans = scene.world.addComponent(node, BasicTransform(name=name + "_trans", trs=util.translate(*position)))
    mesh = scene.world.addComponent(node, RenderMesh(name=name + "_mesh"))
    
    mesh.vertex_attributes.append(vertexSphere)
    mesh.vertex_index.append(indexSphere)
    
    vArray = scene.world.addComponent(node, VertexArray())
    shaderDec = scene.world.addComponent(node, ShaderGLDecorator(Shader(vertex_source = Shader.SIMPLE_TEXTURE_PHONG_VERT, fragment_source=Shader.SIMPLE_TEXTURE_PHONG_FRAG)))
    
    return node, trans, vArray, shaderDec, mesh


# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())


vertexSkybox, indexSkybox, _ = norm.generateUniqueVertices(vertexSkybox,indexSkybox)

meshSkybox.vertex_attributes.append(vertexSkybox)
meshSkybox.vertex_index.append(indexSkybox)
vArraySkybox = scene.world.addComponent(skybox, VertexArray())
shaderSkybox = scene.world.addComponent(skybox, ShaderGLDecorator(Shader(vertex_source = Shader.STATIC_SKYBOX_VERT, fragment_source=Shader.STATIC_SKYBOX_FRAG)))

_, _, _, normals = norm.generateFlatNormalsMesh(vertexSphere , indexSphere)
earth, trans_earth, vArray_earth, shader_earth, mesh_earth = addSphereToScene("earth", [0.0, 0.0, 0.0])
mesh_earth.vertex_attributes.append(normals)
mesh_earth.vertex_attributes.append(uvSphere)
sun, trans_sun, vArray_sun, shader_sun, mesh_sun = addSphereToScene("sun", [2.0, 0.0, 2.0])


# MAIN RENDERING LOOP

running = True
scene.init(imgui=True, windowWidth = winWidth, windowHeight = winHeight, windowTitle = "Elements: Cube Mapping Example", customImGUIdecorator = ImGUIecssDecorator, openGLversion = 4)

# pre-pass scenegraph to initialise all GL context dependent geometry, shader classes
# needs an active GL context
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


eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
projMat = util.perspective(50.0, 1.0, 0.01, 100.0)   

gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

# skybox_texture_locations = TEXTURE_DIR / "Skyboxes" / "Cloudy"
front_img =  "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/front.jpg"
right_img =  "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/right.jpg"
left_img =  "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/left.jpg"
back_img = "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/back.jpg"
bottom_img = "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/bottom.jpg"
top_img = "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/Skyboxes/Stars/top.jpg"

 
mat_img = "/home/exelixis_desk/358_exercises/Elements_HY358/Elements/files/textures/earth.jpg"

face_data = get_texture_faces(front_img,back_img,top_img,bottom_img,left_img,right_img)
face_data_2 = Texture(mat_img)

shaderSkybox.setUniformVariable(key='cubemap', value=face_data, texture3D=True)
shader_earth.setUniformVariable(key='ImageTexture', value=face_data_2, texture=True)

while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(transUpdate, scene.world.root)
    
    view =  gWindow._myCamera # updates view via the imgui

    shader_earth.setUniformVariable(key='Proj', value=projMat, mat4=True)
    shader_earth.setUniformVariable(key='View', value=view, mat4=True)
    shader_earth.setUniformVariable(key='model', value=trans_earth.l2world, mat4=True)
    shader_earth.setUniformVariable(key='ambientColor',value=Lambientcolor,float3=True)
    shader_earth.setUniformVariable(key='ambientStr',value=Lambientstr,float1=True)
    shader_earth.setUniformVariable(key='viewPos',value=LviewPos,float3=True)
    shader_earth.setUniformVariable(key='lightPos',value=Lposition,float3=True)
    shader_earth.setUniformVariable(key='lightColor',value=Lcolor,float3=True)
    shader_earth.setUniformVariable(key='lightIntensity',value=Lintensity,float1=True)
    shader_earth.setUniformVariable(key='shininess',value=Mshininess,float1=True)

    shaderSkybox.setUniformVariable(key='Proj', value=projMat, mat4=True)
    shaderSkybox.setUniformVariable(key='View', value=view, mat4=True)

    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.render_post()
    
scene.shutdown()