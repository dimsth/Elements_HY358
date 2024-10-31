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

import Elements.utils.normals as norm

from Elements.utils.Shortcuts import displayGUI_text
example_description = \
"This is a scene with 2 cube, a trinagle and house." 

FRAG_PHONG_WITH_DIFFUSE = """
        #version 410

        in vec4 pos;
        in vec4 color;
        in vec3 normal;

        out vec4 outputColor;

        // Phong products
        uniform vec3 ambientColor;
        uniform float ambientStr;

        // Lighting 
        uniform vec3 viewPos;
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform float lightIntensity;

        // Material
        uniform float shininess;
        uniform vec3 matColor;
        uniform vec3 diffuseColor;

        void main()
        {
            vec3 norm = normalize(normal);
            vec3 lightDir = normalize(lightPos - pos.xyz);
            vec3 viewDir = normalize(viewPos - pos.xyz);
            vec3 reflectDir = reflect(-lightDir, norm);
            

            // Ambient
            vec3 ambientProduct = ambientStr * ambientColor;

            // Diffuse
            float diffuseStr = max(dot(norm, lightDir), 0.0);
            vec3 diffuseProduct = diffuseStr * lightColor * diffuseColor;

            // Specular
            float specularStr = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specularProduct = shininess * specularStr * matColor;
            
            vec3 result = (ambientProduct + (diffuseProduct + specularProduct) * lightIntensity) * color.xyz;
            outputColor = vec4(result, 1);
        }
    """

winWidth = 1920
winHeight = 1080

scene = Scene()    

#Light
Lposition = util.vec(5.0, 5.5, 5.0) #uniform lightpos
Lambientcolor = util.vec(1.0, 1.0, 1.0) #uniform ambient color
Lambientstr = 0.3 #uniform ambientStr
LviewPos = util.vec(2.5, 2.8, 5.0) #uniform viewpos
Lcolor = util.vec(1.0,1.0,1.0)
Lintensity = 1.0
#Material
Mshininess = 0.4 
Mcolor = util.vec(0.8, 0.0, 0.8)
Mdiffuse = util.vec(1.0, 0.5, 0.0)

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

def addObjectToScene(name, vertecies, index, colorArray, position, addAsChild=True):
    node = scene.world.createEntity(Entity(name=name))
    if addAsChild:
        scene.world.addEntityChild(rootEntity, node)
    trans = scene.world.addComponent(node, BasicTransform(name=name + "_trans", trs=util.translate(*position)))
    mesh = scene.world.addComponent(node, RenderMesh(name=name + "_mesh"))

    vertices, indices, colors, normals = norm.generateSmoothNormalsMesh(vertecies , index, colorArray)

    mesh.vertex_attributes.append(vertices)
    mesh.vertex_attributes.append(colors)
    mesh.vertex_attributes.append(normals)
    mesh.vertex_index.append(indices)
    
    vArray = scene.world.addComponent(node, VertexArray())
    shaderDec = scene.world.addComponent(node, ShaderGLDecorator(Shader(vertex_source = Shader.VERT_PHONG_MVP, fragment_source=FRAG_PHONG_WITH_DIFFUSE)))
    
    return node, trans, vArray, shaderDec, mesh

# Systems
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())


## ADD OBJECTS ##

node4, trans4, vArray4, shaderDec4, mesh4 = addObjectToScene("node4", vertexCube, indexCube, colorCube, (0,0.5,0))
cube2, trans_cube2, vArray_cube2, shaderDec_cube2, mesh_cube2 = addObjectToScene("cube2", vertexCube, indexCube, colorCube, (-1.25,0.5,0.75))
pyramid, trans_tri, vArray_tri, shaderDec_tri, mesh_tri = addObjectToScene("pyramid", vertexData, indexData, colorVertexData, (1.0,0.5,1.25))

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

cube1_color = [1.0, 0.5, 0.0, 1.0]
cube2_color = [1.0, 0.5, 0.0, 1.0]
pyramid_color = [0.5, 0.0, 1.0, 1.0]

def Colorscreen():
    global cube1_color
    global cube2_color
    global pyramid_color

    c1_changed, c2_changed, p_changed = False, False, False

    imgui.begin("Color")

    c1_changed = c2_changed = p_changed = False

    c1_changed, cube1_color = imgui.color_edit4(
        "Cube1 Color", *cube1_color
    )

    c2_changed, cube2_color = imgui.color_edit4(
        "Cube2 Color", *cube2_color
    )

    p_changed, pyramid_color = imgui.color_edit4(
        "Pyramid Color", *pyramid_color
    )

    imgui.end()

    return c1_changed, c2_changed, p_changed

def Lightscreen():
    global Lambientcolor
    global Lambientstr
    global LviewPos
    global Lposition
    global Lcolor
    global Lintensity
    global Mshininess
    global Mcolor
    global Mdiffuse

    imgui.begin("Light")
    _, Lposition = imgui.drag_float3(
        "Position", *Lposition
    )

    _, Lambientcolor = imgui.drag_float3(
        "Ambient Color", *Lambientcolor
    )

    _, Lambientstr = imgui.drag_float(
        "Ambient Strength", Lambientstr
    )

    _, LviewPos = imgui.drag_float3(
        "View Position", *LviewPos
    )

    _, Lcolor = imgui.drag_float3(
        "Color", *Lcolor
    )

    _, Lintensity = imgui.drag_float(
        "Intensity", Lintensity
    )

    _, Mshininess = imgui.drag_float(
        "Shininess", Mshininess
    )

    _, Mcolor = imgui.drag_float3(
        "Matierial Color", *Mcolor
    )

    _, Mdiffuse = imgui.drag_float3(
        "Matierial Diffuse", *Mdiffuse
    )

    imgui.end()



eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)

projMat = util.perspective(50.0, 1.0, 0.01, 10.0)
gWindow._myCamera = view

model_cube = trans4.trs
model_tri = trans_tri.trs
model_cube2 = trans_cube2.trs

def setLightOnShader(shader, mvp, model):
    global Lambientcolor, Lambientstr, LviewPos, Lposition, Lcolor, Lintensity, Mshininess, Mcolor, Mdiffuse
    
    try:
        shader.setUniformVariable(key='modelViewProj', value=mvp, mat4=True)
        shader.setUniformVariable(key='model',value=model,mat4=True)
        shader.setUniformVariable(key='ambientColor', value=Lambientcolor, float3=True)
        shader.setUniformVariable(key='ambientStr', value=Lambientstr, float1=True)
        shader.setUniformVariable(key='viewPos', value=LviewPos, float3=True)
        shader.setUniformVariable(key='lightPos', value=Lposition, float3=True)
        shader.setUniformVariable(key='lightColor', value=Lcolor, float3=True)
        shader.setUniformVariable(key='lightIntensity', value=Lintensity, float1=True)
        shader.setUniformVariable(key='shininess', value=Mshininess, float1=True)
        shader.setUniformVariable(key='matColor', value=Mcolor, float3=True)
        shader.setUniformVariable(key='diffuseColor', value=Mdiffuse, float3=True)
    except Exception as e:
        print(f"Error setting shader uniforms: {e}")


while running:
    running = scene.render()
    displayGUI_text(example_description)
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.world.traverse_visit_pre_camera(camUpdate, orthoCam)
    scene.world.traverse_visit(camUpdate, scene.world.root)
    c1_bool, c2_bool, p_bool = Colorscreen()
    Lightscreen()

    if c1_bool:
        mesh4.vertex_attributes[1] = np.array([cube1_color] * len(mesh4.vertex_attributes[1]), dtype=np.float32)
        vArray4.init()
    if c2_bool:
        mesh_cube2.vertex_attributes[1] = np.array([cube2_color] * len(mesh_cube2.vertex_attributes[1]), dtype=np.float32)
        vArray_cube2.init()
    if p_bool:
        mesh_tri.vertex_attributes[1] = np.array([pyramid_color] * len(mesh_tri.vertex_attributes[1]), dtype=np.float32)
        vArray_tri.init()


    view =  gWindow._myCamera

    mvp_cube = projMat @ view @ model_cube
    setLightOnShader(shaderDec4, mvp_cube, model_cube)

    mvp_tri = projMat @ view @ model_tri
    setLightOnShader(shaderDec_tri, mvp_tri, model_tri)
    
    mvp_cube2 = projMat @ view @ model_cube2
    setLightOnShader(shaderDec_cube2, mvp_cube2, model_cube2)

    scene.render_post()
    
scene.shutdown()
