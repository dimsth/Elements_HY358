# from __future__         import annotations
import numpy as np
import imgui
import OpenGL.GL as gl;
import Elements.pyECSS.math_utilities as util
from Elements.pyECSS.System import  TransformSystem, CameraSystem
from Elements.pyECSS.Entity import Entity
from Elements.pyECSS.Component import BasicTransform,  RenderMesh
from Elements.pyECSS.Event import Event

from Elements.pyGLV.GUI.Viewer import RenderGLStateSystem, ImGUIecssDecorator
from Elements.pyGLV.GL.Shader import InitGLShaderSystem, Shader, ShaderGLDecorator, RenderGLShaderSystem
from Elements.pyGLV.GL.VertexArray import VertexArray
from Elements.pyGLV.GL.Scene import Scene


##########################################################
#########################################################

NumVertices     = 200
index_pyr       = 0

points = np.arange(NumVertices * 4, dtype=np.float32)
points = points.reshape((NumVertices, 4))
points = np.zeros_like(points) #Return an array of zeros with the same shape and type as a given array.

colors = np.arange(NumVertices * 4, dtype=np.float32)
colors = colors.reshape((NumVertices, 4))
colors = np.zeros_like(colors)

weights = np.arange(NumVertices * 3, dtype=np.float32)
weights = weights.reshape((NumVertices, 3))
weights = np.zeros_like(weights)

manos_animation_vertexShader = """
    #version 410
    layout (location=0) in vec4 vPosition;
    layout (location=1) in vec4 vColor;
    layout (location=2) in vec3 weights;
    out vec4 color;

    uniform mat4 M0;
    uniform mat4 M1;
    uniform mat4 M2;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {

        mat4 M = (weights[0] * M0) + (weights[1] * M1) + (weights[2] * M2);
        gl_Position = projection * view * model *  M  * vPosition;
        color = vColor;
    }

    """

fragmentShader = """
    #version 410
    in vec4 color;
    out vec4 outputColour;
    void main()
    {
        outputColour = color;
    }
    """


#Simple Cube
vertices = np.array([
    [-0.5, -0.5, 0.5, 1.0],
    [0.5, -0.5, 0.5, 1.0], 
    [-0.5, -0.5, -0.5, 1.0],
    [0.5, -0.5, -0.5, 1.0],

    [-0.5, 0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5, 1.0],
    [-0.5, 0.5, -0.5, 1.0], 
    [0.5, 0.5, -0.5, 1.0], 

    [-0.5, 1.5, 0.5, 1.0],
    [0.5, 1.5, 0.5, 1.0],
    [-0.5, 1.5, -0.5, 1.0], 
    [0.5, 1.5, -0.5, 1.0], 

    [-0.5, 2.5, 0.5, 1.0],
    [0.5, 2.5, 0.5, 1.0],
    [-0.5, 2.5, -0.5, 1.0], 
    [0.5, 2.5, -0.5, 1.0], 

    [-0.5, 3.5, 0.5, 1.0],
    [0.5, 3.5, 0.5, 1.0],
    [-0.5, 3.5, -0.5, 1.0], 
    [0.5, 3.5, -0.5, 1.0], 
],dtype=np.float32) 

vertex_colors = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.5, 1.0, 1.0],
    [0.0, 0.5, 1.0, 1.0],
    [0.0, 0.5, 1.0, 1.0],
    [0.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float32)

"""
joint_weights = np.array([
    (1, 0.0, 0.0),    #0
    (0, 1.0, 0.0),    #1
    (0, 1.0, 0.0),    #2
    (1, 0.0, 0.0),    #3
    (1, 0.0, 0.0),    #4
    (0, 1.0, 0.0),    #5
    (0, 1.0, 0.0),    #6
    (1, 0.0, 0.0),    #7
    (0, 0, 1.0),    #8
    (0, 0, 1.0),    #9
    (0, 0, 1.0),    #10
    (0, 0, 1.0),    #11
    (0, 0, 1.0),    #12
], dtype=np.float32)
"""
joints = np.array([
    [0.0, 0.0, 0.0],  # Joint 0
    [0.0, 1.0, 0.0],  # Joint 1
    [0.0, 2.0, 0.0]   # Joint 2
])

# Compute weights for each vertex
joint_weights = np.zeros((len(vertices), 3), dtype=np.float32)

for i, vertex in enumerate(vertices[:, :3]):  # Ignore the w component
    distances = np.linalg.norm(vertex - joints, axis=1)  # Compute distances to all joints
    inverse_distances = 1.0 / distances  # Compute inverse distances
    joint_weights[i] = inverse_distances / np.sum(inverse_distances)

def quad(a, b, c, d):
    """create a quad out of four coordinates

    Args:
        a ([int]): [index to first vertex]
        b ([int]): [index to second vertex]
        c ([int]): [index to third vertex]
        d ([int]): [index to fourth vertex]
    """

    tri(a, b, c)
    tri(a, c, d)

def tri(a, b, c): 
    """create a triangle out of three coordinates

    Args:
        a ([int]): [index to first vertex]
        b ([int]): [index to second vertex]
        c ([int]): [index to third vertex]
    """
    global index_pyr
    global colors
    global points
    global vertex_colors
    global vertices

    colors[index_pyr] = vertex_colors[a]
    points[index_pyr] = vertices[a]
    weights[index_pyr] = joint_weights[a]
    index_pyr += 1
    colors[index_pyr] = vertex_colors[b]
    points[index_pyr] = vertices[b]
    weights[index_pyr] = joint_weights[b]
    index_pyr += 1
    colors[index_pyr] = vertex_colors[c]
    points[index_pyr] = vertices[c]
    weights[index_pyr] = joint_weights[c]
    index_pyr += 1

def initCandle():
    # First cube (base)
    quad(0, 1, 5, 4)  # Front face
    quad(1, 3, 7, 5)  # Right face
    quad(3, 2, 6, 7)  # Back face
    quad(2, 0, 4, 6)  # Left face
    quad(4, 5, 7, 6)  # Top face
    quad(0, 1, 3, 2)  # Bottom face

    # Second cube (stacked on the first)
    quad(4, 5, 9, 8)  # Front face
    quad(5, 7, 11, 9)  # Right face
    quad(7, 6, 10, 11)  # Back face
    quad(6, 4, 8, 10)  # Left face
    quad(8, 9, 11, 10)  # Top face

    # Third cube (stacked on the second)
    quad(8, 9, 13, 12)  # Front face
    quad(9, 11, 15, 13)  # Right face
    quad(11, 10, 14, 15)  # Back face
    quad(10, 8, 12, 14)  # Left face
    quad(12, 13, 15, 14)  # Top face

    # Fourth cube (stacked on the third)
    quad(12, 13, 17, 16)  # Front face
    quad(13, 15, 19, 17)  # Right face
    quad(15, 14, 18, 19)  # Back face
    quad(14, 12, 16, 18)  # Left face
    quad(16, 17, 19, 18)  # Top face

angle_joint1 = 0.0
angle_joint2 = 0.0
angle_joint3 = 0.0
animation = False
frame = 0

def displayGUI():
    global angle_joint1, angle_joint2, angle_joint3
    global animation
    global frame
    imgui.begin("Joint Controls")
    _, angle_joint1 = imgui.drag_float("Joint 1 Angle", angle_joint1, 0.1)
    _, angle_joint2 = imgui.drag_float("Joint 2 Angle", angle_joint2, 0.1)

    if imgui.checkbox("Animation", animation)[0]:
        if not animation:
            frame = 0
        animation = not animation

    imgui.end()

def displayCandle(candle_shader):
    global view, projMat, angle_joint1, angle_joint2, angle_joint3

    M0 = util.identity()
    M1 = util.translate(0, 1, 0)
    M2 = util.translate(0, 2, 0)

    B0_inv = np.linalg.inv(M0)
    B1_inv = np.linalg.inv(M1)
    B2_inv = np.linalg.inv(M2)

    M0_prime = util.identity()
    M1_prime = util.translate(0, 1, 0) @ util.rotate(axis=util.vec(1, 0, 0), angle=angle_joint1)
    M2_prime = util.translate(0, 2, 0) @ util.rotate(axis=util.vec(0, 1, 0), angle=angle_joint2)

    model = util.scale(0.5)

    candle_shader.setUniformVariable(key='M0', value=M0_prime @ B0_inv, mat4=True)
    candle_shader.setUniformVariable(key='M1', value=M1_prime @ B1_inv, mat4=True)
    candle_shader.setUniformVariable(key='M2', value=M2_prime @ B2_inv, mat4=True)
    candle_shader.setUniformVariable(key='model', value=model, mat4=True)
    candle_shader.setUniformVariable(key='view', value=view, mat4=True)
    candle_shader.setUniformVariable(key='projection', value=projMat, mat4=True)


winWidth = 1920
winHeight = 1080
scene = Scene()    

# Initialize Systems used for this script
transUpdate = scene.world.createSystem(TransformSystem("transUpdate", "TransformSystem", "001"))
camUpdate = scene.world.createSystem(CameraSystem("camUpdate", "CameraUpdate", "200"))
renderUpdate = scene.world.createSystem(RenderGLShaderSystem())
initUpdate = scene.world.createSystem(InitGLShaderSystem())

# Scenegraph with Entities, Components
rootEntity = scene.world.createEntity(Entity(name="Root"))

# Add candle
candle = scene.world.createEntity(Entity(name="candle"))
scene.world.addEntityChild(rootEntity, candle)
candle_trans = scene.world.addComponent(candle, BasicTransform(name="candle_trans", trs=util.identity()))
candle_mesh = scene.world.addComponent(candle, RenderMesh(name="candle_mesh"))
scene.world.addComponent(candle, VertexArray())


candle_shader = scene.world.addComponent(candle, ShaderGLDecorator(
    Shader(vertex_source=manos_animation_vertexShader, fragment_source=fragmentShader)))

initCandle()

candle_mesh.vertex_attributes.append(points)
candle_mesh.vertex_attributes.append(colors)
candle_mesh.vertex_attributes.append(weights)
indexCube = np.array((range(index_pyr)), np.uint32) 
candle_mesh.vertex_index.append(indexCube)

# MAIN RENDERING LOOP
running = True
scene.init(imgui=True, windowWidth = winWidth, windowHeight = winHeight, windowTitle = "Elements: A CameraSystem Example", customImGUIdecorator = ImGUIecssDecorator)

scene.world.traverse_visit(initUpdate, rootEntity)

eManager = scene.world.eventManager
gWindow = scene.renderWindow
gGUI = scene.gContext

#simple Event actuator System
renderGLEventActuator = RenderGLStateSystem()


#setup Events and add them to the EventManager
updateTRS = Event(name="OnUpdateTRS", id=100, value=None)
updateBackground = Event(name="OnUpdateBackground", id=200, value=None)
eManager._events[updateTRS.name] = updateTRS
eManager._events[updateBackground.name] = updateBackground


eManager._subscribers[updateTRS.name] = gGUI
eManager._subscribers[updateBackground.name] = gGUI

eManager._subscribers['OnUpdateWireframe'] = gWindow
eManager._actuators['OnUpdateWireframe'] = renderGLEventActuator
eManager._subscribers['OnUpdateCamera'] = gWindow
eManager._actuators['OnUpdateCamera'] = renderGLEventActuator


# Add RenderWindow to the EventManager publishers
eManager._publishers[updateBackground.name] = gGUI


projMat = util.perspective(50.0, 1.0, 0.01, 100.0) 

eye = util.vec(2.5, 2.5, 2.5)
target = util.vec(0.0, 0.0, 0.0)
up = util.vec(0.0, 1.0, 0.0)
view = util.lookat(eye, target, up)
gWindow._myCamera = view # otherwise, an imgui slider must be moved to properly update

scene.world.traverse_visit(transUpdate, scene.world.root) 


while running:
    view =  gWindow._myCamera
    scene.world.traverse_visit(transUpdate, scene.world.root) 
    running = scene.render()

    if not animation:
        displayCandle(candle_shader)
    else:
        if frame == 0 :
            angle_joint1 = 0.0
            angle_joint2 = 0.0
        elif frame < 85:
            angle_joint1+=0.5
        elif frame < 170:
            angle_joint2+=1.0
        elif frame < 255:
            angle_joint2-=1.0
        elif frame < 340:
            angle_joint1-=0.5
            frame = -1

        frame+=1
        displayCandle(candle_shader)

    displayGUI()
    view =  gWindow._myCamera
    scene.world.traverse_visit(renderUpdate, scene.world.root)
    scene.render_post()
    
scene.shutdown()