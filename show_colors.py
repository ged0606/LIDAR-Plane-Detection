import math
import numpy as np
import pandas as pd
import sys

from glumpy import app, gloo, gl, glm

vert_shader = """
// LIDAR Data 3d vectors
attribute vec3 color;
attribute vec3 pos;

// Output position of center and radius from eye 
varying vec3 eyespacePos;
varying float eyespaceRadius;
varying float dist_from_origin;
varying vec3 clr;

uniform mat4 modelview;
uniform mat4 projection;
uniform float radius;

void main() {
  dist_from_origin = length(pos);

  vec4 test_point = vec4(pos.xyz, 1.0f) + vec4(0, radius, 0, 0); 
  test_point = modelview * test_point;
  vec4 eyespacePos4 = modelview * vec4(pos.xyz, 1.0f);

  eyespacePos = eyespacePos4.xyz;
  eyespaceRadius = length(test_point - eyespacePos4);

  vec4 clipspacePos = projection * eyespacePos4;
  gl_Position = clipspacePos;
  
  test_point = eyespacePos4 + vec4(0, eyespaceRadius, 0, 0); 
  test_point = projection * test_point;
  test_point = test_point / test_point.w;
  clipspacePos = clipspacePos / clipspacePos.w;
 
  gl_PointSize = 3.0f; 

  clr = color;
}
"""

frag_shader = """
// Parameters from the vertex shader
varying vec3 eyespacePos;
varying float eyespaceRadius;
varying float dist_from_origin;
varying vec3 clr;

// Uniforms
uniform mat4 modelview;
uniform mat4 projection;

// Heat map values
uniform vec4 red = vec4(1.0f, 0, 0, 1.0f);
uniform vec4 yellow = vec4(1.0f, 1.0f, 0, 1.0f);
uniform vec4 green = vec4(0.5f, 1.0f, 0.0f, 1.0f);
uniform vec4 blue = vec4(0.0f, 0.5f, 1.0f, 1.0f);

vec4 heat_map_color(float dist) {
  // dist should be normalized between 0 and 1
  float close  = float(dist < 0.34f);
  float medium = float(dist >= 0.34f && dist < 0.67f);
  float far    = float(dist >= 0.67f);

  vec4 close_value = ((0.34f - dist) * red + dist * yellow) / .34f;
  vec4 medium_value = ((0.34f - dist / 2.0f) * yellow + dist / 2.0f * green) / .34f;
  vec4 far_value = ((0.34f - (min(dist, 1.0f) / 3.0f)) * green + min(dist, 1.0f) / 3.0 * blue) / .34f;

  return close * close_value + medium * medium_value + far * far_value;
}

void main() {
  vec3 normal;
  // See where we are inside the point sprite
  normal.xy = (gl_PointCoord * 2.0f) - vec2(1.0);
  float dist = dot(normal.xy, normal.xy);

  // Discard if outside circle
  //if(dist > 1.0f) {
  //  discard;
  //}

  //gl_FragColor = heat_map_color (dist_from_origin / 40.0f) ;
  gl_FragColor = vec4(clr, 1.0);
  
  // Calculate fragment position in eye space, project to find depth
  vec4 fragPos = vec4(eyespacePos + normal * eyespaceRadius, 1.0);
  vec4 clipspacePos = projection * fragPos;

  // Set up output
  float far = gl_DepthRange.far;
  float near = gl_DepthRange.near;
  float deviceDepth = clipspacePos.z / clipspacePos.w;
  float fragDepth = (((far - near) * deviceDepth) + near +far) / 2.0;
  gl_FragDepth = fragDepth;
}

""" 

frag_shader_line = """
// Parameters from the vertex shader
varying vec3 eyespacePos;
varying float eyespaceRadius;
varying float dist_from_origin;
varying vec3 clr;

// Uniforms
uniform mat4 modelview;
uniform mat4 projection;

void main() {
  vec3 normal;
  // See where we are inside the point sprite
  normal.xy = (gl_PointCoord * 2.0f) - vec2(1.0);
  float dist = dot(normal.xy, normal.xy);

  gl_FragColor = vec4(clr, 1.0);
  
  // Calculate fragment position in eye space, project to find depth
  vec4 fragPos = vec4(eyespacePos + normal * eyespaceRadius, 1.0);
  vec4 clipspacePos = projection * fragPos;

  // Set up output
  float far = gl_DepthRange.far;
  float near = gl_DepthRange.near;
  float deviceDepth = clipspacePos.z / clipspacePos.w;
  float fragDepth = (((far - near) * deviceDepth) + near +far) / 2.0;
  gl_FragDepth = fragDepth;
}

""" 
def magnitude(v):
    return math.sqrt(np.sum(v ** 2))

def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m

def translate(xyz):
    x, y, z = xyz
    return np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])

def lookat(eye, target, up):
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    
    res = np.eye(4, dtype=np.float32)
    res[:3, 0] = s
    res[:3, 1] = u
    res[:3, 2] = -f
    res[3, 0] =-np.dot(s, eye);
    res[3, 1] =-np.dot(u, eye);
    res[3, 2] = np.dot(f, eye);

    return res;

def perspective(field_of_view_y, aspect, z_near, z_far):

    fov_radians = math.radians(float(field_of_view_y))
    f = math.tan(fov_radians/2.0)

    perspective_matrix = np.zeros((4, 4))

    perspective_matrix[0][0] = 1.0 / (f*aspect)
    perspective_matrix[1][1] = 1.0 / f
    perspective_matrix[2][2] = -(z_near + z_far)/(z_far - z_near)
    perspective_matrix[2][3] = -1.0;
    perspective_matrix[3][2] = -2*z_near*z_far/(z_far - z_near)

    return perspective_matrix

# Create program
pc_program = gloo.Program(vert_shader, frag_shader)
line_program = gloo.Program(vert_shader, frag_shader_line)

# Load uniforms
pc_program['modelview'] = lookat(np.array([0, -8, 0.5]), np.array([0, 0, 0]), np.array([0, 0, 1])) 
pc_program['radius'] = .03

line_program['modelview'] = lookat(np.array([0, -8, 0.5]), np.array([0, 0, 0]), np.array([0, 0, 1])) 
line_program['radius'] = .03

# Load colors
with open(sys.argv[2], "r") as f:
  color = [[float(x) for x in line.split(',')] for line in f.readlines()]
  color = np.array(color)

pc_program['color'] = color 

# Load point cloud
with open(sys.argv[1], "r") as f:
  lidar = [[float(x) for x in line.split(',')] for line in f.readlines()]
  lidar = np.array(lidar)

pc_program['pos'] = lidar 

# Load line data
with open(sys.argv[3], "r") as f:
  line_coords = []
  line_colors = []
  for line in f.readlines():
    values = [float(x) for x in line.split(',')]
    line_coords.append((values[0], values[1], values[2]))
    line_coords.append((values[3], values[4], values[5]))
    line_colors.append((values[6], values[7], values[8]))
    line_colors.append((values[6], values[7], values[8]))

line_program['pos'] = line_coords 
line_program['color'] = line_colors 

window = app.Window(width=1280, height=720)

@window.event
def on_resize(width, height):
    ratio = width / float(height)

    # Load projection matrix
    pc_program['projection'] = perspective(45.0, ratio, 0.1, 100.0)
    line_program['projection'] = perspective(45.0, ratio, 0.1, 100.0)

@window.event
def on_draw(dt):
    window.clear()
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glLineWidth(30.0)
    pc_program.draw(mode=gl.GL_POINTS)
    line_program.draw(mode=gl.GL_LINES)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    # Rotate about z axis
    a = dx * 0.01
    rot = np.zeros((4, 4))
    rot[0][0] = math.cos(a)
    rot[0][1] = math.sin(a) 
    rot[1][0] = -math.sin(a)
    rot[1][1] = math.cos(a)
    rot[2][2] = math.cos(a) + (1.0 - math.cos(a)) 
    
    rot[3][3] = 1
    pc_program['modelview'] = np.dot(rot, pc_program['modelview'].reshape((4, 4))) 
    line_program['modelview'] = np.dot(rot, pc_program['modelview'].reshape((4, 4)))

# Run the app
app.run()

