import os
import ctypes
import numpy as np
import OpenGL.GL as gl
from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric
from PyQt6.QtGui import QImage
import stl_loader

_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')

def _load_shader_source(filename: str) -> str:
    path = os.path.join(_SHADER_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

QUAD_VERTICES = np.array([
    # Первый треугольник
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
    # Второй треугольник
    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
], dtype=np.float32)

class GLResources:
    """Manages OpenGL resources such as textures, shaders, FBOs, and display lists."""
    def __init__(self):
        self.quadric = None
        self.textures = {}
        self.display_lists = {}
        
        self.fbo = None
        self.fbo_texture = None
        self.fbo_depth = None
        self.fbo_width = 800
        self.fbo_height = 600

        self.shader_program = None
        self.model_shader_program = None
        self.quad_vao = None
        self.quad_vbo = None

    def initialize(self):
        if self.quadric is not None:
            gluDeleteQuadric(self.quadric)
        self.quadric = gluNewQuadric()

        gl.glClearColor(0.4, 0.5, 0.6, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, [0.7, 0.7, 0.7, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        self._compile_shaders()
        self._create_fullscreen_quad()

    def _compile_shaders(self):
        from OpenGL.GL import shaders as gl_shaders
        v_src = _load_shader_source('vertex.glsl')
        f_src = _load_shader_source('fragment.glsl')
        mV_src = _load_shader_source('model_vertex.glsl')
        mF_src = _load_shader_source('model_fragment.glsl')

        vertex = gl_shaders.compileShader(v_src, gl.GL_VERTEX_SHADER)
        fragment = gl_shaders.compileShader(f_src, gl.GL_FRAGMENT_SHADER)
        self.shader_program = gl_shaders.compileProgram(vertex, fragment)
        
        m_vertex = gl_shaders.compileShader(mV_src, gl.GL_VERTEX_SHADER)
        m_fragment = gl_shaders.compileShader(mF_src, gl.GL_FRAGMENT_SHADER)
        self.model_shader_program = gl_shaders.compileProgram(m_vertex, m_fragment)

    def _create_fullscreen_quad(self):
        self.quad_vao = gl.glGenVertexArrays(1)
        self.quad_vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.quad_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, QUAD_VERTICES.nbytes, QUAD_VERTICES, gl.GL_STATIC_DRAW)

        stride = 4 * ctypes.sizeof(ctypes.c_float)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def create_fbo(self, w: int, h: int, dpr: float = 1.0):
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
        if self.fbo_texture is not None:
            gl.glDeleteTextures(1, [self.fbo_texture])
        if self.fbo_depth is not None:
            gl.glDeleteRenderbuffers(1, [self.fbo_depth])

        self.fbo_width = max(int(w * dpr), 1)
        self.fbo_height = max(int(h * dpr), 1)

        self.fbo_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.fbo_width, self.fbo_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.fbo_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.fbo_depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.fbo_width, self.fbo_height)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.fbo_texture, 0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.fbo_depth)
        
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"Ошибка FBO! Статус: {status}")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def load_texture(self, name: str, file_path: str, wrap=gl.GL_CLAMP_TO_BORDER):
        img = QImage(file_path)
        if img.isNull():
            return None
            
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = img.width(), img.height()
        
        ptr = img.constBits()
        ptr.setsize(width * height * 4)
        data = np.frombuffer(ptr, dtype=np.uint8).copy()
        
        tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap)
        if wrap == gl.GL_CLAMP_TO_BORDER:
            gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, [0.0, 0.0, 0.0, 0.0])
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
        self.textures[name] = tex_id
        return tex_id

    def get_texture(self, name: str):
        return self.textures.get(name)

    def build_mesh_display_list(self, name: str, mesh):
        self.display_lists[name] = stl_loader.build_display_list(mesh)

    def get_display_list(self, name: str):
        return self.display_lists.get(name)

    def cleanup(self):
        if self.quadric is not None:
            gluDeleteQuadric(self.quadric)
            self.quadric = None
        for dl in self.display_lists.values():
            gl.glDeleteLists(dl, 1)
        self.display_lists.clear()
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self.fbo_texture is not None:
            gl.glDeleteTextures(1, [self.fbo_texture])
            self.fbo_texture = None
        if self.fbo_depth is not None:
            gl.glDeleteRenderbuffers(1, [self.fbo_depth])
            self.fbo_depth = None
        if self.shader_program is not None:
            gl.glDeleteProgram(self.shader_program)
            self.shader_program = None
        if self.model_shader_program is not None:
            gl.glDeleteProgram(self.model_shader_program)
            self.model_shader_program = None
        for tex_id in self.textures.values():
            gl.glDeleteTextures(1, [tex_id])
        self.textures.clear()
        if self.quad_vao is not None:
            gl.glDeleteVertexArrays(1, [self.quad_vao])
            self.quad_vao = None
        if self.quad_vbo is not None:
            gl.glDeleteBuffers(1, [self.quad_vbo])
            self.quad_vbo = None
