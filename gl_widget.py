"""
OpenGL-виджет для отрисовки 3D сцены (Отрефакторенный).
"""
import os
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl
from OpenGL.GLU import gluPerspective, gluLookAt, gluSphere

import stl_loader
from scene_state import SceneState
from camera_controller import CameraController
from gl_resources import GLResources

class SceneGLWidget(QOpenGLWidget):
    """Виджет для отрисовки 3D сцены, оркестратор компонентов"""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene_state = SceneState()
        self.camera_controller = CameraController()
        self.render_camera_controller = CameraController()
        self.gl_resources = GLResources()

        self.last_mouse_pos = None

        self.projector_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'base_texture.png')

    # Делегирование свойств SceneState, чтобы не ломать внешний API (MainWindow)
    @property
    def airplane_pos(self): return self.scene_state.airplane_pos
    @airplane_pos.setter
    def airplane_pos(self, value): self.scene_state.airplane_pos = value

    @property
    def airplane_rot(self): return self.scene_state.airplane_rot
    @airplane_rot.setter
    def airplane_rot(self, value): self.scene_state.airplane_rot = value

    @property
    def tof_pos(self): return self.scene_state.tof_pos
    @tof_pos.setter
    def tof_pos(self, value): self.scene_state.tof_pos = value

    @property
    def tof_dir(self): return self.scene_state.tof_dir
    @tof_dir.setter
    def tof_dir(self, value): self.scene_state.tof_dir = value

    @property
    def cam_fov(self): return self.render_camera_controller.fov
    @property
    def cam_near(self): return self.render_camera_controller.near
    @property
    def cam_far(self): return self.render_camera_controller.far

    def initializeGL(self):
        self.gl_resources.initialize()
            
        self.gl_resources.create_fbo(self.width() or 800, self.height() or 600, self.devicePixelRatioF())
        
        if not self.gl_resources.get_texture('projector'):
            self.gl_resources.load_texture('projector', self.projector_texture_path)
            
        if getattr(self.scene_state, 'scene_config', None):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for tex_id, tex_path in self.scene_state.scene_config.textures.items():
                if not self.gl_resources.get_texture(tex_id):
                    full_path = os.path.join(base_dir, tex_path)
                    self.gl_resources.load_texture(tex_id, full_path, wrap=gl.GL_REPEAT)

    def resizeGL(self, w, h):
        if self.gl_resources.fbo is not None:
            self.gl_resources.create_fbo(w, h, self.devicePixelRatioF())

    def apply_camera_config(self, config: dict):
        self.render_camera_controller.apply_config(config)
        self.update()

    def cleanup(self):
        self.makeCurrent()
        self.gl_resources.cleanup()
        self.doneCurrent()

    def hideEvent(self, event):
        self.cleanup()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.gl_resources.fbo is None:
            self.makeCurrent()
            self.initializeGL()
            self.doneCurrent()
            self.update()

    def _get_projector_matrix(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        
        gl.glTranslatef(0.5, 0.5, 0.5)
        gl.glScalef(0.5, 0.5, 0.5)
        
        gluPerspective(30.0, 1.0, 0.1, 100.0)
        
        gluLookAt(2.0, 3.0, 4.0,  
                  2.0, 0.0, 0.0,   
                  0.0, 1.0, 0.0)  
        
        gl.glTranslatef(self.scene_state.airplane_pos[0], self.scene_state.airplane_pos[1], self.scene_state.airplane_pos[2])
        gl.glRotatef(self.scene_state.airplane_rot[0], 0.0, 1.0, 0.0)
        gl.glRotatef(self.scene_state.airplane_rot[1], 1.0, 0.0, 0.0)
        gl.glRotatef(self.scene_state.airplane_rot[2], 0.0, 0.0, 1.0)
            
        proj_matrix = gl.glGetFloatv(gl.GL_PROJECTION_MATRIX)
        
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        return proj_matrix

    def _draw_building(self, x, y, z, w=4.0, d=4.0, h=5.0, color=(0.6, 0.6, 0.6), texture=None):
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y,         y + h
        z0, z1 = z - d / 2, z + d / 2

        r, g, b = color

        gl.glPushMatrix()
        gl.glColor3f(r, g, b)

        use_tex = texture is not None
        if use_tex:
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        else:
            gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glBegin(gl.GL_QUADS)
        gl.glNormal3f(0, 0, 1)
        if use_tex: gl.glTexCoord2f(0, 0)
        gl.glVertex3f(x0, y0, z1)
        if use_tex: gl.glTexCoord2f(1, 0)
        gl.glVertex3f(x1, y0, z1)
        if use_tex: gl.glTexCoord2f(1, 1)
        gl.glVertex3f(x1, y1, z1)
        if use_tex: gl.glTexCoord2f(0, 1)
        gl.glVertex3f(x0, y1, z1)
        
        gl.glNormal3f(0, 0, -1)
        if use_tex: gl.glTexCoord2f(0, 0)
        gl.glVertex3f(x1, y0, z0)
        if use_tex: gl.glTexCoord2f(1, 0)
        gl.glVertex3f(x0, y0, z0)
        if use_tex: gl.glTexCoord2f(1, 1)
        gl.glVertex3f(x0, y1, z0)
        if use_tex: gl.glTexCoord2f(0, 1)
        gl.glVertex3f(x1, y1, z0)
        
        gl.glNormal3f(-1, 0, 0)
        if use_tex: gl.glTexCoord2f(0, 0)
        gl.glVertex3f(x0, y0, z0)
        if use_tex: gl.glTexCoord2f(1, 0)
        gl.glVertex3f(x0, y0, z1)
        if use_tex: gl.glTexCoord2f(1, 1)
        gl.glVertex3f(x0, y1, z1)
        if use_tex: gl.glTexCoord2f(0, 1)
        gl.glVertex3f(x0, y1, z0)
        
        gl.glNormal3f(1, 0, 0)
        if use_tex: gl.glTexCoord2f(0, 0)
        gl.glVertex3f(x1, y0, z1)
        if use_tex: gl.glTexCoord2f(1, 0)
        gl.glVertex3f(x1, y0, z0)
        if use_tex: gl.glTexCoord2f(1, 1)
        gl.glVertex3f(x1, y1, z0)
        if use_tex: gl.glTexCoord2f(0, 1)
        gl.glVertex3f(x1, y1, z1)
        gl.glEnd()

        if use_tex:
            gl.glDisable(gl.GL_TEXTURE_2D)
            
        gl.glBegin(gl.GL_QUADS)
        gl.glNormal3f(0, 1, 0)
        gl.glColor3f(0.2, 0.2, 0.2)
        gl.glVertex3f(x0, y1, z0); gl.glVertex3f(x0, y1, z1)
        gl.glVertex3f(x1, y1, z1); gl.glVertex3f(x1, y1, z0)
        gl.glEnd()

        if use_tex:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glPopMatrix()

    def paintGL(self):
        if self.gl_resources.fbo is None:
            return

        if self.gl_resources.quadric is None:
            self.gl_resources.initialize()

        dpr = self.devicePixelRatioF()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.gl_resources.fbo)
        gl.glViewport(0, 0, self.gl_resources.fbo_width, self.gl_resources.fbo_height)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = self.gl_resources.fbo_width / max(self.gl_resources.fbo_height, 1)
        gluPerspective(self.camera_controller.fov, aspect, self.camera_controller.near, self.camera_controller.far)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gluLookAt(
            self.camera_controller.pos[0], self.camera_controller.pos[1], self.camera_controller.pos[2],
            self.camera_controller.target[0], self.camera_controller.target[1], self.camera_controller.target[2],
            0.0, 1.0, 0.0
        )

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (5, 5, -5, 1))

        # Render SceneConfig objects
        if getattr(self.scene_state, 'scene_config', None):
            for obj in self.scene_state.scene_config.objects:
                pos = self.scene_state.airplane_pos if obj.dynamic_pos == 'airplane_pos' else obj.position
                rot = self.scene_state.airplane_rot if obj.dynamic_rot == 'airplane_rot' else obj.rotation
                
                gl.glPushMatrix()
                gl.glTranslatef(pos[0], pos[1], pos[2])
                gl.glRotatef(rot[0], 0.0, 1.0, 0.0)
                gl.glRotatef(rot[1], 1.0, 0.0, 0.0)
                gl.glRotatef(rot[2], 0.0, 0.0, 1.0)
                gl.glScalef(obj.scale[0], obj.scale[1], obj.scale[2])
                
                if obj.type == 'mesh':
                    gl.glUseProgram(self.gl_resources.model_shader_program)
                    
                    loc_tex = gl.glGetUniformLocation(self.gl_resources.model_shader_program, "projectorTexture")
                    loc_use_proj = gl.glGetUniformLocation(self.gl_resources.model_shader_program, "useProjector")
                    loc_mat = gl.glGetUniformLocation(self.gl_resources.model_shader_program, "projectorMatrix")
                    
                    loc_base_tex = gl.glGetUniformLocation(self.gl_resources.model_shader_program, "baseTexture")
                    loc_use_base = gl.glGetUniformLocation(self.gl_resources.model_shader_program, "useBaseTexture")
                    
                    gl.glUniform1i(loc_tex, 1) 
                    gl.glUniform1i(loc_base_tex, 0)
                    
                    gl.glActiveTexture(gl.GL_TEXTURE1)
                    proj_tex = self.gl_resources.get_texture('projector')
                    if proj_tex is not None and obj.dynamic_pos == 'airplane_pos':
                        gl.glBindTexture(gl.GL_TEXTURE_2D, proj_tex)
                        gl.glUniform1i(loc_use_proj, 0)
                    else:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                        gl.glUniform1i(loc_use_proj, 0)
                        
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    air_tex = self.gl_resources.get_texture(obj.texture) if obj.texture else None
                    if air_tex is not None:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, air_tex)
                        gl.glUniform1i(loc_use_base, 1)
                    else:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                        gl.glUniform1i(loc_use_base, 0)

                    # Lazy load model
                    mesh_id = obj.model_path
                    if not self.gl_resources.get_display_list(mesh_id):
                        mesh = stl_loader.load_stl(mesh_id)
                        if mesh:
                            self.gl_resources.build_mesh_display_list(mesh_id, mesh)
                            
                    dl = self.gl_resources.get_display_list(mesh_id)
                    if dl is not None:
                        airplane_proj = self._get_projector_matrix()
                        gl.glUniformMatrix4fv(loc_mat, 1, gl.GL_FALSE, airplane_proj)
                        
                        gl.glColor3f(*obj.color)
                        gl.glCallList(dl)
                        
                    gl.glUseProgram(0)
                
                elif obj.type == 'box':
                    tex = self.gl_resources.get_texture(obj.texture) if obj.texture else None
                    box_w, box_h, box_d = obj.dimensions
                    # The box is drawn centered internally using obj.position? Usually box is at 0,0,0 and scaled.
                    # _draw_building logic expects absolute coords but we are translated already.
                    # So we pass 0,0,0
                    self._draw_building(0.0, 0.0, 0.0, w=box_w, d=box_d, h=box_h, color=tuple(obj.color), texture=tex)
                    
                elif obj.type == 'plane':
                    gl.glEnable(gl.GL_TEXTURE_2D)
                    tex = self.gl_resources.get_texture(obj.texture) if obj.texture else None
                    if tex is not None:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                        gl.glColor3f(*(obj.color if len(obj.color) == 3 else [1.0, 1.0, 1.0]))
                    else:
                        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                        gl.glColor3f(*obj.color)
                        
                    gl.glBegin(gl.GL_QUADS)
                    gl.glNormal3f(0.0, 1.0, 0.0)
                    
                    # Size is scaled relative to 1x1 plane centered at 0,0,0
                    # For planes, we just draw unit quad and scale handles size.
                    # Wait, our scene object scale is already half-width?
                    # Let's verify scene_loader: "sx, sy, sz = self.scale; v0 = np.array([px - sx, py, pz - sz])"
                    # So unit quad is from -1 to 1 based on X and Z.
                    
                    # Some texture scaling logic if needed:
                    tiles_x = obj.scale[0] / 3.0 if obj.texture == 'ground' else (obj.scale[0] / 4.0 if obj.texture == 'road' else 1.0)
                    tiles_y = obj.scale[2] / 3.0 if obj.texture == 'ground' else (obj.scale[2] / 4.0 if obj.texture == 'field' else 1.0)
                    
                    gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(-1.0, 0.0, -1.0)
                    gl.glTexCoord2f(0.0, tiles_y); gl.glVertex3f(-1.0, 0.0,  1.0)
                    gl.glTexCoord2f(tiles_x, tiles_y); gl.glVertex3f( 1.0, 0.0,  1.0)
                    gl.glTexCoord2f(tiles_x, 0.0); gl.glVertex3f( 1.0, 0.0, -1.0)
                    gl.glEnd()
                    gl.glDisable(gl.GL_TEXTURE_2D)

                gl.glPopMatrix()

        # Маркер ToF камеры
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_LIGHTING)

        gl.glPushMatrix()
        gl.glTranslatef(self.scene_state.tof_pos[0], self.scene_state.tof_pos[1], self.scene_state.tof_pos[2])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, [0.8, 0.0, 0.0, 1.0])
        gl.glColor3f(1.0, 0.0, 0.0)
        gluSphere(self.gl_resources.quadric, 0.2, 16, 16)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
        gl.glPopMatrix()

        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(2.0)
        gl.glColor3f(1.0, 0.4, 0.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(self.scene_state.tof_pos[0], self.scene_state.tof_pos[1], self.scene_state.tof_pos[2])
        target = [self.scene_state.tof_pos[i] + self.scene_state.tof_dir[i] for i in range(3)]
        gl.glVertex3f(target[0], target[1], target[2])
        gl.glEnd()
        gl.glLineWidth(1.0)
        gl.glEnable(gl.GL_LIGHTING)

        # Маркер камеры рендера
        render_pos = self.render_camera_controller.pos
        render_target = self.render_camera_controller.target

        gl.glPushMatrix()
        gl.glTranslatef(render_pos[0], render_pos[1], render_pos[2])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, [0.0, 0.35, 0.9, 1.0])
        gl.glColor3f(0.2, 0.6, 1.0)
        gluSphere(self.gl_resources.quadric, 0.2, 16, 16)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
        gl.glPopMatrix()

        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(2.0)
        gl.glColor3f(0.2, 0.75, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(render_pos[0], render_pos[1], render_pos[2])
        gl.glVertex3f(render_target[0], render_target[1], render_target[2])
        gl.glEnd()
        gl.glLineWidth(1.0)
        gl.glEnable(gl.GL_LIGHTING)

        # Full screen quad draw
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glViewport(0, 0, w, h)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)

        gl.glUseProgram(self.gl_resources.shader_program)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_resources.fbo_texture)
        loc = gl.glGetUniformLocation(self.gl_resources.shader_program, "screenTexture")
        gl.glUniform1i(loc, 0)

        gl.glBindVertexArray(self.gl_resources.quad_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

        gl.glUseProgram(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = event.position()
            return

        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.position()

        if event.buttons() & Qt.MouseButton.LeftButton:
            self.camera_controller.process_mouse_rotate(dx, dy)
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            self.camera_controller.process_mouse_pan(dx, dy)
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_controller.process_mouse_zoom(delta)
        self.update()
