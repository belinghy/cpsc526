from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

#  from pyquaternion import Quaternion    ## would be useful for 3D simulation
import numpy as np

window = 0     # number of the glut window
theta = 0.0
simTime = 0
dT = 0.001
simRun = True
RAD_TO_DEG = 180.0/3.1416
init_angle = 3.1416 / 4
xoffset = np.sin(init_angle)
yoffset = -np.cos(init_angle)
init_cpoint1 = np.array([-0.5*xoffset, -0.5*yoffset, 0])
Ixyz = 1.0 / 3
k_d = 0.8

#####################################################
#### Link class, i.e., for a rigid body
#####################################################

class Link:
    color=[0,0,0]    ## draw color
    size=[1,1,1]     ## dimensions
    mass = 1.0       ## mass in kg
    I = np.array([[Ixyz, 0, 0], [0, Ixyz, 0], [0, 0, Ixyz]])        ## moment of inertia about z-axis
    theta=np.array([0.0, 0.0, 0.0])          ## 2D orientation  (will need to change for 3D)
    omega=np.array([0.0, 0.0, 0.0])          ## 2D angular velocity
    posn=np.array([0.0,0.0,0.0])     ## 3D position (keep z=0 for 2D)
    vel=np.array([0.0,0.0,0.0])      ## initial velocity
    def draw(self):      ### steps to draw a link
        glPushMatrix()                                            ## save copy of coord frame
        glTranslatef(self.posn[0], self.posn[1], self.posn[2])    ## move 
        glRotatef(self.theta[2]*RAD_TO_DEG,  0,0,1)
        glRotatef(self.theta[1]*RAD_TO_DEG,  0,1,0)
        glRotatef(self.theta[0]*RAD_TO_DEG,  1,0,0)                            ## rotate
        glScale(self.size[0], self.size[1], self.size[2])         ## set size
        glColor3f(self.color[0], self.color[1], self.color[2])    ## set colour
        DrawCube()                                                ## draw a scaled cube
        glPopMatrix()                                             ## restore old coord frame

#####################################################
#### main():   launches app
#####################################################

def main():
    global window
    global link1, link2, link3     
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)     # display mode 
    glutInitWindowSize(640, 480)                                  # window size
    glutInitWindowPosition(0, 0)                                  # window coords for mouse start at top-left
    window = glutCreateWindow("CPSC 526 Simulation Template")
    glutDisplayFunc(DrawWorld)       # register the function to draw the world
    # glutFullScreen()               # full screen
    glutIdleFunc(SimWorld)          # when doing nothing, redraw the scene
    glutReshapeFunc(ReSizeGLScene)   # register the function to call when window is resized
    glutKeyboardFunc(keyPressed)     # register the function to call when keyboard is pressed
    InitGL(640, 480)                 # initialize window
    
    link1 = Link()
    link2 = Link()
    link3 = Link()
    resetSim()
    
    glutMainLoop()                   # start event processing loop

#####################################################
#### keyPressed():  called whenever a key is pressed
#####################################################

def resetSim():
    global link1, link2
    global simTime, simRun

    printf("Simulation reset\n")
    simRun = True
    simTime = 0

    link1.size=[0.04, 1.0, 0.12]
    link1.color=[1,0.9,0.9]
    link1.posn=np.array([0.0,0.0,0.0])
    link1.vel=np.array([0.0,0.0,0.0])
    link1.theta = np.array([init_angle, 0, init_angle])
    link1.omega = np.array([0.0, 0.0, 0.0])        ## radians per second
    
    link2.size=[0.04, 1.0, 0.12]
    link2.color=[0.9,0.9,1.0]
    link2.vel=np.array([0.0,0.0,0.0])
    link2.theta = np.array([init_angle, 0, init_angle])
    link2.omega = np.array([0.0, 0.0, 0.0])        ## radians per second

    temp_offset1 = rotate3(np.array([0.0, -0.5, 0.0]), link1.theta)
    temp_offset2 = rotate3(np.array([0.0, -0.5, 0.0]), link2.theta)
    link2.posn= link1.posn + (temp_offset1 + temp_offset2) * 0.86

    link3.size=[0.04, 1.0, 0.12]
    link3.color=[0.9,0.9,1.0]
    #link3.posn=temp_offset + temp_offset2 + temp_offset3
    link3.vel=np.array([0.0,0.0,0.0])
    link3.theta = np.array([0.0, 0.0, 3.1416/2]) # rotate about z by 90
    link3.omega = np.array([0.0, 0.0, 0.0])        ## radians per second

    temp_offset1 = rotate3(np.array([0.0, -0.5, 0.0]), link2.theta)
    temp_offset2 = rotate3(np.array([0.0, -0.5, 0.0]), link3.theta)
    link3.posn=link2.posn + (temp_offset1 + temp_offset2) * 0.9

#####################################################
#### keyPressed():  called whenever a key is pressed
#####################################################

def keyPressed(key,x,y):
    global simRun
    ch = key.decode("utf-8")
    if ch == ' ':                #### toggle the simulation
        if (simRun == True):
            simRun = False
        else:
            simRun = True
    elif ch == chr(27):          #### ESC key
        sys.exit()
    elif ch == 'q':              #### quit
        sys.exit()
    elif ch == 'r':              #### reset simulation
        resetSim()

#####################################################
#### SimWorld():  simulates a time step
#####################################################

def SimWorld():
    global simTime, dT, simRun
    global link1, link2, link3

    if (simRun==False):             ## is simulation stopped?
        return

    m = link1.mass
    I = Ixyz
    k = -1.0
    o = 1.0
    g = -10.0
    z = 0.0

    r1 = rotate3(np.array([0.0, 0.5, 0.0]), link1.theta)
    r1x = r1[0]
    r1y = r1[1]
    r1z = r1[2]

    r2 = rotate3(np.array([0.0, 0.5, 0.0]), link2.theta)
    r2x = r2[0]
    r2y = r2[1]
    r2z = r2[2]

    r3 = rotate3(np.array([0.0, 0.5, 0.0]), link3.theta)
    r3x = r3[0]
    r3y = r3[1]
    r3z = r3[2]

    w1 = link1.omega
    I1 = np.matmul(np.matmul(rotate3(link1.I, link1.theta), link1.I), np.linalg.inv(rotate3(link1.I, link1.theta)))
    I1x = I1[0][0]
    I1y = I1[1][1]
    I1z = I1[2][2]

    w2 = link2.omega
    I2 = np.matmul(np.matmul(rotate3(link2.I, link2.theta), link2.I), np.linalg.inv(rotate3(link2.I, link2.theta)))
    I2x = I2[0][0]
    I2y = I2[1][1]
    I2z = I2[2][2]

    w3 = link3.omega
    I3 = np.matmul(np.matmul(rotate3(link3.I, link3.theta), link3.I), np.linalg.inv(rotate3(link3.I, link3.theta)))
    I3x = I3[0][0]
    I3y = I3[1][1]
    I3z = I3[2][2]

    ####  for the constrained one-link pendulum, and the 4-link pendulum,
    ####  you will want to build the equations of motion as a linear system, and then solve that.
    ####  Here is a simple example of using numpy to solve a linear system.
    a = np.array([[m, z, z, z, z, z,     z, z, z, z, z, z,     z, z, z, z, z, z,     k,   z,   z,     k,   z,   z,     z,   z,   z ],
                  [z, m, z, z, z, z,     z, z, z, z, z, z,     z, z, z, z, z, z,     z,   k,   z,     z,   k,   z,     z,   z,   z ],
                  [z, z, m, z, z, z,     z, z, z, z, z, z,     z, z, z, z, z, z,     z,   z,   k,     z,   z,   k,     z,   z,   z ],
                  [z, z, z,I1x,z, z,     z, z, z, z, z, z,     z, z, z, z, z, z,     z,  r1z,-r1y,    z, -r1z, r1y,    z,   z,   z ],
                  [z, z, z, z,I1y,z,     z, z, z, z, z, z,     z, z, z, z, z, z,   -r1z,  z,  r1x,   r1z,  z, -r1x,    z,   z,   z ],
                  [z, z, z, z, z,I1z,    z, z, z, z, z, z,     z, z, z, z, z, z,    r1y,-r1x,  z,   -r1y, r1x,  z,     z,   z,   z ],

                  [z, z, z, z, z, z,     m, z, z, z, z, z,     z, z, z, z, z, z,     z,   z,   z,     o,   z,   z ,    k,   z,   z ],
                  [z, z, z, z, z, z,     z, m, z, z, z, z,     z, z, z, z, z, z,     z,   z,   z,     z,   o,   z ,    z,   k,   z ],
                  [z, z, z, z, z, z,     z, z, m, z, z, z,     z, z, z, z, z, z,     z,   z,   z,     z,   z,   o ,    z,   z,   k ],
                  [z, z, z, z, z, z,     z, z, z,I2x,z, z,     z, z, z, z, z, z,     z,   z,   z,     z, -r2z, r2y,    z, -r2z, r2y],
                  [z, z, z, z, z, z,     z, z, z, z,I2y,z,     z, z, z, z, z, z,     z,   z,   z,    r2z,  z, -r2x,   r2z,  z, -r2x],
                  [z, z, z, z, z, z,     z, z, z, z, z,I2z,    z, z, z, z, z, z,     z,   z,   z,   -r2y, r2x,  z ,  -r2y, r2x,  z ],

                  [z, z, z, z, z, z,     z, z, z, z, z, z,     m, z, z, z, z, z,     z,   z,   z,     z,   z,   z,     o,   z,   z ],
                  [z, z, z, z, z, z,     z, z, z, z, z, z,     z, m, z, z, z, z,     z,   z,   z,     z,   z,   z,     z,   o,   z ],
                  [z, z, z, z, z, z,     z, z, z, z, z, z,     z, z, m, z, z, z,     z,   z,   z,     z,   z,   z,     z,   z,   o ],
                  [z, z, z, z, z, z,     z, z, z, z, z, z,     z, z, z,I3x,z, z,     z,   z,   z,     z,   z,   z,     z, -r3z, r3y],
                  [z, z, z, z, z, z,     z, z, z, z, z, z,     z, z, z, z,I3y,z,     z,   z,   z,     z,   z,   z,    r3z,  z, -r3x],
                  [z, z, z, z, z, z,     z, z, z, z, z, z,     z, z, z, z, z,I3z,    z,   z,   z,     z,   z,   z,   -r3y, r3x,  z ],

                  [k, z, z,   z,  -r1z,  r1y,     z, z, z,    z,    z,    z,     z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],
                  [z, k, z,  r1z,   z,  -r1x,     z, z, z,    z,    z,    z,     z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],
                  [z, z, k, -r1y,  r1x,   z ,     z, z, z,    z,    z,    z,     z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],

                  [k, z, z,   z,   r1z, -r1y,     o, z, z,    z,   r2z, -r2y,    z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],
                  [z, k, z, -r1z,   z,   r1x,     z, o, z,  -r2z,   z,   r2x,    z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],
                  [z, z, k,  r1y, -r1x,   z,      z, z, o,   r2y, -r2x,   z,     z, z, z,  z,   z,   z,      z, z, z,   z, z, z,   z, z, z],

                  [z, z, z,   z,    z,    z,      k, z, z,    z,   r2z, -r2y,    o, z, z,  z,  r3z,-r3y,     z, z, z,   z, z, z,   z, z, z],
                  [z, z, z,   z,    z,    z,      z, k, z,  -r2z,   z,   r2x,    z, o, z,-r3z,  z,  r3x,     z, z, z,   z, z, z,   z, z, z],
                  [z, z, z,   z,    z,    z,      z, z, k,   r2y, -r2x,   z,     z, z, o, r3y, -r3x, z,      z, z, z,   z, z, z,   z, z, z]
                  ])

    w1I1w1 = np.cross(w1, np.matmul(I1, w1))
    w2I2w2 = np.cross(w2, np.matmul(I2, w2))
    w3I3w3 = np.cross(w3, np.matmul(I3, w3))

    w1w1r1_up = np.cross(w1, np.cross(w1, r1))
    w1w1r1_do = np.cross(w1, np.cross(w1, -r1))
    
    w2w2r2_up = np.cross(w2, np.cross(w2, r2))
    w2w2r2_do = np.cross(w2, np.cross(w2, -r2))

    w3w3r3_up = np.cross(w3, np.cross(w3, r3))
    w3w3r3_do = np.cross(w3, np.cross(w3, -r3))

    b = np.array([z, g, z, -k_d*w1[0]-w1I1w1[0], -k_d*w1[1]-w1I1w1[1], -k_d*w1[2]-w1I1w1[2],     
                  z, g, z, -k_d*w2[0]-w2I2w2[0], -k_d*w2[1]-w2I2w2[1], -k_d*w2[2]-w2I2w2[2],
                  z, g, z, -k_d*w3[0]-w3I3w3[0], -k_d*w3[1]-w3I3w3[1], -k_d*w3[2]-w3I3w3[2],
                  w1w1r1_up[0], w1w1r1_up[1], w1w1r1_up[2], 
                  w1w1r1_do[0]-w2w2r2_up[0], w1w1r1_do[1]-w2w2r2_up[1], w1w1r1_do[2]-w2w2r2_up[2],
                  w2w2r2_do[0]-w3w3r3_up[0], w2w2r2_do[1]-w3w3r3_up[1], w2w2r2_do[2]-w3w3r3_up[2]])

    x = np.linalg.solve(a, b)
    print(x)   # [ -2.17647059  53.54411765  56.63235294]

    acc1 = np.array([x[0], x[1], x[2]])
    omega_dot1 = np.array([x[3], x[4], x[5]])
    acc2 = np.array([x[6], x[7], x[8]])
    omega_dot2 = np.array([x[9], x[10], x[11]])
    acc3 = np.array([x[12], x[13], x[14]])
    omega_dot3 = np.array([x[15], x[16], x[17]])


    #### Stabilization ?
    cpoint1 = link1.posn + r1
    cvelocity1 = link1.vel + np.cross(w1, r1)
    factor1 = 1*(cpoint1 - init_cpoint1) + 10*(cvelocity1)
    acc1 = acc1 - factor1

    cpoint2 = link2.posn + r2
    cdiff2 = cpoint2 - (link1.posn - r1)
    cvelocity2 = (link2.vel + np.cross(w2, r2)) - (link1.vel + np.cross(w1, -r1))
    factor2 = 100*(cdiff2) + 200*(cvelocity2)
    acc2 = acc2 - factor2 + acc1

    cpoint3 = link3.posn + r3
    cdiff3 = cpoint3 - (link2.posn - r2)
    cvelocity3 = (link3.vel + np.cross(w3, r3)) - (link2.vel + np.cross(w2, -r2))
    factor3 = 500*(cdiff3) + 100*(cvelocity3)
    acc3 = acc3 - factor3 + acc2

    #### explicit Euler integration to update the state
    link1.posn += link1.vel*dT
    link1.vel += acc1*dT
    link1.theta += link1.omega*dT
    link1.omega += omega_dot1*dT

    link2.posn += link2.vel*dT
    link2.vel += acc2*dT
    link2.theta += link2.omega*dT
    link2.omega += omega_dot2*dT

    link3.posn += link3.vel*dT
    link3.vel += acc3*dT
    link3.theta += link3.omega*dT
    link3.omega += omega_dot3*dT

    simTime += dT

    #### draw the updated state
    DrawWorld()
    printf("simTime=%.2f\n",simTime)

#####################################################
#### DrawWorld():  draw the world
#####################################################

def DrawWorld():
    global link1, link2, link3

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	# Clear The Screen And The Depth Buffer
    glLoadIdentity();
    gluLookAt(2,2,6,  0,0,0,  0,1,0)

    DrawOrigin()
    link1.draw()
    link2.draw()
    link3.draw()

    glutSwapBuffers()                      # swap the buffers to display what was just drawn

#####################################################
#### initGL():  does standard OpenGL initialization work
#####################################################

def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
    glClearColor(1.0, 1.0, 0.9, 0.0)	# This Will Clear The Background Color To Black
    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);    glEnable( GL_LINE_SMOOTH );
    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()					# Reset The Projection Matrix
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

#####################################################
#### ReSizeGLScene():    called when window is resized
#####################################################

def ReSizeGLScene(Width, Height):
    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
        Height = 1
    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)    ## 45 deg horizontal field of view, aspect ratio, near, far
    glMatrixMode(GL_MODELVIEW)

#####################################################
#### DrawOrigin():  draws RGB lines for XYZ origin of coordinate system
#####################################################

def DrawOrigin():
    glLineWidth(3.0);

    glColor3f(1,0.5,0.5)   ## light red x-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(1,0,0)
    glEnd()

    glColor3f(0.5,1,0.5)   ## light green y-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,1,0)
    glEnd()

    glColor3f(0.5,0.5,1)   ## light blue z-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,0,1)
    glEnd()

#####################################################
#### DrawCube():  draws a cube that spans from (-1,-1,-1) to (1,1,1)
#####################################################

def DrawCube():

    glScalef(0.5,0.5,0.5);                  # dimensions below are for a 2x2x2 cube, so scale it down by a half first
    glBegin(GL_QUADS);			# Start Drawing The Cube

    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Top)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Top)
    glVertex3f(-1.0, 1.0, 1.0);		# Bottom Left Of The Quad (Top)
    glVertex3f( 1.0, 1.0, 1.0);		# Bottom Right Of The Quad (Top)

    glVertex3f( 1.0,-1.0, 1.0);		# Top Right Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0, 1.0);		# Top Left Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Bottom)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Bottom)

    glVertex3f( 1.0, 1.0, 1.0);		# Top Right Of The Quad (Front)
    glVertex3f(-1.0, 1.0, 1.0);		# Top Left Of The Quad (Front)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Front)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Front)

    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Back)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Back)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Right Of The Quad (Back)
    glVertex3f( 1.0, 1.0,-1.0);		# Top Left Of The Quad (Back)

    glVertex3f(-1.0, 1.0, 1.0);		# Top Right Of The Quad (Left)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Left)

    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Right)
    glVertex3f( 1.0, 1.0, 1.0);		# Top Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Right)
    glEnd();				# Done Drawing The Quad

    ### Draw the wireframe edges
    glColor3f(0.0, 0.0, 0.0);
    glLineWidth(1.0);
     
    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Top)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Top)
    glVertex3f(-1.0, 1.0, 1.0);		# Bottom Left Of The Quad (Top)
    glVertex3f( 1.0, 1.0, 1.0);		# Bottom Right Of The Quad (Top)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0,-1.0, 1.0);		# Top Right Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0, 1.0);		# Top Left Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Bottom)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Bottom)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0, 1.0);		# Top Right Of The Quad (Front)
    glVertex3f(-1.0, 1.0, 1.0);		# Top Left Of The Quad (Front)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Front)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Front)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Back)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Back)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Right Of The Quad (Back)
    glVertex3f( 1.0, 1.0,-1.0);		# Top Left Of The Quad (Back)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f(-1.0, 1.0, 1.0);		# Top Right Of The Quad (Left)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Left)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Right)
    glVertex3f( 1.0, 1.0, 1.0);		# Top Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Right)
    glEnd();				# Done Drawing The Quad

####################################################
# printf()  
####################################################

def printf(format, *args):
    sys.stdout.write(format % args)

################################################################################
# start the app

def rotate3(vector, theta):
    rotz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
    roty = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]])
    rotx = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]])
    return np.matmul(rotx, np.matmul(roty, np.matmul(rotz, vector)))

print ("Hit ESC key to quit.")
main()
        
