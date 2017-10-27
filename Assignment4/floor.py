from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

#  from pyquaternion import Quaternion    ## would be useful for 3D simulation
import numpy as np

window = 0     # number of the glut window
theta = 0
simTime = 0
dT = 0.0003 
simRun = True
RAD_TO_DEG = 180.0/3.14160
init_angle1 = 3.1416 / 4
init_angle3 = 0.0
xoffset = np.sin(init_angle1)
yoffset = -np.cos(init_angle1)
init_cpoint1 = np.array([-1.0 * np.sin(init_angle1) / 2, 1 * np.cos(init_angle1) / 2, 0])
k_d = 0.2
floor_posy = -3

#####################################################
#### Link class, i.e., for a rigid body
#####################################################

class Link:
    color=[0,0,0]    ## draw color
    size=[1,1,1]     ## dimensions
    mass = 1.0       ## mass in kg
    izz = 1.0        ## moment of inertia about z-axis
    theta=0          ## 2D orientation  (will need to change for 3D)
    omega=0          ## 2D angular velocity
    posn=np.array([0.0,0.0,0.0])     ## 3D position (keep z=0 for 2D)
    vel=np.array([0.0,0.0,0.0])      ## initial velocity
    def draw(self):      ### steps to draw a link
        glPushMatrix()                                            ## save copy of coord frame
        glTranslatef(self.posn[0], self.posn[1], self.posn[2])    ## move 
        glRotatef(self.theta*RAD_TO_DEG,  0,0,1)                             ## rotate
        glScale(self.size[0], self.size[1], self.size[2])         ## set size
        glColor3f(self.color[0], self.color[1], self.color[2])    ## set colour
        DrawCube()                                                ## draw a scaled cube
        glPopMatrix()                                             ## restore old coord frame

#####################################################
#### main():   launches app
#####################################################

def main():
    global window
    global link1, link2, link3, link4, floor
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
    link4 = Link()
    floor = Link()
    resetSim()
    
    glutMainLoop()                   # start event processing loop

#####################################################
#### keyPressed():  called whenever a key is pressed
#####################################################

def resetSim():
    global link1, link2, link3, link4, floor
    global simTime, simRun, xoffset, yoffset

    printf("Simulation reset\n")
    simRun = True
    simTime = 0

    link1.size=[0.04, 1.0, 0.12]
    link1.color=[1,0.9,0.9]
    link1.posn=np.array([0.0,0.0,0.0])
    link1.vel=np.array([0.0,0.0,0.0])
    link1.theta = init_angle1
    link1.omega = 0        ## radians per second
    link1.izz = 1.0 / 3

    link2.size=[0.04, 1.0, 0.12]
    link2.color=[1,0.9,0.9]
    link2.posn=np.array([xoffset,yoffset,0.0])
    link2.vel=np.array([0.0,0.0,0.0])
    link2.theta = init_angle1
    link2.omega = 0        ## radians per second
    link2.izz = 1.0 / 3

    link3.size=[0.04, 1.0, 0.12]
    link3.color=[1,0.9,0.9]
    link3.posn=np.array([2*xoffset,2*yoffset,0.0])
    link3.vel=np.array([0.0,0.0,0.0])
    link3.theta = init_angle1
    link3.omega = 0        ## radians per second
    link3.izz = 1.0 / 3

    link4.size=[0.04, 1.0, 0.12]
    link4.color=[1,0.9,0.9]
    link4.posn=np.array([3*xoffset,3*yoffset,0.0])
    link4.vel=np.array([0.0,0.0,0.0])
    link4.theta = init_angle1
    link4.omega = 0        ## radians per second
    link4.izz = 1.0 / 3

    floor.size=[5.0, 0.1, 5.0]
    floor.color=[1,1,1]
    floor.posn=np.array([0.0, floor_posy, 0.0])
    floor.vel=np.array([0.0, 0.0, 0.0])
    floor.theta = 0
    floor.omega = 0
    floor.izz = 100

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
    global link1, link2, link3, link4, floor

    if (simRun==False):             ## is simulation stopped?
        return

    #### solve for the equations of motion (simple in this case!)
    #acc1 = np.array([0, -10, 0])       ### linear acceleration = [0, -G, 0]
    #omega_dot1 = 0.0                 ### assume no angular acceleration

    ####  for the constrained one-link pendulum, and the 4-link pendulum,
    ####  you will want to build the equations of motion as a linear system, and then solve that.
    ####  Here is a simple example of using numpy to solve a linear system.
    m = link1.mass
    I = link1.izz
    r1x = 1.0 * np.sin(link1.theta)/2
    r1y = -1.0 * np.cos(link1.theta)/2
    r2x = 1.0 * np.sin(link2.theta)/2
    r2y = -1.0 * np.cos(link2.theta)/2
    r3x = 1.0 * np.sin(link3.theta)/2
    r3y = -1.0 * np.cos(link3.theta)/2
    r4x = 1.0 * np.sin(link4.theta)/2
    r4y = -1.0 * np.cos(link4.theta)/2

    w1 = link1.omega
    w2 = link2.omega
    w3 = link3.omega
    w4 = link4.omega

    z = 0.0
    k = -1.0
    o = 1.0

    # Floor contact forces
    endpoint4 = (link4.posn + np.array([r4x, r4y, 0]))[1]
    endpoint4_vel = (link4.vel + np.cross(np.array([0, 0, link4.omega]), np.array([-r4x, -r4y, 0])))[1]
    
    Ffloor = 0
    if ((endpoint4 + endpoint4_vel*dT) < floor_posy):
        Ffloor = max(0*(floor_posy - endpoint4) - 200*endpoint4_vel, 0)
    if (endpoint4 < floor_posy):
        Ffloor = max(1000*(floor_posy - endpoint4) - 10*endpoint4_vel, 0)
    

    #### Two-link case:       1   ,  2,     3   ,  4  ,  5  ,  6  ,  7  ,  8  ,    9  ,   10 ,  11 ,  12 ,  13 ,  14 ,  15  , 16  ,  17  ,  18
    ####                x = [p1x__, p1y__, p1z__, w1x_, w1y_, w1z_, p2x__, p2y__, p2z__, w2x_, w2y_, w2z_, F1cx, F1cy, F1cz, F12cx, F12cy, F12cz]
    a = np.array([[m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    k,     z,   z,    k,       z,      z,      z,      z,      z,     z,      z,      z],
                  [z, m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     k,   z,    z,       k,      z,      z,      z,      z,     z,      z,      z],
                  [z, z, m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   k,    z,       z,      k,      z,      z,      z,     z,      z,      z],
                  [z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   r1y,  z,       z,      -r1y,   z,      z,      z,     z,      z,      z],
                  [z, z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   -r1x, z,       z,      r1x,    z,      z,      z,     z,      z,      z],
                  [z, z, z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    -r1y,  r1x, z,    r1y,    -r1x,    z,      z,      z,      z,     z,      z,      z],

                  [z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    o,       z,      z,      k,      z,      z,     z,      z,      z],
                  [z, z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       o,      z,      z,      k,      z,     z,      z,      z],
                  [z, z, z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      o,      z,      z,      k,     z,      z,      z],
                  [z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      -r2y,   z,      z,      -r2y,  z,      z,      z],
                  [z, z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      r2x,    z,      z,      r2x,   z,      z,      z],
                  [z, z, z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    r2y,     -r2x,   z,      r2y,    -r2x,   z,     z,      z,      z],

                  [z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      o,      z,      z,     k,      z,      z],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      o,      z,     z,      k,      z],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      o,     z,      z,      k],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      -r3y,  z,      z,     -r3y],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      r3x,   z,      z,      r3x],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I, z, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      r3y,   -r3x,    z,     r3y,    -r3x,   z],
                                                                                                                                                                                
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      z,     o,      z,      z],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      z,     z,      o,      z],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, m, z, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      z,     z,      z,      o],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I, z, z,    z,     z,   z,    z,       z,      z,      z,      z,      z,     z,      z,      -r4y],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I, z,    z,     z,   z,    z,       z,      z,      z,      z,      z,     z,      z,      r4x],
                  [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, I,    z,     z,   z,    z,       z,      z,      z,      z,      z,     r4y,    -r4x,   z],

                  [k, z, z, z, z, -r1y,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, k, z, z, z, r1x,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, k, r1y, -r1x, z,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],

                  [k, z, z, z, z, r1y,   o, z, z, z, z, r2y,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, k, z, z, z, -r1x,   z, o, z, z, z, -r2x,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, k, -r1y, r1x, z,   z, z, o, -r2y, r2x, z,   z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  
                  [z, z, z, z, z, z,   k, z, z, z, z, r2y,   o, z, z, z, z, r3y,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, z, z, z, z,   z, k, z, z, z, -r2x,   z, o, z, z, z, -r3x,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, z, z, z, z,   z, z, k, -r2y, r2x, z,   z, z, o, -r3y, r3x, z,   z, z, z, z, z, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  
                  [z, z, z, z, z, z,   z, z, z, z, z, z,   k, z, z, z, z, r3y,   o, z, z, z, z, r4y,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, z, z, z, z,   z, z, z, z, z, z,   z, k, z, z, z, -r3x,   z, o, z, z, z, -r4x,   z, z, z,  z, z, z,  z, z, z,  z, z, z],
                  [z, z, z, z, z, z,   z, z, z, z, z, z,   z, z, k, -r3y, r3x, z,   z, z, o, -r4y, r4x, z,   z, z, z,  z, z, z,  z, z, z,  z, z, z]
                  ])

    b = np.array([z, -10.0, z, z, z, -k_d*w1,   z, -10.0, z, z, z, -k_d*w2,   z, -10.0, z, z, z, -k_d*w3,   z, -10.0+Ffloor, z, z, z, -k_d*w4 + np.sin(link4.theta)*0.1*Ffloor,   w1*w1*r1x, w1*w1*r1y, z,   -w1*w1*r1x-(w2*w2*r2x), -w1*w1*r1y-(w2*w2*r2y), z,   -w2*w2*r2x-(w3*w3*r3x), -w2*w2*r2y-(w3*w3*r3y), z,   -w3*w3*r3x-(w4*w4*r4x), -w3*w3*r3y-(w4*w4*r4y), z])
    x = np.linalg.solve(a, b)
    #print(x)   # [ -2.17647059  53.54411765  56.63235294]
    acc1 = np.array([x[0], x[1], x[2]])
    omega_dot1 = x[5]
    acc2 = np.array([x[6], x[7], x[8]])
    omega_dot2 = x[11]
    acc3 = np.array([x[12], x[13], x[14]])
    omega_dot3 = x[17]
    acc4 = np.array([x[18], x[19], x[20]])
    omega_dot4 = x[23]
    # print(acc1)

    # Constraint things
    cpoint1 = link1.posn - np.array([r1x, r1y, 0])
    cvelocity1 = (link1.vel + np.cross(np.array([0, 0, link1.omega]), np.array([-r1x, -r1y, 0])))
    factor1 = 100*(cpoint1 - init_cpoint1) + 100*(cvelocity1)
    acc1 -= factor1

    cpoint2 = link2.posn - np.array([r2x, r2y, 0])
    cdiff2 = cpoint2 - (link1.posn + np.array([r1x, r1y, 0]))
    cvelocity2 = (link2.vel + np.cross(np.array([0, 0, link2.omega]), np.array([-r2x, -r2y, 0]))) - (link1.vel + np.cross(np.array([0, 0, link1.omega]), np.array([r1x, r1y, 0])))
    factor2 = 100*(cdiff2) + 100*(cvelocity2)
    acc2 = acc2 - factor2 + acc1

    cpoint3 = link3.posn - np.array([r3x, r3y, 0])
    cdiff3 = cpoint3 - (link2.posn + np.array([r2x, r2y, 0]))
    cvelocity3 = (link3.vel + np.cross(np.array([0, 0, link3.omega]), np.array([-r3x, -r3y, 0]))) - (link2.vel + np.cross(np.array([0, 0, link2.omega]), np.array([r2x, r2y, 0])))
    factor3 = 100*(cdiff3) + 100*(cvelocity3)
    acc3 = acc3 - factor3 + acc2

    cpoint4 = link4.posn - np.array([r4x, r4y, 0])
    cdiff4 = cpoint4 - (link3.posn + np.array([r3x, r3y, 0]))
    cvelocity4 = (link4.vel + np.cross(np.array([0, 0, link4.omega]), np.array([-r4x, -r4y, 0]))) - (link3.vel + np.cross(np.array([0, 0, link3.omega]), np.array([r3x, r3y, 0])))
    factor4 = 100*(cdiff4) + 100*(cvelocity4)
    acc4 = acc4 - factor4 + acc3

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

    link4.posn += link4.vel*dT
    link4.vel += acc4*dT
    link4.theta += link4.omega*dT
    link4.omega += omega_dot4*dT

    # print("{} / {} / {}".format(link1.posn, link1.vel, acc1))

    simTime += dT

    #### draw the updated state
    DrawWorld()
    printf("simTime=%.2f\n",simTime)

#####################################################
#### DrawWorld():  draw the world
#####################################################

def DrawWorld():
    global link1, link2, link3, link4, floor

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	# Clear The Screen And The Depth Buffer
    glLoadIdentity();
    gluLookAt(3,3,9,  0,0,0,  0,1,0)

    DrawOrigin()
    link1.draw()
    link2.draw()
    link3.draw()
    link4.draw()

    floor.draw()

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

    glColor3f(1,0.0,0.0)   ## light red x-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(1,0,0)
    glEnd()

    glColor3f(0.0,1,0.0)   ## light green y-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,1,0)
    glEnd()

    glColor3f(0.0,0.0,1)   ## light blue z-axis
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

print ("Hit ESC key to quit.")
main()
        
